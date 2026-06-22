#%%
import xarray as xr
import numpy as np
import os
import dask.array as da
from functools import partial
from constants import HANSEN_DIR, PERCENTILES_DIR, STORAGE_DIR
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
from contextlib import contextmanager
from numba import njit
import gc
import warnings
import pickle

USE_DASK_DASHBOARD = True
INDEPEDENT_VARIABLE = 'heatwaves_smoothed'  # or 'heatwaves_smoothed'
DEPENDENT_VARIABLE = 'cc_loss'  # or 'cc_loss'
MWA_DAILY_DIR = os.path.join(STORAGE_DIR, 'moving_window_analysis_daily')


@contextmanager
def local_cluster(n_workers=16, threads_per_worker=1, memory_limit='20GB',
                  suppress_dask_warnings=False):

    if suppress_dask_warnings:
        warnings.filterwarnings(
            "ignore", 
            message="Sending large graph of size.*", 
            category=UserWarning, 
            module="distributed.client"
        )

    if USE_DASK_DASHBOARD:
        with LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker,
                        memory_limit=memory_limit, dashboard_address=':7878') as cluster:
            with Client(cluster) as client:
                print(f'LocalCluster started: {client.dashboard_link}')
                yield client
    else:
        yield None


def _mwa_kernel_time(block, window_radius):
    """
    block: (2, T, H+2wr, W+2wr) — channel 0 = independent var, channel 1 = dependent var.
    Returns same shape, with results embedded in centre for each timestep.
    """

    ind_block = block[0].astype(np.float32)  # (T, H+2wr, W+2wr)
    dep_block = block[1].astype(np.float32)  # (T, H+2wr, W+2wr)

    ind = np.where(ind_block == -9999.0, np.nan, ind_block)
    dep = np.where(dep_block == -9999.0, np.nan, dep_block)
    # dep = dep[0]  # (H+2wr, W+2wr)

    wr = window_radius

    t_size = ind.shape[0]
    h_size = ind.shape[1] - 2 * wr
    w_size = ind.shape[2] - 2 * wr

    out = np.full((2, t_size, h_size, w_size), np.nan, dtype=np.float32)

    center_ind = ind[:, wr:wr+h_size, wr:wr+w_size]
    center_dep = dep[:, wr:wr+h_size, wr:wr+w_size]

    max_ind_diff = np.full((t_size, h_size, w_size), -np.inf, dtype=np.float32)
    max_dep_diff = np.zeros((1, h_size, w_size), dtype=np.float32)
    has_higher = np.zeros((1, h_size, w_size), dtype=np.bool_)

    for di in range(-wr, wr + 1):
        for dj in range(-wr, wr + 1):
            if di == 0 and dj == 0:
                continue
            ri, rj = wr + di, wr + dj
            
            n_ind = ind[:, ri:ri+h_size, rj:rj+w_size]
            n_dep = dep[:, ri:ri+h_size, rj:rj+w_size]
            dep_diff = n_dep - center_dep
            ind_diff = n_ind - center_ind
            
            better = dep_diff > max_dep_diff
            max_dep_diff = np.where(better, dep_diff, max_dep_diff)
            max_ind_diff = np.where(better, ind_diff, max_ind_diff)

            has_higher = has_higher | (n_dep > center_dep)

    valid = (center_ind >= 0)  & ~np.isnan(center_dep) & has_higher


    out[0] = np.where(valid, max_ind_diff, np.nan)
    out[1] = np.where(valid, max_dep_diff, np.nan)

    out[0] = np.nan_to_num(out[0], nan=-9999, posinf=-9999, neginf=-9999)
    out[1] = np.nan_to_num(out[1], nan=-9999, posinf=-9999, neginf=-9999)
    return out.astype(np.int16)

def moving_window_analysis_daily(ds, window_radius, ind_var_name, dep_var_name):
    """Daily moving-window analysis on (time, lat, lon) inputs."""
    ind_arr = ds[ind_var_name].data
    dep_arr = ds[dep_var_name].data

    dep_arr = da.broadcast_to(dep_arr, ind_arr.shape)

    year = ds.time.dt.year.values[0]

    stacked = da.stack([ind_arr, dep_arr]).rechunk({0: -1})
    stacked = stacked.persist()
    
    # block = stacked.blocks[:,:,9,9]
    result = da.map_overlap(
        partial(_mwa_kernel_time, window_radius=window_radius),
        stacked,
        depth={0: 0, 1: 0, 2: window_radius, 3: window_radius},
        boundary=np.nan,
        trim=False,
        dtype=np.int16,
    )

    ind_ds = xr.Dataset(
    data_vars={
        f'{ind_var_name}_cdiff': (("time", "lat", "lon"), result[0]),
    },
    coords={
        "time": ds.coords["time"],
        "lat": ds.coords["lat"],
        "lon": ds.coords["lon"]
    }
    )

    dep_ds = xr.Dataset(
    data_vars={
        f'{dep_var_name}_cdiff': (("year", "lat", "lon"), result[1, 0][None]),
    },
    coords={
        "year": [year],
        "lat": ds.coords["lat"],
        "lon": ds.coords["lon"]
    }
    )

    return ind_ds, dep_ds


def calc_moving_window_daily(ind_var, dep_var, client=None,
                             coarsen_factor=None, window_radius=1):
    
    ind_var_name = ind_var.name
    dep_var_name = dep_var.name


    sname = f'{dep_var_name}_vs_{ind_var_name}_daily'
    if coarsen_factor:
        sname += f'_coarsened{coarsen_factor}'


    elev = xr.open_dataset(os.path.join(STORAGE_DIR, '..', 'elevation', 'elevation_regridded.nc'))['elevation']
    elev = elev.assign_coords(lat=ind_var.lat, lon=ind_var.lon)
    ind_var = ind_var.where(elev < 1000)
    dep_var = dep_var.where(elev < 1000)

    if coarsen_factor:
        ind_var = ind_var.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary='trim').mean()
        dep_var = dep_var.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary='trim').mean()

    years_daily = np.unique(ind_var.time.dt.year.values.astype(np.int32))


    os.makedirs(MWA_DAILY_DIR, exist_ok=True)
    out_path_ind = os.path.join(MWA_DAILY_DIR, f'{sname}_mwa_results_ind.zarr')
    out_path_dep = os.path.join(MWA_DAILY_DIR, f'{sname}_mwa_results_dep.zarr')

    write_mode = 'w'
    for year in years_daily:
        print(f"Running moving window analysis for year {year}...")
        ind_year = ind_var.sel(time=ind_var.time.dt.year == year)
        dep_year = dep_var.sel(time=dep_var.time.dt.year == year)

        dep_year = dep_year.rename({'time': 'year'})
        dep_year = dep_year.assign_coords(year=[year])
        ds_year = xr.Dataset({ind_var_name: ind_year, dep_var_name: dep_year})


        ind_ds, dep_ds = moving_window_analysis_daily(
            ds_year,
            window_radius=window_radius,
            ind_var_name=ind_var_name,
            dep_var_name=dep_var_name,
        )

        if write_mode == 'w':
            ind_ds.to_zarr(out_path_ind, mode='w', consolidated=False)
            dep_ds.to_zarr(out_path_dep, mode='w', consolidated=False)
            write_mode = 'a'
        else:
            ind_ds.to_zarr(out_path_ind, mode='a', append_dim='time', consolidated=False)
            dep_ds.to_zarr(out_path_dep, mode='a', append_dim='year', consolidated=False)

        if client is not None:
            client.cancel(ind_ds)
            client.cancel(dep_ds)
        gc.collect()
        print(f"Year {year} results written to {out_path_ind} and {out_path_dep}")
  
def calculate_binned_stats(dep_var_name, ind_var_name, coarsen_factor=None, sname=None):

    ind = xr.open_zarr(os.path.join(MWA_DAILY_DIR, f'{sname}_mwa_results_ind.zarr'), consolidated=False)[ind_var_name+'_cdiff']
    # cast to float32 and replace -9999 with NaN
    ind = ind.astype(np.float32).where(ind != -9999.0)
    dep = xr.open_zarr(os.path.join(MWA_DAILY_DIR, f'{sname}_mwa_results_dep.zarr'), consolidated=False)[dep_var_name+'_cdiff']
    dep = dep.astype(np.float32).where(dep != -9999.0)


    bins = np.arange(10, 110, 10)
    stats_dss = []
    for year in dep.year.values:
        
        ind_year = ind.sel(time=ind.time.dt.year == year)
        dep_year = dep.sel(year=year)

        binned = ind_year.groupby_bins(dep_year, bins=bins)

        stats = xr.Dataset({
            'sum': binned.sum(),
            'count': binned.count(),
            'sumsq': (ind_year**2).groupby_bins(dep_year, bins=bins).sum()
        })
        stats = stats.drop_vars('year')
        stats_dss.append(stats)

    stats_ds = xr.concat(stats_dss, dim='time', join='outer', coords='minimal')
    stats_summed = stats_ds.sum('time')
    stats_summed['mean'] = stats_summed['sum'] / stats_summed['count']
    stats_summed['std'] = np.sqrt(stats_summed['sumsq'] / stats_summed['count'] - stats_summed['mean']**2)

    stats_computed = stats_summed.compute()


    pickle_path = os.path.join(MWA_DAILY_DIR, f'{sname}_binned_stats.P')
    print(f"Saving binned statistics to as pickle {pickle_path}...")
    with open(pickle_path, 'wb') as f:
        pickle.dump(stats_computed, f)

def load_ind_dep_data(ind_var_name, dep_var_name):

    if dep_var_name == 'cc_loss':
        ds = xr.open_zarr(os.path.join(HANSEN_DIR, 'hansen_tc_2000_2024_rechunked.zarr'))
        tc = ds['tree_cover'].loc[dict(time=slice('2003', '2020'))]
        tc = tc.assign_coords(time=xr.date_range(start='2003-01-01', end='2020-12-31', 
                                                freq='YS', calendar='noleap', use_cftime=True))
        dep_var = -(tc - tc[0]).round()
        dep_var.name = dep_var_name
    else:
        raise ValueError(f"Unsupported dependent variable name: {dep_var_name}")

    if ind_var_name == 'tmax':
        ind_var = xr.open_zarr(os.path.join(STORAGE_DIR, 'rechunked', 'zhang_zhou_regridded_rechunked.zarr'))['Ta_TMAX']
        ind_var.name = ind_var_name
    elif ind_var_name == 'heatwaves_smoothed':
        ind_var = xr.open_zarr(os.path.join(STORAGE_DIR, 'heatdays', 'heatwaves_zz_2003-2020_smooth.zarr'))['heatwaves']
        ind_var.name = ind_var_name
    else:
        raise ValueError(f"Unsupported independent variable name: {ind_var_name}")

    dep_var = dep_var.assign_coords(lat=ind_var.lat, lon=ind_var.lon)
    return ind_var, dep_var


def run_mwa_daily(coarsen_factor=None, window_radius=1, client=None):

    ind_var, dep_var = load_ind_dep_data(ind_var_name=INDEPEDENT_VARIABLE, dep_var_name=DEPENDENT_VARIABLE)

    # check if output zarr already exists
    sname = f'{dep_var.name}_vs_{ind_var.name}_daily'
    if coarsen_factor:
        sname += f'_coarsened{coarsen_factor}'
    out_path_ind = os.path.join(MWA_DAILY_DIR, f'{sname}_mwa_results_ind.zarr')
    out_path_dep = os.path.join(MWA_DAILY_DIR, f'{sname}_mwa_results_dep.zarr')
    if not os.path.exists(out_path_ind) or not os.path.exists(out_path_dep):
        print(f"Output zarrs do not exist for {sname}. Proceeding with moving window analysis.")
        calc_moving_window_daily(ind_var=ind_var, dep_var=dep_var, client=client, window_radius=window_radius,
                                coarsen_factor=coarsen_factor)
        print(f"Moving window analysis completed for {dep_var.name} vs {ind_var.name} with coarsen factor {coarsen_factor}.")
    else:
        print(f"Output zarrs already exist for {sname}. Skipping moving window analysis.")


    if not os.path.exists(os.path.join(MWA_DAILY_DIR, f'{sname}_binned_stats.P')):
        
        print(f"Calculating binned statistics for {dep_var.name} vs {ind_var.name} with coarsen factor {coarsen_factor}.")

        calculate_binned_stats(dep_var_name=dep_var.name, ind_var_name=ind_var.name,
                            coarsen_factor=coarsen_factor, sname=sname)
        print(f"Binned statistics calculation completed for {dep_var.name} vs {ind_var.name} with coarsen factor {coarsen_factor}.")
    else:
        print(f"Binned statistics already exist for {dep_var.name} vs {ind_var.name} with coarsen factor {coarsen_factor}. Skipping calculation.")

def run_mwa_coarsen_loop(client=None):
        for coarsen_factor in [None, 2, 5, 10, 25]:#[None, 2, 4, 8, 16]:
            run_mwa_daily(coarsen_factor=coarsen_factor, window_radius=1, client=client)
            
#%%
if __name__ == "__main__":
    
    with local_cluster(n_workers=6, threads_per_worker=6, memory_limit='100GB',
                       suppress_dask_warnings=True) as client:
        run_mwa_coarsen_loop(client=client)


#%%

# # open pickle to check
# with open(os.path.join(MWA_DAILY_DIR, f'{dep_var.name}_vs_{ind_var.name}_daily_binned_stats.P'), 'rb') as f:
#     stats_summed = pickle.load(f)