#%%
import rioxarray as rxr
import xarray as xr
import pandas as pd
import numpy as np
import os
import dask.array as da
from functools import partial
from ufuncs import load_heat
from constants import HANSEN_DIR, PERCENTILES_DIR, STORAGE_DIR
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
from contextlib import contextmanager
from numba import njit

USE_DASK_DASHBOARD = True
MWA_DAILY_DIR = STORAGE_DIR + 'moving_window_analysis_daily/'


@contextmanager
def local_cluster(n_workers=16, threads_per_worker=1, memory_limit='20GB'):

    if USE_DASK_DASHBOARD:
        with LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker,
                        memory_limit=memory_limit, dashboard_address=':7878') as cluster:
            with Client(cluster) as client:
                print(f'LocalCluster started: {client.dashboard_link}')
                yield client
    else:
        yield None

def open_hansen_loss(start_year, end_year):
    ds = xr.open_zarr(HANSEN_DIR + 'hansen_tc_2000_2024_rechunked.zarr')
    tc = ds['tree_cover']
    tc = tc.sel(time=slice(str(start_year), str(end_year)))
    loss = tc.diff('time').sum('time')
    loss = -loss
    loss = loss.assign_coords(lat=np.round(loss.lat, 2), lon=np.round(loss.lon, 2))
    loss = loss.drop_vars('spatial_ref')
    return loss

def load_tmax_trends(mask_insignificant=True):
    dpath = os.path.join(PERCENTILES_DIR, 'zz_tmax_ts_trend.zarr')
    ds = xr.open_zarr(dpath)
    trends = ds['Ta_TMAX_trend']
    if mask_insignificant:
        pvals = ds['Ta_TMAX_trend_pvalue']
        trends = trends.where(pvals < 0.05)
    return trends


def calculate_binned_stats(results, 
                           dep_var_name='heatwaves_trend',
                           bin_width=10):
    """Calculate binned statistics and save to CSV.
    
    Parameters
    ----------
    results : xr.Dataset
        Moving window analysis results with 'cc_loss' and '{dep_var_name}_diff'.
    csv_path : str
        Path to save the CSV file with bin statistics.
    dep_var_name : str
        Name of the dependent variable (e.g., 'heatwaves_trend').
    bin_width : int
        Width of each bin in percent.
    """

    # flatted and combine zarrs in results list
    combined = xr.concat(results, dim='zarr_index')
    combined = combined.stack(points=('lat', 'lon', 'zarr_index'))
    combined = combined.dropna(dim='points')


    # Auto-detect max bin limit: find last bin with n > 3 observations
    preliminary_bins = np.arange(0, 110, bin_width)
    preliminary_counts = combined[f'{dep_var_name}_diff'].groupby_bins(combined['cc_loss'], preliminary_bins).count().compute()
    
    # Find the last bin with count > 3
    valid_bins = preliminary_counts[preliminary_counts > 3]
    valid_bins_pd = valid_bins.to_pandas()
    if len(valid_bins_pd) > 0:
        last_bin_upper = valid_bins_pd.index[-1].right
        max_bin_limit = last_bin_upper + bin_width  # Include one more bin
    else:
        max_bin_limit = bin_width  # Fallback if no bins have n > 3
    
    # Create final bins and compute statistics
    bins = np.arange(0, max_bin_limit, bin_width)
    binned = combined[f'{dep_var_name}_diff'].groupby_bins(combined['cc_loss'], bins).mean() 
    bin_counts = combined[f'{dep_var_name}_diff'].groupby_bins(combined['cc_loss'], bins).count()
    bin_stds = combined[f'{dep_var_name}_diff'].groupby_bins(combined['cc_loss'], bins).std()
    bin_sems = bin_stds / np.sqrt(bin_counts)
    bin_means = binned.values
    ci_lower = bin_means - 1.96 * bin_sems
    ci_upper = bin_means + 1.96 * bin_sems
    
    # Save the bin means, counts and CIs to a CSV for later use
    bin_dim = next(dim for dim in binned.dims if dim.endswith('_bins'))
    bin_stats = pd.DataFrame({
        'bin': [f'{int(interval.left)}-{int(interval.right)}' for interval in binned[bin_dim].values],
        'mean_diff': bin_means,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'count': bin_counts.values
    })
    return bin_stats

def slice_elev_mask(da):
    westlimit=-81.8; southlimit=-17.7; eastlimit=-41.6; northlimit=12.9
    da = da.sel(lat=slice(southlimit, northlimit), lon=slice(westlimit, eastlimit))
    elev = xr.open_dataset(STORAGE_DIR + '../elevation/elevation_regridded.nc')['elevation']
    elev = elev.reindex_like(da, method='nearest')
    da_elevmask = da.where(elev<1000)
    return da_elevmask


#@njit(parallel=False)
def _mwa_kernel_time(block, window_radius):
    """
    block: (2, T, H+2wr, W+2wr) — channel 0 = independent var, channel 1 = dependent var.
    Returns same shape, with results embedded in centre for each timestep.
    """

    ind_block = block[0]  # (T, H+2wr, W+2wr)
    dep_block = block[1]  # (T, H+2wr, W+2wr)
    
    ind = np.where(ind_block == -9999.0, np.nan, ind_block.astype(np.float32))
    dep = np.where(dep_block == -9999.0, np.nan, dep_block.astype(np.float32))
    # dep = dep[0]  # (H+2wr, W+2wr)

    wr = window_radius

    t_size = ind.shape[0]
    h_size = ind.shape[1] - 2 * wr
    w_size = ind.shape[2] - 2 * wr

    out = np.full((2, t_size, h_size, w_size), np.nan, dtype=np.float32)

    center_ind = ind[:, wr:wr+h_size, wr:wr+w_size]
    center_dep = dep[:, wr:wr+h_size, wr:wr+w_size]

    max_ind_diff = np.full((t_size, h_size, w_size), -np.inf, dtype=ind.dtype)
    max_dep_diff = np.zeros((1, h_size, w_size), dtype=dep.dtype)
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
    return out

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
        dtype=np.float32,
    )

    results = xr.Dataset(
    data_vars={
        "ind_var": (("time", "lat", "lon"), result[0]),
        "dep_var": (("year", "lat", "lon"), result[1, 0][None, :, :])
    },
    coords={
        "time": ds.coords["time"],
        "year": [year],
        "lat": ds.coords["lat"],
        "lon": ds.coords["lon"]
    }
    )

    return results


def calc_moving_window_daily(ind_var, dep_var,
                             coarsen_factor=None, window_radius=1):
    
    ind_var_name = ind_var.name
    dep_var_name = dep_var.name


    sname = f'{dep_var_name}_vs_{ind_var_name}_daily'
    if coarsen_factor:
        sname += f'_coarsened{coarsen_factor}'


    elev = xr.open_dataset(STORAGE_DIR + '../elevation/elevation_regridded.nc')['elevation']
    elev = elev.assign_coords(lat=ind_var.lat, lon=ind_var.lon)
    ind_var = ind_var.where(elev < 1000)
    dep_var = dep_var.where(elev < 1000)

    if coarsen_factor:
        ind_var = ind_var.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary='trim').mean()
        dep_var = dep_var.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary='trim').mean()

    years_daily = np.unique(ind_var.time.dt.year.values.astype(np.int32))


    os.makedirs(MWA_DAILY_DIR, exist_ok=True)
    out_path = os.path.join(MWA_DAILY_DIR, f'{sname}_mwa_results.zarr')

    write_mode = 'w'
    for year in years_daily:
        print(f"Running moving window analysis for year {year}...")
        ind_year = ind_var.sel(time=ind_var.time.dt.year == year)
        dep_year = dep_var.sel(time=dep_var.time.dt.year == year)

        dep_year = dep_year.rename({'time': 'year'})
        dep_year = dep_year.assign_coords(year=[year])
        ds_year = xr.Dataset({ind_var_name: ind_year, dep_var_name: dep_year})


        results_year = moving_window_analysis_daily(
            ds_year,
            window_radius=window_radius,
            ind_var_name=ind_var_name,
            dep_var_name=dep_var_name,
        )

        if write_mode == 'w':
            results_year.to_zarr(out_path, mode='w', consolidated=True)
            write_mode = 'a'
        else:
            results_year.to_zarr(out_path, mode='a', append_dim='time', consolidated=True)
        print(f"Results for year {year} written to {out_path}.")
    #return xr.open_zarr(out_path, consolidated=True)

# def run_mwa_coarsen_loop(ind_var, dep_var, ind_var_name,dep_var_name):
#         dep_var = load_tmax_trends(mask_insignificant=False)
#         for coarsen_factor in [None, 2, 4, 8, 16]:
#             print(f"Running moving window analysis with coarsen_factor={coarsen_factor}...")
#             calc_moving_window(ind_var=ind_var, dep_var=dep_var, ind_var_name=ind_var_name, dep_var_name=dep_var_name,
#                                 coarsen_factor=coarsen_factor, coarsen_offset_resample=True)
            
def load_heatwave_diffs():
    heatwaves = xr.open_zarr(STORAGE_DIR + 'heatdays/heatwaves_zz_2003-2020.zarr', consolidated=True)['heatwaves']
    heatwaves_smoothed = xr.open_zarr(STORAGE_DIR + 'heatdays/heatwaves_zz_2003-2020_smooth.zarr', consolidated=True)['heatwaves']
    ds = xr.Dataset({'d_heatwaves': heatwaves, 'd_heatwaves_smooth': heatwaves_smoothed})
    annual = ds.resample(time='YE').sum()
    heatwaves_diff = annual.isel(time=slice(-3, None)).sum('time') - annual.isel(time=slice(0, 3)).sum('time')
    return heatwaves_diff['d_heatwaves'], heatwaves_diff['d_heatwaves_smooth']

#%%
if __name__ == "__main__":
    
    with local_cluster(n_workers=4, threads_per_worker=4, memory_limit='100GB') as client:


        ds = xr.open_zarr(HANSEN_DIR + 'hansen_tc_2000_2024_rechunked.zarr')
        tc = ds['tree_cover'].loc[dict(time=slice('2003', '2020'))]
        tc = tc.assign_coords(time=xr.date_range(start='2003-01-01', end='2020-12-31', 
                                                freq='YS', calendar='noleap', use_cftime=True))
        dep_var = -(tc - tc[0]).round()
        dep_var.name = 'cc_loss'



        ind_var = xr.open_zarr(STORAGE_DIR + 'rechunked/zhang_zhou_regridded_rechunked.zarr')['Ta_TMAX']
        ind_var.name = 'tmax'

        dep_var = dep_var.assign_coords(lat=ind_var.lat, lon=ind_var.lon)

        calc_moving_window_daily(ind_var=ind_var, dep_var=dep_var, window_radius=1)

#%%

# results = xr.open_zarr(os.path.join(MWA_DAILY_DIR, 'cc_loss_vs_tmax_daily_mwa_results.zarr'), consolidated=True)

# combined = xr.concat(results, dim='zarr_index')
# combined = combined.stack(points=('lat', 'lon', 'zarr_index'))
# combined = combined.dropna(dim='points')


# # Auto-detect max bin limit: find last bin with n > 3 observations
# preliminary_bins = np.arange(0, 110, bin_width)
# preliminary_counts = combined[f'{dep_var_name}_diff'].groupby_bins(combined['cc_loss'], preliminary_bins).count().compute()

# # Find the last bin with count > 3
# valid_bins = preliminary_counts[preliminary_counts > 3]
# valid_bins_pd = valid_bins.to_pandas()
# if len(valid_bins_pd) > 0:
#     last_bin_upper = valid_bins_pd.index[-1].right
#     max_bin_limit = last_bin_upper + bin_width  # Include one more bin
# else:
#     max_bin_limit = bin_width  # Fallback if no bins have n > 3

# # Create final bins and compute statistics
# bins = np.arange(0, max_bin_limit, bin_width)
# binned = combined[f'{dep_var_name}_diff'].groupby_bins(combined['cc_loss'], bins).mean() 
# bin_counts = combined[f'{dep_var_name}_diff'].groupby_bins(combined['cc_loss'], bins).count()
# bin_stds = combined[f'{dep_var_name}_diff'].groupby_bins(combined['cc_loss'], bins).std()
# bin_sems = bin_stds / np.sqrt(bin_counts)
# bin_means = binned.values
# ci_lower = bin_means - 1.96 * bin_sems
# ci_upper = bin_means + 1.96 * bin_sems

# # Save the bin means, counts and CIs to a CSV for later use
# bin_dim = next(dim for dim in binned.dims if dim.endswith('_bins'))
# bin_stats = pd.DataFrame({
#     'bin': [f'{int(interval.left)}-{int(interval.right)}' for interval in binned[bin_dim].values],
#     'mean_diff': bin_means,
#     'ci_lower': ci_lower,
#     'ci_upper': ci_upper,
#     'count': bin_counts.values
# })