#%%
import xarray as xr
import numpy as np
import os
import dask.array as da
from functools import partial
from constants import HANSEN_DIR, MWA_DAILY_DIR, STORAGE_DIR, LAT_CHUNK_SIZE, LON_CHUNK_SIZE
import matplotlib.pyplot as plt
import gc
from ufuncs import local_cluster

COARSEN_FACTORS = [None, 2, 5, 10, 25]
SPATH_SUFFIX = None
USE_DASK_DASHBOARD = True

def load_ind_dep_data(ind_var_name, dep_var_name):

    if ind_var_name == 'cc_loss':
        ds = xr.open_zarr(os.path.join(HANSEN_DIR, 'hansen_tc_2000_2024_rechunked.zarr'))
        tc = ds['tree_cover']
        ind_var = -(tc - tc[0])
        ind_var = ind_var.loc[{'time': slice('2003', '2020')}]
        ind_var = ind_var.rename({'time': 'year'})
        ind_var.name = ind_var_name
        
    else:
        raise ValueError(f"Unsupported independent variable name: {ind_var_name}")

    if dep_var_name == 'tmax':
        dep_var = xr.open_zarr(os.path.join(STORAGE_DIR, 'rechunked', 'zhang_zhou_regridded_rechunked.zarr'))['Ta_TMAX']
        dep_var = dep_var.astype(np.float32).where(dep_var != -9999.0) / 10.0  # convert to °C
        dep_var.name = dep_var_name
    elif dep_var_name == 'heatwaves':
        dep_var = xr.open_zarr(os.path.join(STORAGE_DIR, 'heatdays', 'heatwaves_zz_2003-2020.zarr'))['heatwaves']
        dep_var.name = dep_var_name
    elif dep_var_name == 'heatwaves_smoothed':
        dep_var = xr.open_zarr(os.path.join(STORAGE_DIR, 'heatdays', 'heatwaves_zz_2003-2020_smooth.zarr'))['heatwaves']
        dep_var.name = dep_var_name
    else:
        raise ValueError(f"Unsupported dependent variable name: {dep_var_name}")

    ind_var = ind_var.assign_coords(lat=dep_var.lat, lon=dep_var.lon)
    return ind_var, dep_var



def _mwa_kernel(block, window_radius):
    """
    block: (2, T, H+2wr, W+2wr) — channel 0 = independent var, channel 1 = dependent var.
    Returns same shape, with results embedded in centre for each timestep.
    """

    ind = block[0]
    dep = block[1]
    wr = window_radius

    t_size = ind.shape[0]
    h_size = ind.shape[1] - 2 * wr
    w_size = ind.shape[2] - 2 * wr

    out = np.full((2, t_size, h_size, w_size), np.nan, dtype=np.float32)

    center_ind = ind[:, wr:wr+h_size, wr:wr+w_size]
    center_dep = dep[:, wr:wr+h_size, wr:wr+w_size]

    max_ind_diff = np.full((t_size, h_size, w_size), -np.inf, dtype=np.float32)
    max_dep_diff = np.zeros((t_size, h_size, w_size), dtype=np.float32)
    has_higher = np.zeros((t_size, h_size, w_size), dtype=np.bool_)

    for di in range(-wr, wr + 1):
        for dj in range(-wr, wr + 1):
            if di == 0 and dj == 0:
                continue
            ri, rj = wr + di, wr + dj
            
            n_ind = ind[:, ri:ri+h_size, rj:rj+w_size]
            n_dep = dep[:, ri:ri+h_size, rj:rj+w_size]
            # dep_diff = n_dep - center_dep
            # ind_diff = n_ind - center_ind
            dep_diff = center_dep - n_dep
            ind_diff = center_ind - n_ind
            
            better = ind_diff > max_ind_diff
            max_ind_diff = np.where(better, ind_diff, max_ind_diff)
            max_dep_diff = np.where(better, dep_diff, max_dep_diff)

            has_higher = has_higher | (n_ind > center_ind)

    valid = (center_ind >= 0)  & ~np.isnan(center_dep) & has_higher


    out[0] = np.where(valid, max_ind_diff, np.nan)
    out[1] = np.where(valid, max_dep_diff, np.nan)

    return out

def moving_window_analysis(ds, window_radius, ind_var_name, dep_var_name):
    """Moving-window analysis on (time, lat, lon) inputs."""
    ind_arr = ds[ind_var_name].data
    dep_arr = ds[dep_var_name].data


    stacked = da.stack([ind_arr, dep_arr]).rechunk({0: -1})
    stacked = stacked.persist()
    
    result = da.map_overlap(
        partial(_mwa_kernel, window_radius=window_radius),
        stacked,
        depth={0: 0, 1: 0, 2: window_radius, 3: window_radius},
        boundary=np.nan,
        trim=False,
        dtype=np.float32,
    )
    
    ind_ds = xr.Dataset(
    data_vars={
        f'{ind_var_name}_cdiff': (("year", "lat", "lon"), result[0]),
    },
    coords={
        "year": ds.coords["year"],
        "lat": ds.coords["lat"],
        "lon": ds.coords["lon"]
    }
    )

    dep_ds = xr.Dataset(
    data_vars={
        f'{dep_var_name}_cdiff': (("year", "lat", "lon"), result[1]),
    },
    coords={
        "year": ds.coords["year"],
        "lat": ds.coords["lat"],
        "lon": ds.coords["lon"]
    }
    )

    return ind_ds, dep_ds


def _generate_coarsened_versions(ind_var, dep_var, coarsen_factor):
    """Generate coarsened variable pairs, with offset versions if coarsen_factor >= 10.
    
    Returns a list of (ind_var_coarse, dep_var_coarse) tuples.
    For coarsen_factor >= 10, creates offset versions with 5-pixel intervals.
    """
    dss = []
    
    # First version: standard coarsening
    ind_coarse = ind_var.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary='trim').mean()
    dep_coarse = dep_var.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary='trim').mean()
    # rechunk for the coarsened versions
    ind_coarse = ind_coarse.chunk({'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})
    dep_coarse = dep_coarse.chunk({'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})
    dss.append((ind_coarse, dep_coarse))
    
    # Add offset versions if coarsen_factor >= 10
    if coarsen_factor >= 10:
        num_offsets = (coarsen_factor - 10) // 5 + 1
        for offset_idx in range(num_offsets):
            offset = (offset_idx + 1) * 5
            
            # Create shifted versions with offset slicing
            ind_shifted = ind_var.loc[{'lat': slice(ind_var.lat.values[offset], ind_var.lat.values[-offset]),
                                       'lon': slice(ind_var.lon.values[offset], ind_var.lon.values[-offset])}]
            dep_shifted = dep_var.loc[{'lat': slice(dep_var.lat.values[offset], dep_var.lat.values[-offset]),
                                       'lon': slice(dep_var.lon.values[offset], dep_var.lon.values[-offset])}]
            
            # Coarsen the shifted versions
            ind_shifted_coarse = ind_shifted.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary='trim').mean()
            dep_shifted_coarse = dep_shifted.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary='trim').mean()

            # rechunk for the coarsened versions
            ind_shifted_coarse = ind_shifted_coarse.chunk({'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})
            dep_shifted_coarse = dep_shifted_coarse.chunk({'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})

            dss.append((ind_shifted_coarse, dep_shifted_coarse))
    
    return dss


def calc_moving_window(ind_var, dep_var, client=None, time_res='annual_mean',
                             coarsen_factor=None, window_radius=1, sname=None):
    
    ind_var_name = ind_var.name
    dep_var_name = dep_var.name

    elev = xr.open_dataset(os.path.join(STORAGE_DIR, '..', 'elevation', 'elevation_regridded.nc'))['elevation']
    elev = elev.assign_coords(lat=ind_var.lat, lon=ind_var.lon)
    ind_var = ind_var.where(elev < 1000)
    dep_var = dep_var.where(elev < 1000)

    # Generate list of variable pairs (with offsets if applicable)
    if coarsen_factor:
        dss = _generate_coarsened_versions(ind_var, dep_var, coarsen_factor)
    else:
        dss = [(ind_var, dep_var)]
    
    os.makedirs(MWA_DAILY_DIR, exist_ok=True)
    
    # Process each version
    for version_idx, (ind_var_version, dep_var_version) in enumerate(dss):
        # Add version suffix to output paths if there are multiple versions
        if len(dss) > 1:
            out_path_ind = os.path.join(MWA_DAILY_DIR, f'{sname}_mwa_results_ind_{version_idx}.zarr')
            out_path_dep = os.path.join(MWA_DAILY_DIR, f'{sname}_mwa_results_dep_{version_idx}.zarr')
        else:
            out_path_ind = os.path.join(MWA_DAILY_DIR, f'{sname}_mwa_results_ind.zarr')
            out_path_dep = os.path.join(MWA_DAILY_DIR, f'{sname}_mwa_results_dep.zarr')

        # Aggregate if needed
        if time_res in ('annual_mean', 'annual_sum'):
            dep_var_version = aggregate_independent_variable(dep_var_version, time_res)

        if time_res in ('annual_mean', 'annual_sum'):
            years = ind_var.years
            dep_years = dep_var.years
            dep_var = dep_var.isel(time=np.isin(dep_years, years))
            dep_var = dep_var.assign_coords(time=ind_var.time)

            ds_all = xr.Dataset({ind_var_name: ind_var_version, dep_var_name: dep_var_version})
            ind_ds, dep_ds = moving_window_analysis(
                ds_all,
                window_radius=window_radius,
                ind_var_name=ind_var_name,
                dep_var_name=dep_var_name
            )

            ind_ds.to_zarr(out_path_ind, mode='w', consolidated=False)
            dep_ds.to_zarr(out_path_dep, mode='w', consolidated=False)

            if client is not None:
                client.cancel(ind_ds)
                client.cancel(dep_ds)
            gc.collect()
            print(f"All-year results written to {out_path_ind} and {out_path_dep}")
        else:
            raise ValueError(f"Unsupported time_res: {time_res}")



def aggregate_independent_variable(ind_var, time_res):


    if time_res == "annual_mean":
        annual = ind_var.groupby("time.year").mean()
        annual = annual.rename({"year": "time"})
        annual = annual.assign_coords(time=annual.time.astype(np.int32))
        return annual

    elif time_res == "annual_sum":
        annual = ind_var.groupby("time.year").sum()
        annual = annual.rename({"year": "time"})
        annual = annual.assign_coords(time=annual.time.astype(np.int32))
        return annual

    raise ValueError(f"Unsupported time_res: {time_res}")


def _check_zarrs_exist(sname):
    """Check if output zarrs exist (either indexed or non-indexed versions)."""
    # Check for indexed versions (from offset resampling)
    indexed_ind = os.path.join(MWA_DAILY_DIR, f'{sname}_mwa_results_ind_0.zarr')
    indexed_dep = os.path.join(MWA_DAILY_DIR, f'{sname}_mwa_results_dep_0.zarr')
    if os.path.exists(indexed_ind) and os.path.exists(indexed_dep):
        return True
    
    # Check for non-indexed versions
    out_path_ind = os.path.join(MWA_DAILY_DIR, f'{sname}_mwa_results_ind.zarr')
    out_path_dep = os.path.join(MWA_DAILY_DIR, f'{sname}_mwa_results_dep.zarr')
    return os.path.exists(out_path_ind) and os.path.exists(out_path_dep)


def run_mwa(coarsen_factor=None, window_radius=1, time_res='annual_mean', client=None,
            ind_var_name='cc_loss', dep_var_name='heatwaves'):

    ind_var, dep_var = load_ind_dep_data(ind_var_name=ind_var_name, dep_var_name=dep_var_name)

    # check if output zarr already exists
    sname = f'{dep_var.name}_vs_{ind_var.name}_{time_res}'
    if coarsen_factor:
        sname += f'_coarsened{coarsen_factor}'
    if SPATH_SUFFIX:
        sname += f'_{SPATH_SUFFIX}'
    
    if not _check_zarrs_exist(sname):
        print(f"Output zarrs do not exist for {sname}. Proceeding with moving window analysis.")
        calc_moving_window(ind_var=ind_var, dep_var=dep_var, client=client, window_radius=window_radius,
                           coarsen_factor=coarsen_factor, sname=sname, time_res=time_res)
        print(f"Moving window analysis completed for {dep_var.name} vs {ind_var.name} with coarsen factor {coarsen_factor}.")
    else:
        print(f"Output zarrs already exist for {sname}. Skipping moving window analysis.")



def run_mwa_coarsen_loop(client=None, time_res='annual_mean', ind_var_name='cc_loss', dep_var_name='heatwaves'):
    for coarsen_factor in COARSEN_FACTORS:
        run_mwa(coarsen_factor=coarsen_factor, window_radius=1, time_res=time_res, client=client,
                ind_var_name=ind_var_name, dep_var_name=dep_var_name)
            
#%%
if __name__ == "__main__":
    
    with local_cluster(n_workers=24, threads_per_worker=2, memory_limit='20GB',
                       suppress_dask_warnings=True, use_dask_dashboard=USE_DASK_DASHBOARD) as client:
        run_mwa_coarsen_loop(client=client, time_res='annual_mean', ind_var_name='cc_loss', dep_var_name='tmax')
        run_mwa_coarsen_loop(client=client, time_res='annual_sum', ind_var_name='cc_loss', dep_var_name='heatwaves')
        run_mwa_coarsen_loop(client=client, time_res='annual_sum', ind_var_name='cc_loss', dep_var_name='heatwaves_smoothed')
        # run_mwa_coarsen_loop(client=client, time_res='daily', ind_var_name='cc_loss', dep_var_name='tmax')

