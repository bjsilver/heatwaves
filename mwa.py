#%%
from pdb import run

import xarray as xr
import numpy as np
import os
import dask.array as da
from functools import partial
from constants import HANSEN_DIR, MWA_DAILY_DIR, STORAGE_DIR, LAT_CHUNK_SIZE, LON_CHUNK_SIZE
import matplotlib.pyplot as plt
import gc
from ufuncs import local_cluster
from calc_era5_percentiles import load_era5_t2m


COARSEN_FACTORS = [None, 2, 5, 10, 25]
SPATH_SUFFIX = None
USE_DASK_DASHBOARD = True
SEASON_MAP = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11]}

def _load_single_ind_var(ind_var_name):

    def __calc_tc_loss():
            tc = ds['tree_cover']
            ind_var = -(tc - tc[0])
            ind_var = ind_var.loc[{'time': slice('2003', '2020')}]
            ind_var = ind_var.rename({'time': 'year'})
            return ind_var
    if ind_var_name == 'cc_loss':
        ds = xr.open_zarr(os.path.join(HANSEN_DIR, 'hansen_tc_2000_2024_rechunked.zarr'))
        ind_var = __calc_tc_loss()
        ind_var.name = ind_var_name
        return ind_var
    
    if ind_var_name == 'cc_loss_era5grid':
        ds = xr.open_zarr(os.path.join(HANSEN_DIR, 'hansen_regridded.zarr'))
        ind_var = __calc_tc_loss()
        ind_var.name = ind_var_name
        return ind_var

    if ind_var_name == 'forest_loss_age':
        ds = xr.open_zarr(os.path.join(HANSEN_DIR, 'hansen_tc_2000_2024_loss_age.zarr'))
        var_name = 'forest_loss_age' if 'forest_loss_age' in ds.data_vars else list(ds.data_vars)[0]
        ind_var = ds[var_name]
        ind_var = ind_var.loc[{'time': slice('2003', '2020')}]
        ind_var = ind_var.rename({'time': 'year'})
        ind_var.name = ind_var_name
        return ind_var

    raise ValueError(f"Unsupported independent variable name: {ind_var_name}")


def load_ind_dep_data(ind_var_name, dep_var_name, driver_ind_var_name=None):

    if dep_var_name == 'tmax':
        dep_var = xr.open_zarr(os.path.join(STORAGE_DIR, 'rechunked', 'zhang_zhou_regridded_rechunked.zarr'))['Ta_TMAX']
        dep_var = dep_var.astype(np.float32).where(dep_var != -9999.0) / 10.0  # convert to °C
        dep_var.name = dep_var_name
    elif dep_var_name == 'heatwaves':
        dep_var = xr.open_zarr(os.path.join(STORAGE_DIR, 'heatdays', 'heatwaves_annual_sum_zz_2003-2020.zarr'))['heatwave_days']
        dep_var.name = dep_var_name
    elif dep_var_name == 'heatwaves_smoothed':
        dep_var = xr.open_zarr(os.path.join(STORAGE_DIR, 'heatdays', 'heatwaves_annual_sum_zz_2003-2020_smooth.zarr'))['heatwave_days']
        dep_var.name = dep_var_name
    elif dep_var_name == 'heatwaves_gapfilled':
        dep_var = xr.open_zarr(os.path.join(STORAGE_DIR, 'heatdays', 'heatwaves_annual_sum_zz_2003-2020_gapfilled.zarr'))['heatwave_days']
        dep_var.name = dep_var_name
    elif dep_var_name == 'heatwaves_era5-p95':
        dep_var = xr.open_zarr(os.path.join(STORAGE_DIR, 'heatdays', 'heatwaves_annual_sum_era5_2003-2020.zarr'))['heatwave_days']
        dep_var.name = dep_var_name

    elif dep_var_name == 'tmax_era5':
        dep_var = load_era5_t2m()
        dep_var.name = dep_var_name
    else:
        raise ValueError(f"Unsupported dependent variable name: {dep_var_name}")

    ind_var = _load_single_ind_var(ind_var_name).assign_coords(lat=dep_var.lat, lon=dep_var.lon)

    if driver_ind_var_name is None:
        driver_ind_var_name = ind_var_name
        driver_ind_var = ind_var
    elif driver_ind_var_name == ind_var_name:
        driver_ind_var = ind_var
    else:
        driver_ind_var = _load_single_ind_var(driver_ind_var_name).assign_coords(
            lat=dep_var.lat,
            lon=dep_var.lon,
        )

    return ind_var, driver_ind_var, dep_var



def _mwa_kernel(block, window_radius):
    """
    block: (3, T, H+2wr, W+2wr)
      channel 0 = analysis independent variable
      channel 1 = driver independent variable used for neighbor selection
      channel 2 = dependent variable
    Returns (4, T, H, W): ind_center, ind_neighbor, dep_center, dep_neighbor
    """

    ind = block[0]
    driver_ind = block[1]
    dep = block[2]
    wr = window_radius

    t_size = driver_ind.shape[0]
    h_size = driver_ind.shape[1] - 2 * wr
    w_size = driver_ind.shape[2] - 2 * wr

    out = np.full((4, t_size, h_size, w_size), np.nan, dtype=np.float32)

    center_ind = ind[:, wr:wr+h_size, wr:wr+w_size]
    center_driver = driver_ind[:, wr:wr+h_size, wr:wr+w_size]
    center_dep = dep[:, wr:wr+h_size, wr:wr+w_size]

    max_driver_diff = np.full((t_size, h_size, w_size), -np.inf, dtype=np.float32)
    neighbor_ind_selected = np.full((t_size, h_size, w_size), np.nan, dtype=np.float32)
    neighbor_dep_selected = np.full((t_size, h_size, w_size), np.nan, dtype=np.float32)
    has_lower = np.zeros((t_size, h_size, w_size), dtype=np.bool_)

    for di in range(-wr, wr + 1):
        for dj in range(-wr, wr + 1):
            if di == 0 and dj == 0:
                continue
            ri, rj = wr + di, wr + dj

            n_ind = ind[:, ri:ri+h_size, rj:rj+w_size]
            n_driver = driver_ind[:, ri:ri+h_size, rj:rj+w_size]
            n_dep = dep[:, ri:ri+h_size, rj:rj+w_size]

            driver_diff = center_driver - n_driver

            better = driver_diff > max_driver_diff
            max_driver_diff = np.where(better, driver_diff, max_driver_diff)
            neighbor_ind_selected = np.where(better, n_ind, neighbor_ind_selected)
            neighbor_dep_selected = np.where(better, n_dep, neighbor_dep_selected)

            # Keep only locations where at least one neighbor has lower driver value
            # so center_driver - neighbor_driver can be positive.
            has_lower = has_lower | (n_driver < center_driver)

    valid = (center_driver >= 0) & ~np.isnan(center_dep) & has_lower

    out[0] = np.where(valid, center_ind, np.nan)
    out[1] = np.where(valid, neighbor_ind_selected, np.nan)
    out[2] = np.where(valid, center_dep, np.nan)
    out[3] = np.where(valid, neighbor_dep_selected, np.nan)

    return out

def moving_window_analysis(ds, window_radius, ind_var_name, dep_var_name,
                           driver_ind_var_name=None, dep_var_agg=None):
    """Moving-window analysis on (time, lat, lon) inputs."""
    if driver_ind_var_name is None:
        driver_ind_var_name = ind_var_name

    if ind_var_name not in ds:
        raise ValueError(f"Independent variable '{ind_var_name}' is missing from dataset")
    if driver_ind_var_name not in ds:
        raise ValueError(f"Driver variable '{driver_ind_var_name}' is missing from dataset")

    ind_arr = ds[ind_var_name].data
    driver_arr = ds[driver_ind_var_name].data
    dep_arr = ds[dep_var_name].data

    # Stack with a dummy 4th channel so input and output both have 4 channels,
    # which allows map_overlap with trim=True to work correctly.
    dummy = da.zeros_like(ind_arr)
    stacked = da.stack([ind_arr, driver_arr, dep_arr, dummy]).rechunk({0: -1})
    stacked = stacked.persist()

    def _kernel_wrapper(block):
        # block: (4, T, H+2wr, W+2wr) — channel 3 is dummy, ignored
        wr = window_radius
        result = _mwa_kernel(block[:3], wr)  # (4, T, H, W)
        # Re-pad to match input block size so trim=True can strip the borders
        return np.pad(
            result,
            [(0, 0), (0, 0), (wr, wr), (wr, wr)],
            mode='constant',
            constant_values=np.nan,
        )  # (4, T, H+2wr, W+2wr)

    result = da.map_overlap(
        _kernel_wrapper,
        stacked,
        depth={0: 0, 1: 0, 2: window_radius, 3: window_radius},
        boundary=np.nan,
        trim=True,
        dtype=np.float32,
    )

    data_vars = {
        f'{ind_var_name}_center':   (("year", "lat", "lon"), result[0]),
        f'{ind_var_name}_neighbor': (("year", "lat", "lon"), result[1]),
        f'{dep_var_name}_center':   (("year", "lat", "lon"), result[2]),
        f'{dep_var_name}_neighbor': (("year", "lat", "lon"), result[3]),
    }

    result_ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "year": ds.coords["year"],
            "lat": ds.coords["lat"],
            "lon": ds.coords["lon"],
        },
        attrs={
            "driver_ind_var_name": driver_ind_var_name,
            "ind_var_name": ind_var_name,
            "window_radius": window_radius,
            "dep_var_agg": dep_var_agg if dep_var_agg is not None else "",
        },
    )

    return result_ds


def _generate_coarsened_versions(ind_var, driver_ind_var, dep_var, coarsen_factor):
    """Generate coarsened variable triplets, with offset versions if coarsen_factor >= 10.

    Returns a list of (ind_var_coarse, driver_ind_var_coarse, dep_var_coarse) tuples.
    For coarsen_factor >= 10, creates offset versions with 5-pixel intervals.
    """
    dss = []

    # First version: standard coarsening
    ind_coarse = ind_var.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary='trim').mean()
    driver_ind_coarse = driver_ind_var.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary='trim').mean()
    dep_coarse = dep_var.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary='trim').mean()
    # rechunk for the coarsened versions
    ind_coarse = ind_coarse.chunk({'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})
    driver_ind_coarse = driver_ind_coarse.chunk({'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})
    dep_coarse = dep_coarse.chunk({'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})
    dss.append((ind_coarse, driver_ind_coarse, dep_coarse))

    # Add offset versions if coarsen_factor >= 10
    if coarsen_factor >= 10:
        num_offsets = (coarsen_factor - 10) // 5 + 1
        for offset_idx in range(num_offsets):
            offset = (offset_idx + 1) * 5

            # Create shifted versions with offset slicing
            ind_shifted = ind_var.loc[
                {
                    'lat': slice(ind_var.lat.values[offset], ind_var.lat.values[-offset]),
                    'lon': slice(ind_var.lon.values[offset], ind_var.lon.values[-offset]),
                }
            ]
            driver_ind_shifted = driver_ind_var.loc[
                {
                    'lat': slice(driver_ind_var.lat.values[offset], driver_ind_var.lat.values[-offset]),
                    'lon': slice(driver_ind_var.lon.values[offset], driver_ind_var.lon.values[-offset]),
                }
            ]
            dep_shifted = dep_var.loc[{'lat': slice(dep_var.lat.values[offset], dep_var.lat.values[-offset]),
                                       'lon': slice(dep_var.lon.values[offset], dep_var.lon.values[-offset])}]

            # Coarsen the shifted versions
            ind_shifted_coarse = ind_shifted.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary='trim').mean()
            driver_ind_shifted_coarse = driver_ind_shifted.coarsen(
                lat=coarsen_factor,
                lon=coarsen_factor,
                boundary='trim',
            ).mean()
            dep_shifted_coarse = dep_shifted.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary='trim').mean()

            # rechunk for the coarsened versions
            ind_shifted_coarse = ind_shifted_coarse.chunk({'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})
            driver_ind_shifted_coarse = driver_ind_shifted_coarse.chunk({'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})
            dep_shifted_coarse = dep_shifted_coarse.chunk({'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})

            dss.append((ind_shifted_coarse, driver_ind_shifted_coarse, dep_shifted_coarse))

    return dss


def calc_moving_window(ind_var, driver_ind_var, dep_var, client=None, dep_var_agg='annual_mean',
                       coarsen_factor=None, window_radius=1, sname=None,
                       driver_ind_var_name=None):

    ind_var_name = ind_var.name
    dep_var_name = dep_var.name

    if driver_ind_var_name is None:
        driver_ind_var_name = ind_var_name

    if len(dep_var.lat) == 5000 and len(dep_var.lon) == 4750:    
        elev = xr.open_dataset(os.path.join(STORAGE_DIR, '..', 'elevation', 'elevation_regridded.nc'))['elevation']
        ref_ind_var = ind_var
        elev = elev.assign_coords(lat=ref_ind_var.lat, lon=ref_ind_var.lon)
        ind_var = ind_var.where(elev < 1000)
        driver_ind_var = driver_ind_var.where(elev < 1000)
        dep_var = dep_var.where(elev < 1000)

    # Generate list of variable triplets (with offsets if applicable)
    if coarsen_factor:
        dss = _generate_coarsened_versions(ind_var, driver_ind_var, dep_var, coarsen_factor)
    else:
        dss = [(ind_var, driver_ind_var, dep_var)]

    os.makedirs(MWA_DAILY_DIR, exist_ok=True)

    # Process each version
    for version_idx, (ind_var_version, driver_ind_var_version, dep_var_version) in enumerate(dss):
        # Add version suffix to output paths if there are multiple versions
        if len(dss) > 1:
            out_path = os.path.join(MWA_DAILY_DIR, f'{sname}_mwa_results_{version_idx}.zarr')
        else:
            out_path = os.path.join(MWA_DAILY_DIR, f'{sname}_mwa_results.zarr')

        dep_var_version = aggregate_dependent_variable(dep_var_version, dep_var_agg)

        ds_vars = {
            ind_var_name: ind_var_version,
            dep_var_name: dep_var_version,
        }
        if driver_ind_var_name != ind_var_name:
            ds_vars[driver_ind_var_name] = driver_ind_var_version
        ds_all = xr.Dataset(ds_vars)
        result_ds = moving_window_analysis(
            ds_all,
            window_radius=window_radius,
            ind_var_name=ind_var_name,
            dep_var_name=dep_var_name,
            driver_ind_var_name=driver_ind_var_name,
            dep_var_agg=dep_var_agg,
        )

        result_ds.to_zarr(out_path, mode='w', consolidated=False)

        if client is not None:
            client.cancel(result_ds)
        gc.collect()
        print(f"All-year results written to {out_path}")




def aggregate_dependent_variable(dep_var, aggregation):
    if 'annual' in aggregation and 'year' in dep_var.dims:
        return dep_var
    match aggregation:
        case "annual_mean":
            annual = dep_var.groupby('time.year').mean()
        case "annual_sum":
            annual = dep_var.groupby('time.year').sum()
        case "annual_95pct":
            annual = dep_var.groupby('time.year').quantile(0.95)
        case s if s in ["DJF_mean", "MAM_mean", "JJA_mean", "SON_mean"]:
            season = s.split("_")[0]
            months = SEASON_MAP[season]
            annual = dep_var.where(dep_var['time.month'].isin(months)).groupby('time.year').mean()
        case _:
            raise ValueError(f"Unsupported aggregation: {aggregation}")

    return annual



def _check_zarrs_exist(sname):
    """Check if output zarrs exist (either indexed or non-indexed versions)."""
    # Check for indexed versions (from offset resampling)
    indexed_out = os.path.join(MWA_DAILY_DIR, f'{sname}_mwa_results_0.zarr')
    if os.path.exists(indexed_out):
        return True

    # Check for non-indexed versions
    out_path = os.path.join(MWA_DAILY_DIR, f'{sname}_mwa_results.zarr')
    return os.path.exists(out_path)


def run_mwa(coarsen_factor=None, window_radius=1, dep_var_agg='annual_mean', client=None,
            ind_var_name='cc_loss', driver_ind_var_name=None, dep_var_name='heatwaves'):

    if driver_ind_var_name is None:
        driver_ind_var_name = ind_var_name

    ind_var, driver_ind_var, dep_var = load_ind_dep_data(
        ind_var_name=ind_var_name,
        dep_var_name=dep_var_name,
        driver_ind_var_name=driver_ind_var_name,
    )

    # check if output zarr already exists
    sname = f'{dep_var.name}_vs_{ind_var_name}_driver_{driver_ind_var_name}_{dep_var_agg}'
    if coarsen_factor:
        sname += f'_coarsened{coarsen_factor}'
    if SPATH_SUFFIX:
        sname += f'_{SPATH_SUFFIX}'

    if not _check_zarrs_exist(sname):
        print(f"Output zarrs do not exist for {sname}. Proceeding with moving window analysis.")
        calc_moving_window(ind_var=ind_var, driver_ind_var=driver_ind_var, dep_var=dep_var, client=client, window_radius=window_radius,
                           coarsen_factor=coarsen_factor, sname=sname, dep_var_agg=dep_var_agg,
                           driver_ind_var_name=driver_ind_var_name)
        print(
            f"Moving window analysis completed for {dep_var.name} vs {ind_var_name} "
            f"(driver={driver_ind_var_name}) with coarsen factor {coarsen_factor}."
        )
    else:
        print(f"Output zarrs already exist for {sname}. Skipping moving window analysis.")


def run_mwa_coarsen_loop(ind_var_name, driver_ind_var_name, dep_var_agg,
                         dep_var_name,
                         client=None):
    for coarsen_factor in COARSEN_FACTORS:
        run_mwa(coarsen_factor=coarsen_factor, window_radius=1, dep_var_agg=dep_var_agg, client=client,
                ind_var_name=ind_var_name, driver_ind_var_name=driver_ind_var_name,
                dep_var_name=dep_var_name)
            
#%%
if __name__ == "__main__":
    
    with local_cluster(n_workers=40, threads_per_worker=1, memory_limit='12GB',
                       suppress_dask_warnings=True, use_dask_dashboard=USE_DASK_DASHBOARD) as client:
        run_mwa_coarsen_loop(ind_var_name='cc_loss', driver_ind_var_name='cc_loss',
                             dep_var_name='tmax', dep_var_agg='annual_mean', client=client)
        run_mwa_coarsen_loop(ind_var_name='forest_loss_age', driver_ind_var_name='forest_loss_age',
                            dep_var_name='tmax', dep_var_agg='annual_mean', client=client)
        run_mwa_coarsen_loop(ind_var_name='cc_loss', driver_ind_var_name='cc_loss',
                             dep_var_name='heatwaves_gapfilled', dep_var_agg='annual_sum', client=client)