#%%

import xarray as xr
import numpy as np
import os
import dask.array as da
from constants import HANSEN_DIR, MWA_DAILY_DIR, STORAGE_DIR
import gc
import json
import shutil
from glob import glob
from ufuncs import local_cluster
from calc_era5_percentiles import load_era5_t2m
from numba import njit
import pandas as pd
from dask import delayed

COARSEN_FACTORS = [None, 2, 5, 10, 25]

SPATH_SUFFIX = None
USE_DASK_DASHBOARD = True
SEASON_MAP = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11]}
NEIGHBOUR_MAP_FILL_VALUE = np.int8(-128)

def _load_single_ind_var(ind_var_name):

    def __calc_tc_loss():
            tc = ds['tree_cover']
            ind_var = -(tc - tc[0])
            ind_var = ind_var.loc[{'time': slice('2003', '2020')}]
            ind_var = ind_var.rename({'time': 'year'})
            return ind_var
    if ind_var_name == 'cc_loss':
        ds = xr.open_zarr(os.path.join(HANSEN_DIR, 'hansen_tc_2000_2024_rechunked.zarr'))
        ds = ds.astype(np.float32)
        ind_var = __calc_tc_loss()
        ind_var.name = ind_var_name
        return ind_var

    if ind_var_name == 'cc':
        ds = xr.open_zarr(os.path.join(HANSEN_DIR, 'hansen_tc_2000_2024_rechunked.zarr'))
        ds = ds.astype(np.float32)
        tc = ds['tree_cover']
        tc_loss = -(tc - tc[0])
        ind_var = tc.where(tc_loss > 0)
        return ind_var
    
    if ind_var_name == 'cc_loss_era5grid':
        ds = xr.open_zarr(os.path.join(HANSEN_DIR, 'hansen_regridded.zarr'))
        ds = ds.astype(np.float32)
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


def load_ind_dep_data(ind_var_name, dep_var_name):

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

    return ind_var, dep_var

@njit
def _mwa_kernel(block, window_radius):
    ind = block
    wr = window_radius

    t_size = ind.shape[0]
    h_size = ind.shape[1] - 2 * wr
    w_size = ind.shape[2] - 2 * wr

    out = np.full((2, t_size, h_size, w_size), NEIGHBOUR_MAP_FILL_VALUE, dtype=np.int8)

    center_ind = ind[:, wr:wr+h_size, wr:wr+w_size]

    max_ind_diff = np.full((t_size, h_size, w_size), np.float32(-np.inf), dtype=np.float32)
    max_ind_diff_ioffset = np.full((t_size, h_size, w_size), NEIGHBOUR_MAP_FILL_VALUE, dtype=np.int8)
    max_ind_diff_joffset = np.full((t_size, h_size, w_size), NEIGHBOUR_MAP_FILL_VALUE, dtype=np.int8)

    for di in range(-wr, wr + 1):
        for dj in range(-wr, wr + 1):
            if di == 0 and dj == 0:
                continue
            ri, rj = wr + di, wr + dj

            n_ind = ind[:, ri:ri+h_size, rj:rj+w_size]
            ind_diff = center_ind - n_ind

            better = ind_diff > max_ind_diff
            max_ind_diff = np.where(better, ind_diff, max_ind_diff)
            max_ind_diff_ioffset = np.where(better, np.int8(di), max_ind_diff_ioffset)
            max_ind_diff_joffset = np.where(better, np.int8(dj), max_ind_diff_joffset)

    valid = max_ind_diff > 0

    out[0] = np.where(valid, max_ind_diff_ioffset, NEIGHBOUR_MAP_FILL_VALUE)
    out[1] = np.where(valid, max_ind_diff_joffset, NEIGHBOUR_MAP_FILL_VALUE)

    return out

def _get_mwa_neighbour_map(ind_var, ind_var_name, window_radius, sname):
    spath = os.path.join(MWA_DAILY_DIR, f'neighbour_maps/{sname}.zarr')
    if os.path.exists(spath):
        print(f"Neighbour map zarr already exists for {sname}.\nLoading from {spath}.")
        neighbour_maps_ds = xr.open_zarr(spath, consolidated=False)

        years_match = np.array_equal(neighbour_maps_ds['year'].values, ind_var['year'].values)
        lats_match = np.array_equal(neighbour_maps_ds['lat'].values, ind_var['lat'].values)
        lons_match = np.array_equal(neighbour_maps_ds['lon'].values, ind_var['lon'].values)

        if years_match and lats_match and lons_match:
            return neighbour_maps_ds

        print(
            "Existing neighbour map grid does not match current input grid. "
            "Recomputing and overwriting existing neighbour map store."
        )

    ind_arr = ind_var.data

    neighbour_maps = da.map_overlap(
        _mwa_kernel,
        ind_arr,
        window_radius=window_radius,
        depth={0: 0, 1: window_radius, 2: window_radius},
        boundary=np.nan,
        trim=False,
        dtype=np.int8,
        new_axis=0,
        chunks=((2,),) + ind_arr.chunks
    )
    neighbour_maps_ds = xr.Dataset(
        {
            'neighbour_maps': (('offset', 'year', 'lat', 'lon'), neighbour_maps),
        },
        coords={
            'offset': ['i', 'j'],
            'year': ind_var['year'],
            'lat': ind_var['lat'],
            'lon': ind_var['lon'],
        },
    )
    neighbour_maps_ds['neighbour_maps'].attrs['missing_offset_value'] = int(NEIGHBOUR_MAP_FILL_VALUE)

    encoding = {
        'neighbour_maps': {
            'dtype': 'int8',
            '_FillValue': int(NEIGHBOUR_MAP_FILL_VALUE),
        }
    }

    neighbour_maps_ds.to_zarr(spath, mode='w', consolidated=False, encoding=encoding)
    neighbour_maps_ds = xr.open_zarr(spath, consolidated=False)
    return neighbour_maps_ds


def _gather_mwa_results_kernel(arr, neighbour_maps, window_radius):
    t_size = arr.shape[0]
    h_size = arr.shape[1] - 2 * window_radius
    w_size = arr.shape[2] - 2 * window_radius

    

    out = np.full((2, t_size, h_size, w_size), np.nan, dtype=np.float32)
    center_arr = np.full((t_size, h_size, w_size), np.nan, dtype=np.float32)
    neighbour_arr = np.full((t_size, h_size, w_size), np.nan, dtype=np.float32)


    center_vals = arr[:, window_radius:window_radius + h_size, window_radius:window_radius + w_size]
    for di in range(-window_radius, window_radius + 1):
        for dj in range(-window_radius, window_radius + 1):
            if di == 0 and dj == 0:
                continue
            ri, rj = window_radius + di, window_radius + dj
            n_arr = arr[:, ri:ri + h_size, rj:rj + w_size]
            neighbour_in = (neighbour_maps[0] == di) & (neighbour_maps[1] == dj)

            neighbour_arr = np.where(neighbour_in, n_arr, neighbour_arr)
            center_arr = np.where(neighbour_in, center_vals, center_arr)

    out[0] = center_arr
    out[1] = neighbour_arr
    return out


def _gather_mwa_results_kernel_daily(arr, neighbour_maps, window_radius):
    t_size = arr.shape[0]
    h_size = arr.shape[1] - 2 * window_radius
    w_size = arr.shape[2] - 2 * window_radius

    

    out = np.full((2, t_size, h_size, w_size), np.nan, dtype=np.float32)
    center_arr = np.full((t_size, h_size, w_size), np.nan, dtype=np.float32)
    neighbour_arr = np.full((t_size, h_size, w_size), np.nan, dtype=np.float32)


    center_vals = arr[:, window_radius:window_radius + h_size, window_radius:window_radius + w_size]
    for di in range(-window_radius, window_radius + 1):
        for dj in range(-window_radius, window_radius + 1):
            if di == 0 and dj == 0:
                continue
            ri, rj = window_radius + di, window_radius + dj
            n_arr = arr[:, ri:ri + h_size, rj:rj + w_size]
            neighbour_in = (neighbour_maps[0] == di) & (neighbour_maps[1] == dj)

            neighbour_arr = np.where(neighbour_in[ri:ri + h_size, rj:rj + w_size], n_arr, neighbour_arr)
            center_arr = np.where(neighbour_in[ri:ri + h_size, rj:rj + w_size], center_vals, center_arr)

    out[0] = center_arr
    out[1] = neighbour_arr
    return out

def gather_mwa_results(ind_var, dep_var, neighbour_maps, ind_var_name, dep_var_name, window_radius):


    ind_arr = ind_var.data
    
    ind_results = da.map_overlap(
        _gather_mwa_results_kernel,
        ind_arr,
        neighbour_maps.data,
        window_radius=window_radius,
        depth=[
        {0: 0, 1: window_radius, 2: window_radius},
        {0: 0, 1: 0, 2: 0, 3: 0},
        ],
        boundary=[np.nan, NEIGHBOUR_MAP_FILL_VALUE],
        trim=False,
        dtype=np.float32,
        chunks=((2,),) + ind_arr.chunks,
    )

    ind_results_ds = xr.Dataset(
        {
            'ind_results': (('position', 'year', 'lat', 'lon'), 
            ind_results),
        },
        coords={
            'position': ['center', 'neighbour'],
            'year': ind_var['year'],
            'lat': ind_var['lat'],
            'lon': ind_var['lon'],
        },
    )

    dep_arr = dep_var.data
    dep_time_dim = 'year' if 'year' in dep_var.dims else 'time'

    if dep_time_dim == 'year':
    
        dep_results = da.map_overlap(
            _gather_mwa_results_kernel,
            dep_arr,
            neighbour_maps.data,
            window_radius=window_radius,
            depth=[
                    {0: 0, 1: window_radius, 2: window_radius},
                    {0: 0, 1: 0, 2: 0, 3: 0},
                    ],
            boundary=[np.nan, NEIGHBOUR_MAP_FILL_VALUE],
            trim=False,
            dtype=np.float32,
            chunks=((2,),) + dep_arr.chunks,
        )

    elif dep_time_dim == 'time':
        dep_results_list = []
        # dep_var = dep_var.chunk({'time': 73})
        

        for year in neighbour_maps['year'].values:

            dep_var_year = dep_var.sel(time=str(year))
            dep_var_year = dep_var_year.chunk({'time': 73})
            neighbour_maps_year = neighbour_maps.sel(year=year)


            dep_results_year = da.map_overlap(
                _gather_mwa_results_kernel_daily,
                dep_var_year.data,
                neighbour_maps_year.data,
                window_radius=window_radius,
                depth=[
                    {0: 0, 1: window_radius, 2: window_radius},
                    {0: 0, 1: window_radius, 2: window_radius},
                ],
                boundary=[np.nan, NEIGHBOUR_MAP_FILL_VALUE],
                trim=False,
                align_arrays=False,
                dtype=np.float32,
                chunks=((2,),) + dep_var_year.data.chunks,
            )
            dep_results_list.append(dep_results_year)
            
            
        dep_results = da.concatenate(dep_results_list, axis=1)
    dep_results_ds = xr.Dataset(
        {
            'dep_results': (('position', dep_time_dim, 'lat', 'lon'), 
            dep_results),
        },
        coords={
            'position': ['center', 'neighbour'],
            dep_time_dim: dep_var[dep_time_dim],
            'lat': dep_var['lat'],
            'lon': dep_var['lon'],
        },
    )

    return  ind_results_ds, dep_results_ds


def moving_window_analysis(ind_var, dep_var, window_radius, ind_var_name, dep_var_name, sname):
    """Moving-window analysis on (time, lat, lon) inputs."""

    neighbour_maps = _get_mwa_neighbour_map(ind_var, ind_var_name, window_radius, sname)['neighbour_maps']
    ind_results_ds, dep_results_ds = gather_mwa_results(
        ind_var=ind_var,
        dep_var=dep_var,
        neighbour_maps=neighbour_maps,
        ind_var_name=ind_var_name,
        dep_var_name=dep_var_name,
        window_radius=window_radius
    )

    return ind_results_ds, dep_results_ds


def _version_dir(out_root, version_idx):
    return os.path.join(out_root, f'version={int(version_idx)}')


def _dataset_dir(out_root, version_idx, dataset_kind):
    return os.path.join(_version_dir(out_root, version_idx), dataset_kind)


def _chunk_offsets(chunks):
    offsets = [0]
    for c in chunks:
        offsets.append(offsets[-1] + int(c))
    return offsets


def _write_chunk_parquet(
    centre_chunk,
    neighbour_chunk,
    time_vals,
    lat_vals,
    lon_vals,
    time_col,
    centre_col,
    neighbour_col,
    out_path,
    compression,
):
    centre_chunk = np.asarray(centre_chunk, dtype=np.float32)
    neighbour_chunk = np.asarray(neighbour_chunk, dtype=np.float32)

    t_size = centre_chunk.shape[0]
    y_size = centre_chunk.shape[1]
    x_size = centre_chunk.shape[2]

    time_vals = np.asarray(time_vals).astype(str)
    time_rep = np.repeat(time_vals, y_size * x_size)
    lat_rep = np.tile(np.repeat(np.asarray(lat_vals), x_size), t_size)
    lon_rep = np.tile(np.asarray(lon_vals), t_size * y_size)

    df = pd.DataFrame(
        {
            time_col: time_rep,
            'lat': lat_rep,
            'lon': lon_rep,
            centre_col: centre_chunk.ravel(),
            neighbour_col: neighbour_chunk.ravel(),
        }
    )
    df = df.dropna(subset=[centre_col])
    if df.empty:
        return None
    df.to_parquet(out_path, engine='pyarrow', compression=compression, index=False)
    return out_path


def _build_chunk_write_tasks(
    centre_arr,
    neighbour_arr,
    time_vals,
    lat_vals,
    lon_vals,
    time_col,
    centre_col,
    neighbour_col,
    out_dir,
    compression,
):
    centre_delayed = centre_arr.to_delayed()
    neighbour_delayed = neighbour_arr.to_delayed()

    time_chunks, lat_chunks, lon_chunks = centre_arr.chunks
    time_offsets = _chunk_offsets(time_chunks)
    lat_offsets = _chunk_offsets(lat_chunks)
    lon_offsets = _chunk_offsets(lon_chunks)

    tasks = []
    for ti, yi, xi in np.ndindex(centre_delayed.shape):
        t0, t1 = time_offsets[ti], time_offsets[ti + 1]
        y0, y1 = lat_offsets[yi], lat_offsets[yi + 1]
        x0, x1 = lon_offsets[xi], lon_offsets[xi + 1]

        out_path = os.path.join(out_dir, f'part-t{ti}-y{yi}-x{xi}.parquet')
        tasks.append(
            delayed(_write_chunk_parquet)(
                centre_delayed[ti, yi, xi],
                neighbour_delayed[ti, yi, xi],
                time_vals[t0:t1],
                lat_vals[y0:y1],
                lon_vals[x0:x1],
                time_col,
                centre_col,
                neighbour_col,
                out_path,
                compression,
            )
        )
    return tasks


def _write_member_parquet(ind_results, dep_results, out_root, version_idx, client, compression='zstd'):
    os.makedirs(_version_dir(out_root, version_idx), exist_ok=True)

    ind_out = _dataset_dir(out_root, version_idx, 'ind')
    dep_out = _dataset_dir(out_root, version_idx, 'dep')

    if os.path.isdir(ind_out):
        shutil.rmtree(ind_out)
    if os.path.isdir(dep_out):
        shutil.rmtree(dep_out)
    os.makedirs(ind_out, exist_ok=True)
    os.makedirs(dep_out, exist_ok=True)

    # if client is None:
    #     raise ValueError('A Dask client must be provided to _write_member_parquet.')

    ind_centre = ind_results.sel(position='center').reset_coords('position', drop=True).astype(np.float32).data
    ind_neighbour = ind_results.sel(position='neighbour').reset_coords('position', drop=True).astype(np.float32).data
    ind_tasks = _build_chunk_write_tasks(
        centre_arr=ind_centre,
        neighbour_arr=ind_neighbour,
        time_vals=ind_results['year'].values,
        lat_vals=ind_results['lat'].values,
        lon_vals=ind_results['lon'].values,
        time_col='year',
        centre_col='ind_centre',
        neighbour_col='ind_neighbour',
        out_dir=ind_out,
        compression=compression,
    )

    dep_time_dim = 'year' if 'year' in dep_results.dims else 'time'
    dep_centre = dep_results.sel(position='center').reset_coords('position', drop=True).astype(np.float32).data
    dep_neighbour = dep_results.sel(position='neighbour').reset_coords('position', drop=True).astype(np.float32).data
    dep_tasks = _build_chunk_write_tasks(
        centre_arr=dep_centre,
        neighbour_arr=dep_neighbour,
        time_vals=dep_results[dep_time_dim].values,
        lat_vals=dep_results['lat'].values,
        lon_vals=dep_results['lon'].values,
        time_col=dep_time_dim,
        centre_col='dep_centre',
        neighbour_col='dep_neighbour',
        out_dir=dep_out,
        compression=compression,
    )

    print(
        f"Submitting parquet chunk writes: ind={len(ind_tasks)} tasks, "
        f"dep={len(dep_tasks)} tasks, total={len(ind_tasks) + len(dep_tasks)} tasks."
    )

    print("Submitting independent variable parquet write tasks...")
    futures = client.compute(ind_tasks)
    client.gather(futures)

    print("Submitting dependent variable parquet write tasks...")
    futures = client.compute(dep_tasks)
    client.gather(futures)


def _has_parquet_files(path):
    if not os.path.isdir(path):
        return False
    return bool(glob(os.path.join(path, '**', '*.parquet'), recursive=True))


def _has_complete_version_members(out_root, version_idx):
    ind_path = _dataset_dir(out_root, version_idx, 'ind')
    dep_path = _dataset_dir(out_root, version_idx, 'dep')
    return _has_parquet_files(ind_path) and _has_parquet_files(dep_path)


def _expected_version_count(coarsen_factor):
    if not coarsen_factor or coarsen_factor < 10:
        return 1
    num_offsets = (coarsen_factor - 10) // 5 + 1
    return 1 + num_offsets


def _write_run_metadata(out_root, sname, coarsen_factor, dep_var_agg, expected_versions):
    completed_versions = [
        idx for idx in range(expected_versions)
        if _has_complete_version_members(out_root, idx)
    ]
    metadata = {
        'sname': sname,
        'coarsen_factor': coarsen_factor,
        'dep_var_agg': dep_var_agg,
        'expected_versions': expected_versions,
        'completed_versions': completed_versions,
        'complete': len(completed_versions) == expected_versions,
    }
    with open(os.path.join(out_root, '_run_metadata.json'), 'w', encoding='utf-8') as fobj:
        json.dump(metadata, fobj, indent=2)
def _generate_coarsened_versions(ind_var, dep_var, coarsen_factor):
    """Generate coarsened variable triplets, with offset versions if coarsen_factor >= 10.

    Returns a list of (ind_var_coarse, dep_var_coarse) tuples.
    For coarsen_factor >= 10, creates offset versions with 5-pixel intervals.
    """
    dss = []

    # First version: standard coarsening
    ind_coarse = ind_var.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary='trim').mean()
    dep_coarse = dep_var.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary='trim').mean()
    # rechunk for the coarsened versions
    # ind_coarse = ind_coarse.chunk({'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})
    # dep_coarse = dep_coarse.chunk({'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})
    dss.append((ind_coarse, dep_coarse))

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

            dep_shifted = dep_var.loc[{'lat': slice(dep_var.lat.values[offset], dep_var.lat.values[-offset]),
                                       'lon': slice(dep_var.lon.values[offset], dep_var.lon.values[-offset])}]

            # Coarsen the shifted versions
            ind_shifted_coarse = ind_shifted.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary='trim').mean()
            dep_shifted_coarse = dep_shifted.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary='trim').mean()

            # get mode chunk sizes for the coarsened versions
            mode_lat_chunk_size = np.bincount(ind_shifted_coarse.chunks[ind_shifted_coarse.dims.index('lat')]).argmax()
            mode_lon_chunk_size = np.bincount(ind_shifted_coarse.chunks[ind_shifted_coarse.dims.index('lon')]).argmax()

            # rechunk for the coarsened versions
            ind_shifted_coarse = ind_shifted_coarse.chunk({'lat': mode_lat_chunk_size, 'lon': mode_lon_chunk_size})
            dep_shifted_coarse = dep_shifted_coarse.chunk({'lat': mode_lat_chunk_size, 'lon': mode_lon_chunk_size})

            dss.append((ind_shifted_coarse, dep_shifted_coarse))

    return dss


def calc_moving_window(ind_var, dep_var, ind_var_name, dep_var_name,
                       client=None, dep_var_agg=None,
                       coarsen_factor=None, window_radius=1, sname=None):


    if len(dep_var.lat) == 5000 and len(dep_var.lon) == 4750:    
        elev = xr.open_dataset(os.path.join(STORAGE_DIR, '..', 'elevation', 'elevation_regridded.nc'))['elevation']
        ref_ind_var = ind_var
        elev = elev.assign_coords(lat=ref_ind_var.lat, lon=ref_ind_var.lon)
        ind_var = ind_var.where(elev < 1000)
        dep_var = dep_var.where(elev < 1000)

    # Generate list of variable triplets (with offsets if applicable)
    if coarsen_factor:
        dss = _generate_coarsened_versions(ind_var, dep_var, coarsen_factor)
    else:
        dss = [(ind_var, dep_var)]

    os.makedirs(MWA_DAILY_DIR, exist_ok=True)
    out_root = os.path.join(MWA_DAILY_DIR, f'{sname}_mwa_members_parquet')
    os.makedirs(out_root, exist_ok=True)
    expected_versions = len(dss)

    # Process each version
    for version_idx, (ind_var_version, dep_var_version) in enumerate(dss):
        if _has_complete_version_members(out_root, version_idx):
            print(f"Version {version_idx} parquet outputs already exist. Skipping recompute.")
            continue

        version_root = _version_dir(out_root, version_idx)
        if os.path.isdir(version_root):
            # Clear any partial output for this version only.
            shutil.rmtree(version_root)

        suffix = f'_{version_idx}' if expected_versions > 1 else ''
        map_sname = f'{sname}{suffix}'

        if dep_var_agg is not None:
            dep_var_version = aggregate_dependent_variable(dep_var_version, dep_var_agg)

        ind_results_ds, dep_results_ds = moving_window_analysis(
            ind_var=ind_var_version,
            dep_var=dep_var_version,
            window_radius=window_radius,
            ind_var_name=ind_var_name,
            dep_var_name=dep_var_name,
            sname=map_sname
        )

        ind_results = ind_results_ds['ind_results']
        dep_results = dep_results_ds['dep_results']
        _write_member_parquet(ind_results, dep_results, out_root, version_idx, client=client)

        gc.collect()
        print(f"Member parquet output written to {_version_dir(out_root, version_idx)}")

    _write_run_metadata(
        out_root=out_root,
        sname=sname,
        coarsen_factor=coarsen_factor,
        dep_var_agg=dep_var_agg,
        expected_versions=expected_versions,
    )


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

def _check_member_parquets_exist(sname, coarsen_factor=None):
    """Check whether expected per-version ind/dep parquet outputs exist."""
    out_root = os.path.join(MWA_DAILY_DIR, f'{sname}_mwa_members_parquet')
    expected_versions = _expected_version_count(coarsen_factor)
    for version_idx in range(expected_versions):
        if not _has_complete_version_members(out_root, version_idx):
            return False
    return True


def run_mwa(coarsen_factor=None, window_radius=1, dep_var_agg=None, client=None,
            ind_var_name='cc_loss', dep_var_name='heatwaves'):

    ind_var, dep_var = load_ind_dep_data(
        ind_var_name=ind_var_name,
        dep_var_name=dep_var_name,
    )

    # check if parquet output already exists
    agg_name = dep_var_agg if dep_var_agg else "daily"
    sname = f'{dep_var.name}_{agg_name}_vs_{ind_var_name}'
    if coarsen_factor:
        sname += f'_coarsened{coarsen_factor}'

    if SPATH_SUFFIX:
        sname += f'_{SPATH_SUFFIX}'

    if _check_member_parquets_exist(sname=sname, coarsen_factor=coarsen_factor):
        print(f"Parquet member outputs already complete for {sname}. Skipping run.")
        return

    print(f"Output parquet members are missing/incomplete for {sname}. Proceeding with moving window analysis.")
    calc_moving_window(ind_var=ind_var, dep_var=dep_var, 
                        ind_var_name=ind_var_name, dep_var_name=dep_var_name,
                        client=client, window_radius=window_radius,
                        coarsen_factor=coarsen_factor, sname=sname, dep_var_agg=dep_var_agg)
    print(
        f"Moving window analysis completed for {dep_var.name} vs {ind_var_name} "
        f"with coarsen factor {coarsen_factor}."
    )



def run_mwa_coarsen_loop(ind_var_name, dep_var_agg,
                         dep_var_name,
                         client=None):
    for coarsen_factor in COARSEN_FACTORS:
        run_mwa(coarsen_factor=coarsen_factor, window_radius=1, dep_var_agg=dep_var_agg, client=client,
                ind_var_name=ind_var_name, 
                dep_var_name=dep_var_name)
            
#%%
if __name__ == "__main__":
    
    with local_cluster(n_workers=32, threads_per_worker=1, memory_limit='20GB',
                       suppress_dask_warnings=True, use_dask_dashboard=USE_DASK_DASHBOARD) as client:
        run_mwa_coarsen_loop(ind_var_name='cc_loss', dep_var_agg='annual_mean',
                             dep_var_name='tmax', client=client)
        # run_mwa_coarsen_loop(ind_var_name='forest_loss_age', dep_var_agg='annual_mean',
        #                     dep_var_name='tmax', client=client)
        # run_mwa_coarsen_loop(ind_var_name='cc_loss', dep_var_agg='annual_sum',
        #                      dep_var_name='heatwaves_gapfilled', client=client)

