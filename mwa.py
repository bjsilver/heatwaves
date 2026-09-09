#%%
import xarray as xr
import numpy as np
import os
import dask.array as da
from constants import HANSEN_DIR, MWA_DIR, STORAGE_DIR
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
CC_LOSS_EXPLAINED_FRACTION = 0.90
MWA_ALGORITHM_TAG = f'cc-loss-explained-{int(CC_LOSS_EXPLAINED_FRACTION * 100):02d}'
# bump when the parquet output schema changes, to force stale cached versions to recompute
MWA_SCHEMA_VERSION = 2
# minimum fraction of non-NaN pixels a coarsen block must have to produce a valid coarsened value
COARSEN_MIN_VALID_FRACTION = 0.90

TEMP_HANSEN_DIR = '/uolstore/Research/b/b0256/Users/earbsi/hansen/regular_regridded_grid/'


def _coarsen_mean_min_valid(data_array, coarsen_factor, min_valid_fraction=COARSEN_MIN_VALID_FRACTION):
    """Coarsen by mean, masking out blocks with too few non-NaN pixels.

    Sums values and valid-pixel counts in a single coarsen pass (stacked along a temporary
    axis) rather than two separate coarsen calls, since a second full coarsen pass over large
    (often daily-resolution) dask-backed inputs doubles graph-construction time.
    """
    valid = data_array.notnull()
    filled = data_array.fillna(0)
    stacked = xr.concat([filled, valid.astype(filled.dtype)], dim='_stat')
    coarsened_sum = stacked.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary='trim').sum()
    value_sum = coarsened_sum.isel(_stat=0, drop=True)
    valid_count = coarsened_sum.isel(_stat=1, drop=True)
    total_pixels = coarsen_factor * coarsen_factor
    mean = value_sum / valid_count.where(valid_count > 0)
    return mean.where((valid_count / total_pixels) >= min_valid_fraction)


def _load_canopy_cover_and_loss(era=False):
    if era:
        ds = xr.open_zarr(HANSEN_DIR+'regular_regridded_grid/hansen_regridded.zarr')
    else:
        ds = xr.open_zarr(os.path.join(TEMP_HANSEN_DIR, 'hansen_tc_2000_2024_rechunked.zarr'), 
                        consolidated=False)
    tree_cover = ds['tree_cover'].astype(np.float32)
    cc = tree_cover.loc[{'time': slice('2003', '2020')}].rename({'time': 'year'})
    cc_loss = (tree_cover.isel(time=0) - tree_cover).loc[{'time': slice('2003', '2020')}]
    cc_loss = cc_loss.rename({'time': 'year'}).astype(np.float32)
    cc_loss = cc_loss.transpose('year', 'lat', 'lon')
    cc.name = 'cc'
    cc_loss.name = 'cc_loss'
    return cc, cc_loss


def load_ind_dep_data(ind_var_name, dep_var_name, include_cc_loss=False):


    if dep_var_name == 'tmax':
        dep_var = xr.open_zarr(os.path.join(STORAGE_DIR, 'rechunked', 'zhang_zhou_regridded_rechunked.zarr'))['Ta_TMAX']
        dep_var = dep_var.astype(np.float32).where(dep_var != -9999.0) / 10.0  # convert to °C
    elif dep_var_name == 'heatwaves':
        dep_var = xr.open_zarr(os.path.join(STORAGE_DIR, 'heatdays', 'heatwaves_annual_sum_zz_2003-2020.zarr'))['heatwave_days']
    elif dep_var_name == 'heatwaves_smoothed':
        dep_var = xr.open_zarr(os.path.join(STORAGE_DIR, 'heatdays', 'heatwaves_annual_sum_zz_2003-2020_smooth.zarr'))['heatwave_days']
    elif dep_var_name == 'heatwaves_gapfilled':
        dep_var = xr.open_zarr(os.path.join(STORAGE_DIR, 'heatdays', 'heatwaves_annual_sum_zz_2003-2020_gapfilled.zarr'))['heatwave_days']
    elif dep_var_name == 'heatwaves_era5-p95':
        dep_var = xr.open_zarr(os.path.join(STORAGE_DIR, 'heatdays', 'heatwaves_annual_sum_era5_2003-2020.zarr'))['heatwave_days']
    elif dep_var_name == 'absthresh_heatwaves':
        dep_var = xr.open_zarr(os.path.join(STORAGE_DIR, 'heat_metrics', 'heat_metrics.zarr'))['absthresh_heatwaves']
    elif dep_var_name == 'warmest_days':
        dep_var = xr.open_zarr(os.path.join(STORAGE_DIR, 'heat_metrics', 'heat_metrics.zarr'))['warmest_days']
    elif dep_var_name == 'tmax_era5':
        dep_var = load_era5_t2m()
    else:
        raise ValueError(f"Unsupported dependent variable name: {dep_var_name}")
    dep_var.name = dep_var_name

    if ind_var_name == 'cc':
        ind_var, cc_loss = _load_canopy_cover_and_loss()

    elif ind_var_name == 'cc_era5':
        ind_var, cc_loss = _load_canopy_cover_and_loss(era=True)
    ind_var = ind_var.assign_coords(lat=dep_var.lat, lon=dep_var.lon)
    cc_loss = cc_loss.assign_coords(lat=dep_var.lat, lon=dep_var.lon)
    

    if include_cc_loss:
        return ind_var, cc_loss, dep_var
    return ind_var, dep_var

@njit
def _mwa_kernel(cc_block, cc_loss_block, window_radius, loss_explained_fraction):
    cc = cc_block
    cc_loss = cc_loss_block
    wr = window_radius

    t_size = cc.shape[0]
    h_size = cc.shape[1] - 2 * wr
    w_size = cc.shape[2] - 2 * wr

    out = np.full((5, t_size, h_size, w_size), np.nan, dtype=np.float32)

    center_cc = cc[:, wr:wr+h_size, wr:wr+w_size]
    center_cc_loss = cc_loss[:, wr:wr+h_size, wr:wr+w_size]

    max_ind_diff = np.full((t_size, h_size, w_size), np.float32(-np.inf), dtype=np.float32)
    selected_neighbour_cc = np.full((t_size, h_size, w_size), np.nan, dtype=np.float32)
    max_ind_diff_ioffset = np.full((t_size, h_size, w_size), NEIGHBOUR_MAP_FILL_VALUE, dtype=np.int8)
    max_ind_diff_joffset = np.full((t_size, h_size, w_size), NEIGHBOUR_MAP_FILL_VALUE, dtype=np.int8)

    neighbour_cc_sum = np.zeros((t_size, h_size, w_size), dtype=np.float32)
    neighbour_cc_count = np.zeros((t_size, h_size, w_size), dtype=np.int16)

    for di in range(-wr, wr + 1):
        for dj in range(-wr, wr + 1):
            if di == 0 and dj == 0:
                continue
            ri, rj = wr + di, wr + dj

            neighbour_cc = cc[:, ri:ri+h_size, rj:rj+w_size]
            cc_diff = neighbour_cc - center_cc

            finite_neighbour = np.isfinite(neighbour_cc)
            neighbour_cc_sum += np.where(finite_neighbour, neighbour_cc, 0.0)
            neighbour_cc_count += finite_neighbour

            better = cc_diff > max_ind_diff
            max_ind_diff = np.where(better, cc_diff, max_ind_diff)
            selected_neighbour_cc = np.where(better, neighbour_cc, selected_neighbour_cc)
            max_ind_diff_ioffset = np.where(better, np.int8(di), max_ind_diff_ioffset)
            max_ind_diff_joffset = np.where(better, np.int8(dj), max_ind_diff_joffset)


    valid = (
        (max_ind_diff > 0)
        & np.isfinite(center_cc_loss)
        & (center_cc_loss >= loss_explained_fraction * max_ind_diff)
    )

    # np.divide's out=/where= kwargs are unsupported in numba njit code; use np.where instead
    has_neighbour_cc = neighbour_cc_count > 0
    safe_neighbour_cc_count = np.where(has_neighbour_cc, neighbour_cc_count, 1)
    mean_neighbour_cc = np.where(
        has_neighbour_cc, neighbour_cc_sum / safe_neighbour_cc_count, np.nan
    )

    out[0] = np.where(valid, max_ind_diff_ioffset, NEIGHBOUR_MAP_FILL_VALUE)
    out[1] = np.where(valid, max_ind_diff_joffset, NEIGHBOUR_MAP_FILL_VALUE)
    out[2] = np.where(valid, center_cc, np.nan)
    out[3] = np.where(valid, selected_neighbour_cc, np.nan)
    out[4] = np.where(valid, mean_neighbour_cc, np.nan)
    return out

def _get_mwa_neighbour_map(cc, cc_loss, window_radius, sname):
    map_tag = sname
    spath = os.path.join(MWA_DIR, f'neighbour_maps/{map_tag}.zarr')
    if os.path.exists(spath):
        print(f"Neighbour map zarr already exists for {sname}.\nLoading from {spath}.")
        neighbour_maps_ds = xr.open_zarr(spath, consolidated=False)

        years_match = np.array_equal(neighbour_maps_ds['year'].values, cc['year'].values)
        lats_match = np.array_equal(neighbour_maps_ds['lat'].values, cc['lat'].values)
        lons_match = np.array_equal(neighbour_maps_ds['lon'].values, cc['lon'].values)

        fields_match = (
        'selection' in neighbour_maps_ds
        and np.array_equal(
            neighbour_maps_ds['selection_field'].values,
            ['i_offset', 'j_offset', 'center_cc', 'neighbour_cc', 'mean_neighbour_cc'],
        )
    )

    if years_match and lats_match and lons_match and fields_match:
        return neighbour_maps_ds
    else:
        print(
            "Existing neighbour map grid does not match current input grid. "
            "Recomputing and overwriting existing neighbour map store."
        )

    cc_arr = cc.data
    cc_loss_arr = cc_loss.data

    neighbour_maps = da.map_overlap(
        _mwa_kernel,
        cc_arr,
        cc_loss_arr,
        window_radius=window_radius,
        loss_explained_fraction=CC_LOSS_EXPLAINED_FRACTION,
        depth=[
            {0: 0, 1: window_radius, 2: window_radius},
            {0: 0, 1: window_radius, 2: window_radius},
        ],
        boundary=[np.nan, np.nan],
        trim=False,
        dtype=np.float32,
        new_axis=0,
        chunks=((5,),) + cc_arr.chunks
    )
    neighbour_maps_ds = xr.Dataset(
        {
            'selection': (('selection_field', 'year', 'lat', 'lon'), neighbour_maps),
        },
        coords={
            'selection_field': ['i_offset', 'j_offset', 'center_cc', 'neighbour_cc', 'mean_neighbour_cc'],
            'year': cc['year'],
            'lat': cc['lat'],
            'lon': cc['lon'],
        },
    )
    neighbour_maps_ds['selection'].attrs['missing_offset_value'] = int(NEIGHBOUR_MAP_FILL_VALUE)

    encoding = {
        'selection': {
            'dtype': 'float32',
        }
    }

    neighbour_maps_ds.to_zarr(spath, mode='w', consolidated=False, encoding=encoding,
                              zarr_format=2)
    neighbour_maps_ds = xr.open_zarr(spath, consolidated=False)
    return neighbour_maps_ds


def _gather_mwa_results_kernel(arr, neighbour_maps, window_radius):
    t_size = arr.shape[0]
    h_size = arr.shape[1] - 2 * window_radius
    w_size = arr.shape[2] - 2 * window_radius

    

    out = np.full((3, t_size, h_size, w_size), np.nan, dtype=np.float32)
    center_arr = np.full((t_size, h_size, w_size), np.nan, dtype=np.float32)
    neighbour_arr = np.full((t_size, h_size, w_size), np.nan, dtype=np.float32)
    neighbour_sum = np.zeros((t_size, h_size, w_size), dtype=np.float32)
    neighbour_count = np.zeros((t_size, h_size, w_size), dtype=np.int16)


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

            finite_n = np.isfinite(n_arr)
            neighbour_sum += np.where(finite_n, n_arr, 0.0)
            neighbour_count += finite_n

    # mean is over all neighbours, but only meaningful where a neighbour was actually selected
    has_neighbours = neighbour_count > 0
    mean_neighbour_arr = np.full_like(neighbour_sum, np.nan)
    np.divide(neighbour_sum, neighbour_count, out=mean_neighbour_arr, where=has_neighbours)
    mean_neighbour_arr = np.where(np.isfinite(neighbour_arr), mean_neighbour_arr, np.nan)

    out[0] = center_arr
    out[1] = neighbour_arr
    out[2] = mean_neighbour_arr
    return out


def _gather_elev_kernel(elev_block, neighbour_maps, window_radius):
    t_size = elev_block.shape[0]
    h_size = elev_block.shape[1] - 2 * window_radius
    w_size = elev_block.shape[2] - 2 * window_radius

    out = np.full((2, t_size, h_size, w_size), np.nan, dtype=np.float32)
    center_arr = np.full((t_size, h_size, w_size), np.nan, dtype=np.float32)
    neighbour_arr = np.full((t_size, h_size, w_size), np.nan, dtype=np.float32)

    center_vals = elev_block[:, window_radius:window_radius + h_size, window_radius:window_radius + w_size]
    for di in range(-window_radius, window_radius + 1):
        for dj in range(-window_radius, window_radius + 1):
            if di == 0 and dj == 0:
                continue
            ri, rj = window_radius + di, window_radius + dj
            n_arr = elev_block[:, ri:ri + h_size, rj:rj + w_size]
            neighbour_in = (neighbour_maps[0] == di) & (neighbour_maps[1] == dj)

            neighbour_arr = np.where(neighbour_in, n_arr, neighbour_arr)
            center_arr = np.where(neighbour_in, center_vals, center_arr)

    out[0] = center_arr
    out[1] = neighbour_arr
    return out


def gather_mwa_results(dep_var, neighbour_maps, dep_var_name, window_radius):

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
            chunks=((3,),) + dep_arr.chunks,
        )

    elif dep_time_dim == 'time':
        dep_results_list = []
        # dep_var = dep_var.chunk({'time': 73})
        

        for year in neighbour_maps['year'].values:

            dep_var_year = dep_var.sel(time=str(year))
            dep_var_year = dep_var_year.chunk({'time': 73})
            neighbour_maps_year = neighbour_maps.sel(year=year)


            dep_results_year = da.map_overlap(
                _gather_mwa_results_kernel,
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
                chunks=((3,),) + dep_var_year.data.chunks,
            )
            dep_results_list.append(dep_results_year)
            
            
        dep_results = da.concatenate(dep_results_list, axis=1)
    dep_results_ds = xr.Dataset(
        {
            'dep_results': (('position', dep_time_dim, 'lat', 'lon'), 
            dep_results),
        },
        coords={
            'position': ['center', 'neighbour', 'mean_neighbour'],
            dep_time_dim: dep_var[dep_time_dim],
            'lat': dep_var['lat'],
            'lon': dep_var['lon'],
        },
    )

    return dep_results_ds


def gather_elev_results(elev, neighbour_maps, window_radius):
    """Gather centre/selected-neighbour elevation using the ind_var neighbour offsets."""
    elev_broadcast = elev.broadcast_like(neighbour_maps.isel(offset=0).drop_vars('offset'))
    elev_broadcast = elev_broadcast.chunk(
        {dim: neighbour_maps.chunksizes[dim] for dim in elev_broadcast.dims}
    )
    elev_arr = elev_broadcast.astype(np.float32).data

    elev_results = da.map_overlap(
        _gather_elev_kernel,
        elev_arr,
        neighbour_maps.data,
        window_radius=window_radius,
        depth=[
            {0: 0, 1: window_radius, 2: window_radius},
            {0: 0, 1: 0, 2: 0, 3: 0},
        ],
        boundary=[np.nan, NEIGHBOUR_MAP_FILL_VALUE],
        trim=False,
        dtype=np.float32,
        chunks=((2,),) + elev_arr.chunks,
    )
    elev_results_ds = xr.Dataset(
        {
            'elev_results': (('position', 'year', 'lat', 'lon'), elev_results),
        },
        coords={
            'position': ['center', 'neighbour'],
            'year': neighbour_maps['year'],
            'lat': neighbour_maps['lat'],
            'lon': neighbour_maps['lon'],
        },
    )
    return elev_results_ds


def moving_window_analysis(ind_var, cc_loss, dep_var, window_radius, dep_var_name, sname, elev=None):
    """Moving-window analysis on (time, lat, lon) inputs."""

    selection = _get_mwa_neighbour_map(ind_var, cc_loss, window_radius, sname)['selection']
    neighbour_maps = (
        selection.sel(selection_field=['i_offset', 'j_offset'])
        .rename({'selection_field': 'offset'})
        .assign_coords(offset=['i', 'j'])
        .astype(np.int8)
    )
    ind_results_ds = xr.Dataset({
    'ind_centre': selection.sel(selection_field='center_cc').drop_vars('selection_field'),
    'ind_neighbour': selection.sel(selection_field='neighbour_cc').drop_vars('selection_field'),
    'ind_neighbour_mean': selection.sel(selection_field='mean_neighbour_cc').drop_vars('selection_field'),
})

    if elev is not None:
        elev_results_ds = gather_elev_results(elev, neighbour_maps, window_radius)
        ind_results_ds['elev_centre'] = elev_results_ds['elev_results'].sel(position='center').drop_vars('position')
        ind_results_ds['elev_neighbour'] = elev_results_ds['elev_results'].sel(position='neighbour').drop_vars('position')

    dep_results_ds = gather_mwa_results(
        dep_var=dep_var,
        neighbour_maps=neighbour_maps,
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
    mean_neighbour_chunk,
    time_vals,
    lat_vals,
    lon_vals,
    time_col,
    centre_col,
    neighbour_col,
    mean_neighbour_col,
    out_path,
    compression,
    elev_centre_chunk=None,
    elev_centre_col=None,
    elev_neighbour_chunk=None,
    elev_neighbour_col=None,
):
    centre_chunk = np.asarray(centre_chunk, dtype=np.float32)
    neighbour_chunk = np.asarray(neighbour_chunk, dtype=np.float32)
    if mean_neighbour_chunk is not None:
        mean_neighbour_chunk = np.asarray(mean_neighbour_chunk, dtype=np.float32)
    if elev_centre_chunk is not None:
        elev_centre_chunk = np.asarray(elev_centre_chunk, dtype=np.float32)
    if elev_neighbour_chunk is not None:
        elev_neighbour_chunk = np.asarray(elev_neighbour_chunk, dtype=np.float32)

    t_size = centre_chunk.shape[0]
    y_size = centre_chunk.shape[1]
    x_size = centre_chunk.shape[2]

    time_vals = np.asarray(time_vals).astype(str)
    time_rep = np.repeat(time_vals, y_size * x_size)
    lat_rep = np.tile(np.repeat(np.asarray(lat_vals), x_size), t_size)
    lon_rep = np.tile(np.asarray(lon_vals), t_size * y_size)

    data = {
        time_col: time_rep,
        'lat': lat_rep,
        'lon': lon_rep,
        centre_col: centre_chunk.ravel(),
        neighbour_col: neighbour_chunk.ravel(),
    }
    if mean_neighbour_chunk is not None:
        if mean_neighbour_col is None:
            raise ValueError(
                'mean_neighbour_col is required when mean_neighbour_chunk is provided.'
            )
        data[mean_neighbour_col] = mean_neighbour_chunk.ravel()
    if elev_centre_chunk is not None:
        if elev_centre_col is None:
            raise ValueError('elev_centre_col is required when elev_centre_chunk is provided.')
        data[elev_centre_col] = elev_centre_chunk.ravel()
    if elev_neighbour_chunk is not None:
        if elev_neighbour_col is None:
            raise ValueError('elev_neighbour_col is required when elev_neighbour_chunk is provided.')
        data[elev_neighbour_col] = elev_neighbour_chunk.ravel()

    df = pd.DataFrame(data)
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
    mean_neighbour_arr=None,
    mean_neighbour_col=None,
    elev_centre_arr=None,
    elev_centre_col=None,
    elev_neighbour_arr=None,
    elev_neighbour_col=None,
):
    centre_delayed = centre_arr.to_delayed()
    neighbour_delayed = neighbour_arr.to_delayed()
    mean_neighbour_delayed = None
    if mean_neighbour_arr is not None:
        mean_neighbour_delayed = mean_neighbour_arr.to_delayed()
    elev_centre_delayed = None
    if elev_centre_arr is not None:
        elev_centre_delayed = elev_centre_arr.to_delayed()
    elev_neighbour_delayed = None
    if elev_neighbour_arr is not None:
        elev_neighbour_delayed = elev_neighbour_arr.to_delayed()
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
                mean_neighbour_delayed[ti, yi, xi] if mean_neighbour_delayed is not None else None,
                time_vals[t0:t1],
                lat_vals[y0:y1],
                lon_vals[x0:x1],
                time_col,
                centre_col,
                neighbour_col,
                mean_neighbour_col,
                out_path,
                compression,
                elev_centre_chunk=elev_centre_delayed[ti, yi, xi] if elev_centre_delayed is not None else None,
                elev_centre_col=elev_centre_col,
                elev_neighbour_chunk=elev_neighbour_delayed[ti, yi, xi] if elev_neighbour_delayed is not None else None,
                elev_neighbour_col=elev_neighbour_col,
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

    ind_centre = ind_results['ind_centre'].astype(np.float32).data
    ind_neighbour = ind_results['ind_neighbour'].astype(np.float32).data
    ind_neighbour_mean = ind_results['ind_neighbour_mean'].astype(np.float32).data
    has_elev = 'elev_centre' in ind_results
    ind_tasks = _build_chunk_write_tasks(
    centre_arr=ind_centre,
    neighbour_arr=ind_neighbour,
    mean_neighbour_arr=ind_neighbour_mean,
    time_vals=ind_results['year'].values,
    lat_vals=ind_results['lat'].values,
    lon_vals=ind_results['lon'].values,
    time_col='year',
    centre_col='ind_centre',
    neighbour_col='ind_neighbour',
    mean_neighbour_col='ind_neighbour_mean',
    out_dir=ind_out,
    compression=compression,
    elev_centre_arr=ind_results['elev_centre'].astype(np.float32).data if has_elev else None,
    elev_centre_col='elev_centre' if has_elev else None,
    elev_neighbour_arr=ind_results['elev_neighbour'].astype(np.float32).data if has_elev else None,
    elev_neighbour_col='elev_neighbour' if has_elev else None,
    )

    dep_time_dim = 'year' if 'year' in dep_results.dims else 'time'
    dep_centre = dep_results.sel(position='center').reset_coords('position', drop=True).astype(np.float32).data
    dep_neighbour = dep_results.sel(position='neighbour').reset_coords('position', drop=True).astype(np.float32).data
    dep_neighbour_mean = dep_results.sel(position='mean_neighbour').reset_coords('position', drop=True).astype(np.float32).data
    dep_tasks = _build_chunk_write_tasks(
        centre_arr=dep_centre,
        neighbour_arr=dep_neighbour,
        mean_neighbour_arr=dep_neighbour_mean,
        time_vals=dep_results[dep_time_dim].values,
        lat_vals=dep_results['lat'].values,
        lon_vals=dep_results['lon'].values,
        time_col=dep_time_dim,
        centre_col='dep_centre',
        neighbour_col='dep_neighbour',
        mean_neighbour_col='dep_neighbour_mean',
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


def _run_metadata_matches_current_algorithm(out_root):
    metadata_path = os.path.join(out_root, '_run_metadata.json')
    if not os.path.isfile(metadata_path):
        return False
    with open(metadata_path, encoding='utf-8') as fobj:
        metadata = json.load(fobj)
    return (
        metadata.get('algorithm') == MWA_ALGORITHM_TAG
        and metadata.get('cc_loss_explained_fraction') == CC_LOSS_EXPLAINED_FRACTION
        and metadata.get('schema_version') == MWA_SCHEMA_VERSION
    )


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
        'ind_var_name': 'cc',
        'algorithm': MWA_ALGORITHM_TAG,
        'cc_loss_explained_fraction': CC_LOSS_EXPLAINED_FRACTION,
        'schema_version': MWA_SCHEMA_VERSION,
        'expected_versions': expected_versions,
        'completed_versions': completed_versions,
        'complete': len(completed_versions) == expected_versions,
    }
    with open(os.path.join(out_root, '_run_metadata.json'), 'w', encoding='utf-8') as fobj:
        json.dump(metadata, fobj, indent=2)
def _generate_coarsened_versions(ind_var, cc_loss, dep_var, coarsen_factor, elev=None):
    """Generate coarsened cc, cumulative-loss, dependent-variable, and elevation versions.

    Returns a list of (cc_coarse, cc_loss_coarse, dep_coarse, elev_coarse) tuples, where
    elev_coarse is None throughout if elev is None. For coarsen_factor >= 10, creates offset
    versions with 5-pixel intervals.
    """
    dss = []

    # First version: standard coarsening
    ind_coarse = _coarsen_mean_min_valid(ind_var, coarsen_factor)
    cc_loss_coarse = _coarsen_mean_min_valid(cc_loss, coarsen_factor)
    dep_coarse = _coarsen_mean_min_valid(dep_var, coarsen_factor)
    elev_coarse = None
    if elev is not None:
        elev_coarse = _coarsen_mean_min_valid(elev, coarsen_factor)
    # rechunk for the coarsened versions
    # ind_coarse = ind_coarse.chunk({'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})
    # dep_coarse = dep_coarse.chunk({'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})
    dss.append((ind_coarse, cc_loss_coarse, dep_coarse, elev_coarse))

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
            cc_loss_shifted = cc_loss.loc[
                {
                    'lat': slice(cc_loss.lat.values[offset], cc_loss.lat.values[-offset]),
                    'lon': slice(cc_loss.lon.values[offset], cc_loss.lon.values[-offset]),
                }
            ]

            dep_shifted = dep_var.loc[{'lat': slice(dep_var.lat.values[offset], dep_var.lat.values[-offset]),
                                       'lon': slice(dep_var.lon.values[offset], dep_var.lon.values[-offset])}]
            elev_shifted = None
            if elev is not None:
                elev_shifted = elev.loc[
                    {
                        'lat': slice(elev.lat.values[offset], elev.lat.values[-offset]),
                        'lon': slice(elev.lon.values[offset], elev.lon.values[-offset]),
                    }
                ]

            # Coarsen the shifted versions
            ind_shifted_coarse = _coarsen_mean_min_valid(ind_shifted, coarsen_factor)
            cc_loss_shifted_coarse = _coarsen_mean_min_valid(cc_loss_shifted, coarsen_factor)
            dep_shifted_coarse = _coarsen_mean_min_valid(dep_shifted, coarsen_factor)
            elev_shifted_coarse = None
            if elev_shifted is not None:
                elev_shifted_coarse = _coarsen_mean_min_valid(elev_shifted, coarsen_factor)

            # get mode chunk sizes for the coarsened versions
            mode_lat_chunk_size = np.bincount(ind_shifted_coarse.chunks[ind_shifted_coarse.dims.index('lat')]).argmax()
            mode_lon_chunk_size = np.bincount(ind_shifted_coarse.chunks[ind_shifted_coarse.dims.index('lon')]).argmax()

            # rechunk for the coarsened versions
            ind_shifted_coarse = ind_shifted_coarse.chunk({'lat': mode_lat_chunk_size, 'lon': mode_lon_chunk_size})
            cc_loss_shifted_coarse = cc_loss_shifted_coarse.chunk(
                {'lat': mode_lat_chunk_size, 'lon': mode_lon_chunk_size}
            )
            dep_shifted_coarse = dep_shifted_coarse.chunk({'lat': mode_lat_chunk_size, 'lon': mode_lon_chunk_size})
            if elev_shifted_coarse is not None:
                elev_shifted_coarse = elev_shifted_coarse.chunk(
                    {'lat': mode_lat_chunk_size, 'lon': mode_lon_chunk_size}
                )

            dss.append((ind_shifted_coarse, cc_loss_shifted_coarse, dep_shifted_coarse, elev_shifted_coarse))

    return dss


def calc_moving_window(ind_var, cc_loss, dep_var, dep_var_name,
                       client=None, dep_var_agg=None,
                       coarsen_factor=None, window_radius=1, sname=None):


    if len(dep_var.lat) == 5000 and len(dep_var.lon) == 4750:    
        elev = xr.open_dataset(os.path.join(STORAGE_DIR, '..', 'elevation', 'elevation_regridded.nc'))['elevation']
        ref_ind_var = ind_var
        elev = elev.assign_coords(lat=ref_ind_var.lat, lon=ref_ind_var.lon)
        ind_var = ind_var.where(elev < 1000)
        cc_loss = cc_loss.where(elev < 1000)
        dep_var = dep_var.where(elev < 1000)
    else:
        elev = None

    # Generate list of variable triplets (with offsets if applicable)
    if coarsen_factor:
        dss = _generate_coarsened_versions(ind_var, cc_loss, dep_var, coarsen_factor, elev=elev)
    else:
        dss = [(ind_var, cc_loss, dep_var, elev)]

    os.makedirs(MWA_DIR, exist_ok=True)
    out_root = os.path.join(MWA_DIR, f'{sname}_mwa_members_parquet')
    os.makedirs(out_root, exist_ok=True)
    expected_versions = len(dss)
    reuse_existing_members = _run_metadata_matches_current_algorithm(out_root)

    # Process each version
    for version_idx, (ind_var_version, cc_loss_version, dep_var_version, elev_version) in enumerate(dss):
        if reuse_existing_members and _has_complete_version_members(out_root, version_idx):
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
            cc_loss=cc_loss_version,
            dep_var=dep_var_version,
            window_radius=window_radius,
            dep_var_name=dep_var_name,
            sname=map_sname,
            elev=elev_version,
        )

        ind_results = ind_results_ds
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
    out_root = os.path.join(MWA_DIR, f'{sname}_mwa_members_parquet')
    if not _run_metadata_matches_current_algorithm(out_root):
        return False
    expected_versions = _expected_version_count(coarsen_factor)
    for version_idx in range(expected_versions):
        if not _has_complete_version_members(out_root, version_idx):
            return False
    return True


def run_mwa(coarsen_factor=None, window_radius=1, dep_var_agg=None, client=None,
        ind_var_name='cc', dep_var_name='heatwaves'):

    ind_var, cc_loss, dep_var = load_ind_dep_data(
        ind_var_name=ind_var_name,
        dep_var_name=dep_var_name,
    include_cc_loss=True,
    )

    # check if parquet output already exists
    agg_name = dep_var_agg if dep_var_agg else "daily"
    sname = f'{dep_var.name}_{agg_name}_vs_cc'
    if coarsen_factor:
        sname += f'_coarsened{coarsen_factor}'

    if SPATH_SUFFIX:
        sname += f'_{SPATH_SUFFIX}'

    if _check_member_parquets_exist(sname=sname, coarsen_factor=coarsen_factor):
        print(f"Parquet member outputs already complete for {sname}. Skipping run.")
        return

    print(f"Output parquet members are missing/incomplete for {sname}. Proceeding with moving window analysis.")
    calc_moving_window(ind_var=ind_var, cc_loss=cc_loss, dep_var=dep_var,
                        dep_var_name=dep_var_name,
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

        # run_mwa_coarsen_loop(ind_var_name='cc', dep_var_agg='annual_mean',
        #                      dep_var_name='tmax', client=client)

        run_mwa_coarsen_loop(ind_var_name='cc', dep_var_agg='annual_sum',
                            dep_var_name='heatwaves_gapfilled', client=client)

        # run_mwa_coarsen_loop(ind_var_name='cc', 
        #                      dep_var_name='absthresh_heatwaves',
        #                      dep_var_agg='annual_sum',
        #                      client=client)

        # run_mwa_coarsen_loop(ind_var_name='cc', 
        #                      dep_var_name='warmest_days',
        #                      dep_var_agg='annual_mean',
        #                      client=client)
        