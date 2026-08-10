#%%

import xarray as xr
import numpy as np
import os
import dask.array as da
from constants import HANSEN_DIR, MWA_DAILY_DIR, STORAGE_DIR, LAT_CHUNK_SIZE, LON_CHUNK_SIZE
import gc
from ufuncs import local_cluster
from calc_era5_percentiles import load_era5_t2m
from numba import njit
from functools import partial

COARSEN_FACTORS = [None]#, 2, 5, 10, 25]
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
    if not os.path.exists(spath):
       
        ind_arr = ind_var.data

        neighbour_maps = da.map_overlap(
            _mwa_kernel,
            ind_arr,
            window_radius=window_radius,
            depth={0: 0, 1: window_radius, 2: window_radius},
            boundary=np.nan,
            trim=True,
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
    else:
        print(f"Neighbour map zarr already exists for {sname}.\nLoading from {spath}.")
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
        boundary=np.nan,
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
            boundary=np.nan,
            trim=False,
            dtype=np.float32,
            chunks=((2,),) + dep_arr.chunks,
        )

    elif dep_time_dim == 'time':
        dep_results_list = []
        for year in neighbour_maps['year'].values:


            dep_var_year = dep_var.sel(time=str(year))
            dep_var_year = dep_var_year.chunk({'time':73})
            neighbour_maps_year = neighbour_maps.sel(year=year)
            # give neighbour maps a time dimension to match dep_var
            neighbour_maps_year = neighbour_maps_year.expand_dims(
                time=dep_var_year.time
            ).transpose('offset', 'time', 'lat', 'lon')
            neighbour_maps_year = neighbour_maps_year.chunk({'offset':2, 'time': 73})

            dep_results_year = da.map_overlap(
                _gather_mwa_results_kernel,
                dep_var_year.data,
                neighbour_maps_year.data,
                window_radius=window_radius,
                depth=[
                    {0: 0, 1: window_radius, 2: window_radius},
                    {0: 0, 1: 0, 2: 0, 3: 0},
                ],
                boundary=np.nan,
                trim=False,
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
    ind_coarse = ind_coarse.chunk({'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})
    dep_coarse = dep_coarse.chunk({'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})
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

            # rechunk for the coarsened versions
            ind_shifted_coarse = ind_shifted_coarse.chunk({'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})
            dep_shifted_coarse = dep_shifted_coarse.chunk({'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})

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

    # Process each version
    for version_idx, (ind_var_version, dep_var_version) in enumerate(dss):
        # Add version suffix to output paths if there are multiple versions
        if len(dss) > 1:
            out_path = os.path.join(MWA_DAILY_DIR, f'{sname}_mwa_results_{version_idx}')
        else:
            out_path = os.path.join(MWA_DAILY_DIR, f'{sname}_mwa_results')

        if dep_var_agg is not None:
            dep_var_version = aggregate_dependent_variable(dep_var_version, dep_var_agg)

        ind_results_ds, dep_results_ds = moving_window_analysis(
            ind_var=ind_var_version,
            dep_var=dep_var_version,
            window_radius=window_radius,
            ind_var_name=ind_var_name,
            dep_var_name=dep_var_name,
            sname=sname
        )

        ind_results_ds.to_zarr(out_path+ '_ind.zarr', mode='w', consolidated=False)
        dep_results_ds.to_zarr(out_path + '_dep.zarr', mode='w', consolidated=False)

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


def run_mwa(coarsen_factor=None, window_radius=1, dep_var_agg=None, client=None,
            ind_var_name='cc_loss', dep_var_name='heatwaves'):

    ind_var, dep_var = load_ind_dep_data(
        ind_var_name=ind_var_name,
        dep_var_name=dep_var_name,
    )

    # check if output zarr already exists
    agg_name = dep_var_agg if dep_var_agg else "daily"
    sname = f'{dep_var.name}_{agg_name}_vs_{ind_var_name}'
    if coarsen_factor:
        sname += f'_coarsened{coarsen_factor}'
    if SPATH_SUFFIX:
        sname += f'_{SPATH_SUFFIX}'

    if not _check_zarrs_exist(sname):
        print(f"Output zarrs do not exist for {sname}. Proceeding with moving window analysis.")
        calc_moving_window(ind_var=ind_var, dep_var=dep_var, 
                           ind_var_name=ind_var_name, dep_var_name=dep_var_name,
                            client=client, window_radius=window_radius,
                           coarsen_factor=coarsen_factor, sname=sname, dep_var_agg=dep_var_agg)
        print(
            f"Moving window analysis completed for {dep_var.name} vs {ind_var_name} "
            f"with coarsen factor {coarsen_factor}."
        )
    else:
        print(f"Output zarrs already exist for {sname}. Skipping moving window analysis.")


def run_mwa_coarsen_loop(ind_var_name, dep_var_agg,
                         dep_var_name,
                         client=None):
    for coarsen_factor in COARSEN_FACTORS:
        run_mwa(coarsen_factor=coarsen_factor, window_radius=1, dep_var_agg=dep_var_agg, client=client,
                ind_var_name=ind_var_name, 
                dep_var_name=dep_var_name)
            
#%%
if __name__ == "__main__":
    
    with local_cluster(n_workers=24, threads_per_worker=1, memory_limit='18GB',
                       suppress_dask_warnings=False, use_dask_dashboard=USE_DASK_DASHBOARD) as client:
        run_mwa_coarsen_loop(ind_var_name='cc_loss', dep_var_agg=None,
                             dep_var_name='tmax', client=client)
        # run_mwa_coarsen_loop(ind_var_name='forest_loss_age', dep_var_agg='annual_mean',
        #                     dep_var_name='tmax', client=client)
        # run_mwa_coarsen_loop(ind_var_name='cc_loss', dep_var_agg='annual_sum',
        #                      dep_var_name='heatwaves_gapfilled', client=client)