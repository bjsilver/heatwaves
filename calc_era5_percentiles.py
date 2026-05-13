#%%
import xarray as xr
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from numba import njit, prange
import os
from constants import AMAZON_BBOX, ERA5_PATH
from ufuncs import lon_to_180
from dask.distributed import Client, LocalCluster

ERA_YEAR_RANGE = range(1970, 2000)


PAD = 7
PERCENTILE = 0.95
ABSOLUTE_THRESHOLD_C = 28.5
HEATWAVE_DAY_THRESH = 3

USE_LOCAL_CLUSTER = True
N_WORKERS = 4
THREADS_PER_WORKER = 4
MEMORY_LIMIT = '80GB'

ERA5_OPEN_CHUNKS = {'time': 365}
ERA5_ANALYSIS_CHUNKS = {'time': 365, 'lat': 120, 'lon': 120}
ERA5_YEAR_DOY_CHUNKS = {'year': 1, 'doy': 365, 'lat': 120, 'lon': 120}
OUTPUT_PATH = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/percentiles/era5_percentile_1970-1999.zarr'

@njit
def _quantile_skipna_1d(vals, q):
    n = vals.size
    buf = np.empty(n, dtype=np.float64)
    m = 0
    for i in range(n):
        v = vals[i]
        if not np.isnan(v):
            buf[m] = v
            m += 1

    if m == 0:
        return np.nan

    a = np.sort(buf[:m])
    if m == 1:
        return a[0]

    h = (m - 1) * q
    lo = int(np.floor(h))
    hi = int(np.ceil(h))
    if lo == hi:
        return a[lo]

    w = h - lo
    return a[lo] * (1.0 - w) + a[hi] * w

@njit(parallel=True)
def quantile_over_year_window_skipna(arr, pad, q):
    ny, nd, nlat, nlon = arr.shape
    nw = 2 * pad + 1
    out = np.empty((nd, nlat, nlon), dtype=np.float64)

    for d in prange(nd):
        for i in range(nlat):
            for j in range(nlon):
                vals = np.empty(ny * nw, dtype=np.float64)
                k = 0
                for y in range(ny):
                    for w in range(-pad, pad + 1):
                        doy_idx = (d + w) % nd
                        vals[k] = arr[y, doy_idx, i, j]
                        k += 1
                out[d, i, j] = _quantile_skipna_1d(vals, q)
    return out

@contextmanager
def maybe_local_cluster():
    if not USE_LOCAL_CLUSTER:
        yield None
        return

    if LocalCluster is None or Client is None:
        print('dask.distributed not available; running without LocalCluster')
        yield None
        return

    with LocalCluster(
        n_workers=N_WORKERS,
        threads_per_worker=THREADS_PER_WORKER,
        memory_limit=MEMORY_LIMIT,
        dashboard_address=':7878',
    ) as cluster:
        with Client(cluster) as client:
            print(f'LocalCluster started: {client.dashboard_link}')
            yield client

def load_era5_skt_celsius():
    da = xr.open_mfdataset(
        ERA5_PATH + 'era5_land_skinT_dailymax*.nc',
        chunks=ERA5_OPEN_CHUNKS,
        parallel=False,
    )['skt'].astype(np.float32)
    da = da.rename({'valid_time': 'time'})
    da = da.sel(time=slice(f'{ERA_YEAR_RANGE.start}-01-01', f'{ERA_YEAR_RANGE.stop - 1}-12-31'))
    da = lon_to_180(da)
    da = da.sortby('latitude')
    da = da.sel(AMAZON_BBOX)
    da = da.rename({'latitude': 'lat', 'longitude': 'lon'})
    da = da.chunk(ERA5_ANALYSIS_CHUNKS)
    da = (da - np.float32(273.15)).astype(np.float32)
    da.name = 'skt_c'
    return da

def reshape_to_year_doy(da):
    is_feb29 = (da.time.dt.month == 2) & (da.time.dt.day == 29)
    da = da.sel(time=~is_feb29)

    years = da.time.dt.year.values.astype(np.int32)
    doy = pd.Series(years).groupby(years).cumcount().to_numpy(dtype=np.int16) + 1

    counts = pd.Series(years).value_counts().sort_index()
    bad_years = counts[counts != 365]
    if not bad_years.empty:
        raise ValueError(
            'All retained years must contain exactly 365 days after leap-day removal. '
            f'Problem years: {bad_years.to_dict()}'
        )

    da = da.assign_coords(year=('time', years), doy=('time', doy))
    da = da.set_index(time=['year', 'doy']).unstack('time')
    da = da.transpose('year', 'doy', 'lat', 'lon')
    da = da.sortby('year')
    da = da.chunk(ERA5_YEAR_DOY_CHUNKS)
    da.name = 'skt_c'
    return da

def stack_time(heatdays):
    years = heatdays['year'].values.astype(int)
    doy_size = heatdays.sizes['doy']
    tindex = np.concatenate([
        xr.date_range(
            start=f'{year}-01-01',
            periods=doy_size,
            freq='D',
            calendar='noleap',
            use_cftime=True,
        ).values
        for year in years
    ])

    heatdays = heatdays.stack(time=('year', 'doy'), create_index=False)
    heatdays = heatdays.reset_coords(drop=True)
    heatdays = heatdays.assign_coords(time=tindex)
    return heatdays

def remove_short_runs(heatdays, thresh=HEATWAVE_DAY_THRESH):
    is_run_end = heatdays
    for k in range(1, thresh):
        is_run_end = is_run_end & heatdays.shift(time=k, fill_value=False)

    mask = is_run_end
    for k in range(1, thresh):
        mask = mask | is_run_end.shift(time=-k, fill_value=False)

    return heatdays & mask

def calculate_percentile95(era_year_doy):
    print('Computing 95th-percentile climatology...')
    with ProgressBar():
        arr = era_year_doy.transpose('year', 'doy', 'lat', 'lon').compute().values

    p95_vals = quantile_over_year_window_skipna(arr, PAD, PERCENTILE).astype(np.float32)

    return xr.DataArray(
        p95_vals,
        dims=('doy', 'lat', 'lon'),
        coords={
            'doy': era_year_doy['doy'].values,
            'lat': era_year_doy['lat'].values,
            'lon': era_year_doy['lon'].values,
        },
        name='skt_p95',
    )

def calculate_heatdays(era_year_doy, p95):
    print('Calculating heatdays...')
    heatdays = (era_year_doy > ABSOLUTE_THRESHOLD_C) & (era_year_doy > p95)
    heatdays = heatdays.astype(np.bool_)
    heatdays.name = 'heatdays'
    return heatdays

def calculate_heatwaves(heatdays, thresh=HEATWAVE_DAY_THRESH):
    print('Removing short runs to isolate heatwaves...')
    heatdays_time = stack_time(heatdays)
    heatwaves = remove_short_runs(heatdays_time, thresh=thresh)
    heatwaves.name = 'heatwaves'
    return heatwaves

def run_pipeline():
    print('Loading ERA5 skin temperature...')
    era = load_era5_skt_celsius()

    print('Reshaping ERA5 data to year/doy...')
    era_year_doy = reshape_to_year_doy(era)

    p95 = calculate_percentile95(era_year_doy)
    return p95

def save_heatwaves(heatwaves, output_path=OUTPUT_PATH):
    print(f'Saving final heatwaves to {output_path}...')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with ProgressBar():
        heatwaves.to_dataset(name='heatwaves').to_zarr(
            output_path,
            mode='w',
            consolidated=True,
        )

#%%
if __name__ == '__main__':
    with maybe_local_cluster():
        p95 = run_pipeline()
        save_heatwaves(p95, output_path=OUTPUT_PATH)