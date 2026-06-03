#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import os
from contextlib import contextmanager
import pandas as pd
import numpy as np
from numba import njit, prange
import xarray as xr
import dask.array as da_lazy
from pathlib import Path
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from constants import FIGDIR


# --- CONFIGURATION ---
DPATH = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/rechunked/'
PERCENTILES_DIR = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/percentiles/'
OUT_ZARR = os.path.join(PERCENTILES_DIR, 'zz_tmax_95pc.zarr')
PAD = 7
PERCENTILE = 0.95

USE_LOCAL_CLUSTER = True
N_WORKERS = 32
THREADS_PER_WORKER = 4
MEMORY_LIMIT = '16GB'

# Ensure output directory exists.
Path(PERCENTILES_DIR).mkdir(parents=True, exist_ok=True)

# --- NUMBA COMPUTATIONAL KERNELS ---


def load_tmax_da():
    print(f"Opening input Zarr: {DPATH}tmax_regridded.zarr")
    # Use consolidated=False to support stores without .zmetadata.
    ds = xr.open_zarr(DPATH + 'tmax_regridded.zarr', chunks={}, consolidated=False)
    da = ds['Ta_TMAX']
    # check if zz time has duplicates
    if len(da.time) != len(np.unique(da.time)):
        print("Warning: Duplicate time entries found in zz dataset. Removing duplicates.")
        da = da.drop_duplicates('time')

    da.coords['lat'] = np.where(np.isnan(da.lat.values), 0, da.lat.values)

    if 'time' in da.dims:
        year_vals = da['time'].dt.year.data
        doy_vals = da['time'].dt.dayofyear.data
        da = da.assign_coords(year=('time', year_vals), doy=('time', doy_vals))
        da = da.set_index(time=['year', 'doy']).unstack('time').transpose('year', 'doy', 'lat', 'lon')

    if 'doy' not in da.coords:
        da = da.assign_coords(doy=np.arange(1, da.sizes['doy'] + 1))

    da = da.transpose('year', 'doy', 'lat', 'lon')

    print('Using native chunks from source/transform pipeline:')
    print(dict(zip(da.dims, da.chunks)))

    min_doy_chunk = min(da.chunks[da.get_axis_num('doy')])
    if min_doy_chunk <= 2 * PAD:
        raise ValueError(
            f'doy chunk size must be > {2 * PAD} for map_overlap with PAD={PAD}. '
            f'Current minimum doy chunk is {min_doy_chunk}.'
        )
    
    da = da / 10.0  # Convert from tenths of degrees to degrees Celsius.

    return da


@contextmanager
def maybe_local_cluster():
    if not USE_LOCAL_CLUSTER:
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




def compute_zz_trend(tmax):
    annual_mean = tmax.mean(dim='doy')
    trend = theilsen_trend_da(annual_mean, dim='year')
    trend.name = 'Ta_TMAX_trend'
    return trend


@njit(cache=True)
def _theilsen_1d(y, x):
    n = y.size
    valid_count = 0
    for i in range(n):
        if np.isfinite(y[i]) and np.isfinite(x[i]):
            valid_count += 1

    if valid_count < 2:
        return np.nan

    yv = np.empty(valid_count, dtype=np.float64)
    xv = np.empty(valid_count, dtype=np.float64)
    idx = 0
    for i in range(n):
        if np.isfinite(y[i]) and np.isfinite(x[i]):
            yv[idx] = y[i]
            xv[idx] = x[i]
            idx += 1

    pair_count = valid_count * (valid_count - 1) // 2
    slopes = np.empty(pair_count, dtype=np.float64)
    k = 0

    for i in range(valid_count - 1):
        xi = xv[i]
        yi = yv[i]
        for j in range(i + 1, valid_count):
            dx = xv[j] - xi
            if dx != 0.0:
                slopes[k] = (yv[j] - yi) / dx
                k += 1

    if k == 0:
        return np.nan

    return np.median(slopes[:k])


def _theilsen_1d_wrapper(y, x):
    return _theilsen_1d(y.astype(np.float64), x.astype(np.float64))


def theilsen_trend_da(da, dim='year'):
    # Core dimensions in apply_ufunc with dask='parallelized' must be single-chunk.
    da = da.chunk({dim: -1})
    x = da[dim].astype(np.float64)
    trend = xr.apply_ufunc(
        _theilsen_1d_wrapper,
        da,
        x,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.float64],
    )
    return trend

def calculate_theilsen_trend(tmax):
    annual_mean = tmax.mean(dim='doy')
    annual_mean = annual_mean.chunk({'year': -1})
    trend = theilsen_trend_da(annual_mean, dim='year')
    trend.name = 'Ta_TMAX_trend'
    return trend


def calculate_seasonal_trend(tmax):
    seas = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11]}
    dates = pd.date_range(start='2001-01-01', end='2001-12-31', freq='D')
    for season, months in seas.items():
        doys_in_season = dates[dates.month.isin(months)].dayofyear

        season_da = tmax.where(tmax['doy'].isin(doys_in_season), drop=True)
        season_trend = calculate_theilsen_trend(season_da)
        season_trend.name = f'Ta_TMAX_trend_{season}'
        save_results(season_trend, os.path.join(PERCENTILES_DIR, f'zz_tmax_ts_trend_{season}.zarr'))

def save_results(da, out_path=OUT_ZARR):
    ds_out = da.to_dataset(name=da.name)
    print(f'Writing output to {out_path}...')
    with ProgressBar():
        ds_out.to_zarr(out_path, mode='w', consolidated=True)



#%%



if __name__ == "__main__":
    with maybe_local_cluster():
        tmax = load_tmax_da()
        trend = compute_zz_trend(tmax)
        ts_trend = calculate_theilsen_trend(tmax)
        save_results(trend, os.path.join(PERCENTILES_DIR, 'zz_tmax_trend.zarr'))
        save_results(ts_trend, os.path.join(PERCENTILES_DIR, 'zz_tmax_ts_trend.zarr'))
        calculate_seasonal_trend(tmax)
        print('Processing Complete.')

#%%
# trend = xr.open_zarr(os.path.join(PERCENTILES_DIR, 'zz_tmax_trend.zarr'))['Ta_TMAX_trend']
# ts_trend = xr.open_zarr(os.path.join(PERCENTILES_DIR, 'zz_tmax_ts_trend.zarr'))['Ta_TMAX_trend']


#%%
def plot_trend_map(trend_zarr):
    dpath = os.path.join(PERCENTILES_DIR, trend_zarr)
    if len(trend_zarr.split('_')) == 5:
        months = trend_zarr.split('_')[4].split('.')[0]
        trend = xr.open_zarr(dpath)['Ta_TMAX_trend_' + months]
    else:
        months = ''
        trend = xr.open_zarr(dpath)['Ta_TMAX_trend']
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(projection=ccrs.PlateCarree())

    lim = max(abs(trend.quantile(.05).compute().item()), abs(trend.quantile(.95).compute().item()))
    trend.plot.imshow(ax=ax, cmap='coolwarm', vmin=-lim, vmax=lim,
                    cbar_kwargs={'label': 'TMAX Trend (°C/year)',
                                'shrink': 0.5})
    ax.set_title('tmax trend 2003-2020 '  + months)
    fig.savefig(os.path.join(FIGDIR, 'percentiles', trend_zarr.replace('.zarr', '.png')), dpi=500,
                bbox_inches='tight')

trend_zarrs = ['zz_tmax_trend.zarr', 'zz_tmax_ts_trend.zarr', 'zz_tmax_ts_trend_DJF.zarr', 'zz_tmax_ts_trend_MAM.zarr', 'zz_tmax_ts_trend_JJA.zarr', 'zz_tmax_ts_trend_SON.zarr']
for tz in trend_zarrs:
    plot_trend_map(tz)

#%%

# with maybe_local_cluster():
bit = tmax.loc[{'lat':slice(-15, -10), 'lon':slice(-45,-40)}]
bit_annual  = bit.mean(dim='doy').compute()