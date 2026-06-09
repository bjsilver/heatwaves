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

USE_LOCAL_CLUSTER = True
N_WORKERS = 64
THREADS_PER_WORKER = 1
MEMORY_LIMIT = '4GB'
BOOTSTRAP_REPS = 250
BOOTSTRAP_SEED = 42

# Ensure output directory exists.
Path(PERCENTILES_DIR).mkdir(parents=True, exist_ok=True)

# --- NUMBA COMPUTATIONAL KERNELS ---


def load_tmax_da():
    print(f"Opening input Zarr: {DPATH}zhang_zhou_regridded_rechunked.zarr")
    # Use consolidated=False to support stores without .zmetadata.
    ds = xr.open_zarr(DPATH + 'zhang_zhou_regridded_rechunked.zarr', chunks={}, consolidated=False)
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
def maybe_local_cluster(n_workers=N_WORKERS, threads_per_worker=THREADS_PER_WORKER, memory_limit=MEMORY_LIMIT):
    if not USE_LOCAL_CLUSTER:
        yield None
        return

    with LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        dashboard_address=':7878',
    ) as cluster:
        with Client(cluster) as client:
            print(f'LocalCluster started: {client.dashboard_link}')
            yield client

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


@njit(cache=True)
def _bootstrap_theilsen_1d(y, x, n_boot, seed):
    n = y.size
    valid_count = 0
    for i in range(n):
        if np.isfinite(y[i]) and np.isfinite(x[i]):
            valid_count += 1

    if valid_count < 2:
        return np.nan, np.nan

    yv = np.empty(valid_count, dtype=np.float64)
    xv = np.empty(valid_count, dtype=np.float64)
    idx = 0
    for i in range(n):
        if np.isfinite(y[i]) and np.isfinite(x[i]):
            yv[idx] = y[i]
            xv[idx] = x[i]
            idx += 1

    np.random.seed(seed)
    orig_slope = _theilsen_1d(yv, xv)
    boot_slopes = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        yb = np.empty(valid_count, dtype=np.float64)
        xb = np.empty(valid_count, dtype=np.float64)
        for i in range(valid_count):
            j = np.random.randint(0, valid_count)
            yb[i] = yv[j]
            xb[i] = xv[j]
        boot_slopes[b] = _theilsen_1d(yb, xb)

    positive = 0
    negative = 0
    for b in range(n_boot):
        if boot_slopes[b] >= 0.0:
            positive += 1
        if boot_slopes[b] <= 0.0:
            negative += 1

    p_value = 2.0 * min(positive / n_boot, negative / n_boot)
    if p_value > 1.0:
        p_value = 1.0

    return orig_slope, p_value


def _bootstrap_theilsen_1d_wrapper(y, x, n_boot, seed):
    slope, p_value = _bootstrap_theilsen_1d(
        y.astype(np.float64),
        x.astype(np.float64),
        n_boot,
        seed,
    )
    return slope, p_value


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

def calculate_theilsen_trend(tmax, name_suffix=''):
    annual_mean = tmax.mean(dim='doy')
    annual_mean = annual_mean.chunk({'year': -1})
    trend, p_value = xr.apply_ufunc(
        _bootstrap_theilsen_1d_wrapper,
        annual_mean,
        annual_mean['year'].astype(np.float64),
        kwargs={'n_boot': BOOTSTRAP_REPS, 'seed': BOOTSTRAP_SEED},
        input_core_dims=[['year'], ['year']],
        output_core_dims=[[], []],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.float64, np.float64],
    )

    trend_name = 'Ta_TMAX_trend' if not name_suffix else f'Ta_TMAX_trend_{name_suffix}'
    p_value_name = f'{trend_name}_pvalue'
    trend.name = trend_name
    p_value.name = p_value_name
    return xr.Dataset({trend_name: trend, p_value_name: p_value})


def calculate_seasonal_trend(tmax):
    seas = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11]}
    dates = pd.date_range(start='2001-01-01', end='2001-12-31', freq='D')
    for season, months in seas.items():
        doys_in_season = dates[dates.month.isin(months)].dayofyear

        season_da = tmax.where(tmax['doy'].isin(doys_in_season), drop=True)
        season_ds = calculate_theilsen_trend(season_da, name_suffix=season)
        save_results(season_ds, os.path.join(PERCENTILES_DIR, f'zz_tmax_ts_trend_{season}.zarr'))

def save_results(da, out_path=OUT_ZARR):
    ds_out = da if isinstance(da, xr.Dataset) else da.to_dataset(name=da.name)
    print(f'Writing output to {out_path}...')
    with ProgressBar():
        ds_out.to_zarr(out_path, mode='w', consolidated=True)


# #%%
if __name__ == "__main__":
    with maybe_local_cluster():
        tmax = load_tmax_da()
        trend_ds = calculate_theilsen_trend(tmax)
        save_results(trend_ds, os.path.join(PERCENTILES_DIR, 'zz_tmax_ts_trend.zarr'))
        # calculate_seasonal_trend(tmax)
        print('Processing Complete.')


#%%
def plot_trend_map(trend_zarr):
    dpath = os.path.join(PERCENTILES_DIR, trend_zarr)
    ds = xr.open_zarr(dpath)

    if len(trend_zarr.split('_')) == 5:
        months = trend_zarr.split('_')[4].split('.')[0]
        trend_var = 'Ta_TMAX_trend_' + months
        p_value_var = trend_var + '_pvalue'
    else:
        months = ''
        trend_var = 'Ta_TMAX_trend'
        p_value_var = 'Ta_TMAX_trend_pvalue'

    trend = ds[trend_var]
    p_value = ds[p_value_var]
    alpha = xr.where(p_value >= 0.05, 0.5, 1.0)
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(projection=ccrs.PlateCarree())

    lim = max(abs(trend.quantile(.05).compute().item()), abs(trend.quantile(.95).compute().item()))
    trend.plot.imshow(ax=ax, cmap='RdBu_r', vmin=-.15, vmax=.15, alpha=alpha.values,
                    cbar_kwargs={'label': 'TMAX Trend (°C/year)',
                                'shrink': 0.5})
    ax.set_title('tmax trend 2003-2020 '  + months)
    fig.savefig(os.path.join(FIGDIR, 'trends', trend_zarr.replace('.zarr', '.png')), dpi=500,
                bbox_inches='tight')

if __name__ == "__main__":
    trend_zarrs = ['zz_tmax_ts_trend.zarr']#, 'zz_tmax_ts_trend_DJF.zarr', 'zz_tmax_ts_trend_MAM.zarr', 'zz_tmax_ts_trend_JJA.zarr', 'zz_tmax_ts_trend_SON.zarr']
    for tz in trend_zarrs:
        plot_trend_map(tz)

    #%% plot trend map and annual mean time series for small area

    westlimit=-62.72; southlimit=-16.49; eastlimit=-49.31; northlimit=-5.24
    AREA = dict(lat=slice(southlimit, northlimit), lon=slice(westlimit, eastlimit))

    trend = xr.open_zarr(os.path.join(PERCENTILES_DIR, 'zz_tmax_ts_trend.zarr'))['Ta_TMAX_trend']
    tmax = load_tmax_da()
    with ProgressBar():
        tmax = tmax.mean('doy').mean(['lat', 'lon']).compute(num_workers=32)
    ds = xr.Dataset({'tmax': tmax, 'trend': trend})
    ds_area = ds.sel(AREA)

    #%%
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ds_area['trend'].plot.imshow(ax=axes[0], cmap='coolwarm', vmin=-0.1, vmax=0.1, cbar_kwargs={'label': 'TMAX Trend (°C/year)'})
    axes[0].set_title('TMAX Trend 2003-2020 (small area)')
    ds_area['tmax'].plot(ax=axes[1])
    axes[1].set_title('Annual Mean TMAX (small area)')

