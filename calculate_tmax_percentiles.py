#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import os
from contextlib import contextmanager
import numpy as np
from numba import njit, prange
import xarray as xr
import dask
import dask.array as da_lazy
from pathlib import Path
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
from constants import TMAX_RECHUNKED_FILE
from ufuncs import local_cluster

# --- CONFIGURATION ---
PERCENTILES_DIR = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/percentiles/'
OUT_ZARR = os.path.join(PERCENTILES_DIR, 'zz_tmax_95pc.zarr')
PAD = 7
PERCENTILE = 0.95

USE_LOCAL_CLUSTER = True
N_WORKERS = 32
THREADS_PER_WORKER = 4
MEMORY_LIMIT = '22GB'
TILE_SIZE = 125  # spatial tile size for lat/lon

# Ensure output directory exists.
Path(PERCENTILES_DIR).mkdir(parents=True, exist_ok=True)

# --- NUMBA COMPUTATIONAL KERNELS ---

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
def quantile_over_window_view_skipna(win_view, q):
    # win_view shape: (year, doy, lat, lon, window)
    ny, nd, nlat, nlon, nw = win_view.shape
    out = np.empty((nd, nlat, nlon), dtype=np.float32)
    for d in prange(nd):
        for i in range(nlat):
            for j in range(nlon):
                vals = np.empty(ny * nw, dtype=np.float64)
                k = 0
                for y in range(ny):
                    for w in range(nw):
                        vals[k] = win_view[y, d, i, j, w]
                        k += 1
                out[d, i, j] = np.float32(_quantile_skipna_1d(vals, q))
    return out

# --- HELPER FUNCTIONS ---

def load_tmax_da():
    print(f"Opening input Zarr: {TMAX_RECHUNKED_FILE}")
    # Use consolidated=False to support stores without .zmetadata.
    ds = xr.open_zarr(TMAX_RECHUNKED_FILE, chunks={}, consolidated=False)
    da = ds['Ta_TMAX']
    # check if zz time has duplicates
    if len(da.time) != len(np.unique(da.time)):
        print("Warning: Duplicate time entries found in zz dataset. Removing duplicates.")
        da = da.drop_duplicates('time')
    # replace -9999 with NaN
    da = da.where(da != -9999, np.nan)
    da.attrs['units'] = 'tenths of degC'

    return da

def _percentile_block_flat_time(block, n_years, pad, q):
    # block shape: (n_years * 365, lat, lon) — no leap days guaranteed
    n_days = 365
    _, nlat, nlon = block.shape

    # Zero-copy view as (n_years, 365, lat, lon) for C-contiguous arrays
    reshaped = block.reshape(n_years, n_days, nlat, nlon)

    # Periodic padding on the doy axis (within each year)
    padded = np.concatenate(
        [reshaped[:, -pad:, :, :], reshaped, reshaped[:, :pad, :, :]],
        axis=1,
    )
    # padded shape: (n_years, 365 + 2*pad, lat, lon)

    window = 2 * pad + 1
    win_view = np.lib.stride_tricks.sliding_window_view(
        padded, window_shape=window, axis=1
    )
    # win_view shape: (n_years, 365, lat, lon, window)

    return quantile_over_window_view_skipna(win_view, q)


def compute_percentiles_dask(tmax_da):
    n_years = tmax_da.sizes['time'] // 365

    # Rechunk with method='tasks' to avoid the P2P shuffle algorithm which
    # spills to /tmp.  The tasks-based rechunk is pure in-memory graph fusion.
    arr = tmax_da.data.rechunk(
        {0: n_years * 365, 1: TILE_SIZE, 2: TILE_SIZE},
        method='tasks',
    )

    out = da_lazy.map_blocks(
        _percentile_block_flat_time,
        arr,
        dtype=np.float32,
        n_years=n_years,
        pad=PAD,
        q=PERCENTILE,
        # drop_axis=0,
        # new_axis=0,
        chunks=((365,), arr.chunks[1], arr.chunks[2]),
    )

    return xr.DataArray(
        out,
        dims=('doy', 'lat', 'lon'),
        coords={
            'doy': np.arange(1, 366, dtype=np.int16),
            'lat': tmax_da['lat'],
            'lon': tmax_da['lon'],
        },
        name='Ta_TMAX_p95',
    )


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


def save_percentiles(p95_da, out_path=OUT_ZARR):
    ds_out = p95_da.to_dataset(name='Ta_TMAX_p95')
    # rechunk back to 250
    ds_out = ds_out.chunk({'doy': -1, 'lat': 250, 'lon': 250})
    print(f'Writing output to {out_path}...')
    with ProgressBar():
        ds_out.to_zarr(out_path, mode='w', consolidated=True,
                       zarr_format=2)


#%%
if __name__ == "__main__":
    with local_cluster(processes=False, n_workers=1, memory_limit='600GB',
                       threads_per_worker=10) as cluster:
        print('Starting processing...')
        tmax = load_tmax_da()
        p95 = compute_percentiles_dask(tmax)
        save_percentiles(p95, OUT_ZARR)
        print('Processing Complete.')


#%%

# with ProgressBar():
#     tmax = load_tmax_da()
#     tmax.isnull().sum(['lat', 'lon']).compute(num_workers=32).plot()

# #%%

# with ProgressBar():
#     all_nan = tmax.isnull().all('time').sum().compute(num_workers=32)
#     any_nan = tmax.isnull().any('time').sum().compute(num_workers=32)