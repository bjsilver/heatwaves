#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import os
from contextlib import contextmanager

import numpy as np
from numba import njit, prange
import xarray as xr
import dask.array as da_lazy
from pathlib import Path
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster


# --- CONFIGURATION ---
DPATH = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/rechunked/'
PERCENTILES_DIR = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/percentiles/'
OUT_ZARR = os.path.join(PERCENTILES_DIR, 'zz_tmax_95pc.zarr')
PAD = 7
PERCENTILE = 0.95

USE_LOCAL_CLUSTER = True
N_WORKERS = 32
THREADS_PER_WORKER = 4
MEMORY_LIMIT = '8GB'

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

    return da

def _percentile_block_with_stride_view(block, pad, q):
    # block shape: (year, doy_with_halo, lat, lon)
    window = 2 * pad + 1
    win_view = np.lib.stride_tricks.sliding_window_view(block, window_shape=window, axis=1)
    return quantile_over_window_view_skipna(win_view, q)


def compute_percentiles_dask(tmax_da):
    arr = tmax_da.data
    out = da_lazy.map_overlap(
        _percentile_block_with_stride_view,
        arr,
        depth={0: 0, 1: PAD, 2: 0, 3: 0},
        boundary={0: 'none', 1: 'periodic', 2: 'none', 3: 'none'},
        trim=False,
        dtype=np.float32,
        pad=PAD,
        q=PERCENTILE,
        drop_axis=0,
        chunks=(arr.chunks[1], arr.chunks[2], arr.chunks[3]),
    )

    return xr.DataArray(
        out,
        dims=('doy', 'lat', 'lon'),
        coords={
            'doy': tmax_da['doy'],
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
    print(f'Writing output to {out_path}...')
    with ProgressBar():
        ds_out.to_zarr(out_path, mode='w', consolidated=True)



if __name__ == "__main__":
    with maybe_local_cluster():
        tmax = load_tmax_da()
        p95 = compute_percentiles_dask(tmax)
        save_percentiles(p95, OUT_ZARR)
        print('Processing Complete.')

