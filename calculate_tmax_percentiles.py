#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import os
import numpy as np
from numba import njit, prange
import xarray as xr
import dask.array as da_lazy
from pathlib import Path
import zarr

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

# --- CONFIGURATION ---
DPATH = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/rechunked/'
PERCENTILES_DIR = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/percentiles/'
OUT_ZARR = os.path.join(PERCENTILES_DIR, 'tmax_95pc.zarr')
PAD = 7
MAX_TILES = None  # Set to None to process all tiles.

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
def quantile95_over_year_window_skipna(arr, pad):
    # arr shape: (year, doy, lat, lon)
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
                out[d, i, j] = _quantile_skipna_1d(vals, 0.95)
    return out

# --- HELPER FUNCTIONS ---

def load_tmax_da():
    print(f"Opening input Zarr: {DPATH}tmax_regridded.zarr")
    # Use consolidated=False to support stores without .zmetadata.
    ds = xr.open_zarr(DPATH + 'tmax_regridded.zarr', chunks={}, consolidated=False)
    da = ds['Ta_TMAX']

    da.coords['lat'] = np.where(np.isnan(da.lat.values), 0, da.lat.values)

    if 'time' in da.dims:
        year_vals = da['time'].dt.year.data
        doy_vals = da['time'].dt.dayofyear.data
        da = da.assign_coords(year=('time', year_vals), doy=('time', doy_vals))
        da = da.set_index(time=['year', 'doy']).unstack('time').transpose('year', 'doy', 'lat', 'lon')

    if 'doy' not in da.coords:
        da = da.assign_coords(doy=np.arange(1, da.sizes['doy'] + 1))
    return da

def iter_chunk_bounds_1d(chunk_lengths):
    start = 0
    for n in chunk_lengths:
        end = start + n
        yield start, end
        start = end

def iter_lat_lon_chunks(da, lat_dim="lat", lon_dim="lon"):
    lat_axis = da.get_axis_num(lat_dim)
    lon_axis = da.get_axis_num(lon_dim)
    lat_chunks = da.chunks[lat_axis]
    lon_chunks = da.chunks[lon_axis]

    for (lat0, lat1) in iter_chunk_bounds_1d(lat_chunks):
        for (lon0, lon1) in iter_chunk_bounds_1d(lon_chunks):
            yield {
                "lat": slice(lat0, lat1),
                "lon": slice(lon0, lon1),
            }

# --- MAIN EXECUTION ---
#%%

def process_tiles_serial(da, max_tiles=None):
    print("Starting tile processing...")
    tile_count = 0

    lat_axis = da.get_axis_num('lat')
    lon_axis = da.get_axis_num('lon')
    total_tiles = len(da.chunks[lat_axis]) * len(da.chunks[lon_axis])
    if max_tiles is not None:
        total_tiles = min(total_tiles, max_tiles)

    progress = tqdm(total=total_tiles, desc="Tiles", unit="tile") if tqdm is not None else None
    try:
        for tile in iter_lat_lon_chunks(da):
            if max_tiles is not None and tile_count >= max_tiles:
                break

            cda = da.isel(lat=tile["lat"], lon=tile["lon"]).load()
            p95_vals = quantile95_over_year_window_skipna(cda.values, PAD)

            percentiles_ds = xr.DataArray(
                p95_vals.astype(np.float32),
                dims=('doy', 'lat', 'lon'),
                coords={
                    'doy': cda.doy.values,
                    'lat': cda.lat.values,
                    'lon': cda.lon.values,
                },
                name='Ta_TMAX_p95',
            ).to_dataset()

            region_dict = {
                "doy": slice(0, da.sizes['doy']),
                "lat": tile["lat"],
                "lon": tile["lon"],
            }
            percentiles_ds.to_zarr(OUT_ZARR, region=region_dict)

            tile_count += 1
            if progress is not None:
                progress.update(1)
            elif tile_count % 20 == 0:
                print(f"Processed {tile_count} tiles")
    finally:
        if progress is not None:
            progress.close()

    if max_tiles is not None and tile_count >= max_tiles:
        print(f"Reached tile limit for tuning run: {max_tiles}")
    print(f"Finished serial tile processing ({tile_count} tiles)")



if __name__ == "__main__":
    # Load data and initialize the output skeleton.
    da = load_tmax_da()

    shape = (da.sizes['doy'], len(da.lat), len(da.lon))
    doy_axis = da.get_axis_num('doy')
    lat_axis = da.get_axis_num('lat')
    lon_axis = da.get_axis_num('lon')
    chunks = (da.chunks[doy_axis], da.chunks[lat_axis], da.chunks[lon_axis])

    # Create a lazy skeleton using dask.
    dummy_data = da_lazy.full(shape, fill_value=np.nan, chunks=chunks)
    full_ds = xr.Dataset(
        data_vars={"Ta_TMAX_p95": (("doy", "lat", "lon"), dummy_data)},
        coords={"doy": da.doy.values, "lat": da.lat, "lon": da.lon}
    )

    # Initialize Zarr on disk (metadata only).
    print(f"Initializing Zarr skeleton at {OUT_ZARR}...")
    full_ds.to_zarr(OUT_ZARR, compute=False, mode="w")

    process_tiles_serial(
        da,
        max_tiles=MAX_TILES,
    )

    print("Consolidating metadata...")
    zarr.consolidate_metadata(OUT_ZARR)
    print("Processing Complete.")

