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
from glob import glob
from concurrent.futures import ThreadPoolExecutor

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
READER_WORKERS = 3
PREFETCH_DEPTH = 9

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
    print(f"Opening input Zarr: {DPATH}tmax_rechunked.zarr")
    # Use consolidated=False to support stores without .zmetadata.
    ds = xr.open_zarr(DPATH + 'tmax_rechunked.zarr', chunks={}, consolidated=False)
    # Align dimension names with downstream processing.
    da = ds["Ta_TMAX"].rename({"time": "year", "DOY": "doy"})
    # Attach explicit year coordinates expected downstream.
    da = da.assign_coords(year=np.arange(2006, 2018))
    if 'doy' not in da.coords:
        da = da.assign_coords(doy=np.arange(1, da.sizes['doy'] + 1))
    return da

def iter_chunk_bounds_1d(chunk_lengths):
    start = 0
    for n in chunk_lengths:
        end = start + n
        yield start, end
        start = end

def get_empty_chunk_map(chunked):

    NC_SOURCE_PATTERN = '/uolstore/Research/b/b0256/Users/earjba/ZhangandZhou/tropical_regions/*TMAX_daily_tropics30N30S.nc'
    files = sorted(glob(NC_SOURCE_PATTERN))
    print(f"Building empty-chunk map from: {files[0]}")

    orig = xr.open_dataset(files[0])
    orig = orig['Ta_TMAX'][0]


    empty_chunks = {}
    lat_axis = chunked.get_axis_num('lat')
    lon_axis = chunked.get_axis_num('lon')
    lat_chunks = chunked.chunks[lat_axis]
    lon_chunks = chunked.chunks[lon_axis]


    for (lat0, lat1) in iter_chunk_bounds_1d(lat_chunks):
        for (lon0, lon1) in iter_chunk_bounds_1d(lon_chunks):

            oc = orig.isel(lat=slice(lat0, lat1), lon=slice(lon0, lon1))
            
            valid = oc.isnull().all().item()
            empty_chunks[(lat0, lat1, lon0, lon1)] = valid

    empty_count = sum(empty_chunks.values())
    print(f"Chunk map ready: {len(empty_chunks)} total, {empty_count} empty")
    return empty_chunks

def iter_lat_lon_chunks(da, empty_chunks, lat_dim="lat", lon_dim="lon"):

    
    lat_axis = da.get_axis_num(lat_dim)
    lon_axis = da.get_axis_num(lon_dim)
    lat_chunks = da.chunks[lat_axis]
    lon_chunks = da.chunks[lon_axis]

    for i, (lat0, lat1) in enumerate(iter_chunk_bounds_1d(lat_chunks)):
        for j, (lon0, lon1) in enumerate(iter_chunk_bounds_1d(lon_chunks)):
            if not empty_chunks[(lat0, lat1, lon0, lon1)]:
                yield {
                    "lat": slice(lat0, lat1),
                    "lon": slice(lon0, lon1),
                }

def prepare_tile_payload(da, tile):
    cda = da.isel(lat=tile["lat"], lon=tile["lon"]).load()

    return {
        "vals": cda.values,
        "lat": cda.lat.values,
        "lon": cda.lon.values,
        "tile": tile,
    }

# --- MAIN EXECUTION ---
#%%

def process_tiles_serial(da, empty_chunks, max_tiles=None, reader_workers=1, prefetch_depth=1):
    print(f"Starting tile processing (reader_workers={reader_workers}, prefetch_depth={prefetch_depth})...")
    tile_count = 0
    non_empty_submitted = 0
    total_tiles = len(empty_chunks) - sum(empty_chunks.values())
    if max_tiles is not None:
        total_tiles = min(total_tiles, max_tiles)

    tiles_iter = iter(iter_lat_lon_chunks(da, empty_chunks))

    def submit_one(executor, pending):
        nonlocal non_empty_submitted
        if max_tiles is not None and non_empty_submitted >= max_tiles:
            return False
        try:
            tile = next(tiles_iter)
        except StopIteration:
            return False
        pending.append(executor.submit(prepare_tile_payload, da, tile))
        non_empty_submitted += 1
        return True

    progress = tqdm(total=total_tiles, desc="Tiles", unit="tile") if tqdm is not None else None
    try:
        with ThreadPoolExecutor(max_workers=reader_workers) as executor:
            pending = []
            target_depth = max(1, prefetch_depth)

            while len(pending) < target_depth and submit_one(executor, pending):
                pass

            while pending:
                payload = pending.pop(0).result()

                p95_vals = quantile95_over_year_window_skipna(payload["vals"], PAD)

                percentiles_ds = xr.DataArray(
                    p95_vals.astype(np.float32),
                    dims=('doy', 'lat', 'lon'),
                    coords={
                        'doy': np.arange(1, 366),
                        'lat': payload["lat"],
                        'lon': payload["lon"],
                    },
                    name='Ta_TMAX_p95',
                ).to_dataset()

                region_dict = {
                    "doy": slice(0, 365),
                    "lat": payload["tile"]["lat"],
                    "lon": payload["tile"]["lon"],
                }
                percentiles_ds.to_zarr(OUT_ZARR, region=region_dict)

                tile_count += 1
                if progress is not None:
                    progress.update(1)
                elif tile_count % 20 == 0:
                    print(f"Processed {tile_count} tiles")

                while len(pending) < target_depth and submit_one(executor, pending):
                    pass
    finally:
        if progress is not None:
            progress.close()

    if max_tiles is not None and non_empty_submitted >= max_tiles:
        print(f"Reached tile limit for tuning run: {max_tiles}")
    print(f"Finished serial tile processing ({tile_count} tiles)")
    print(f"Non-empty tiles submitted: {non_empty_submitted}")



if __name__ == "__main__":
    # Load data and initialize the output skeleton.
    da = load_tmax_da()

    # Skip all-empty chunks to avoid unnecessary compute.
    empty_chunks = get_empty_chunk_map(da)

    shape = (365, len(da.lat), len(da.lon))
    chunks = (365, 100, 100)  # Keep doy in one chunk for rolling-window efficiency.

    # Create a lazy skeleton using dask.
    dummy_data = da_lazy.full(shape, fill_value=np.nan, chunks=chunks)
    full_ds = xr.Dataset(
        data_vars={"Ta_TMAX_p95": (("doy", "lat", "lon"), dummy_data)},
        coords={"doy": np.arange(1, 366), "lat": da.lat, "lon": da.lon}
    )

    # Initialize Zarr on disk (metadata only).
    print(f"Initializing Zarr skeleton at {OUT_ZARR}...")
    full_ds.to_zarr(OUT_ZARR, compute=False, mode="w")

    process_tiles_serial(
        da,
        empty_chunks,
        max_tiles=MAX_TILES,
        reader_workers=READER_WORKERS,
        prefetch_depth=PREFETCH_DEPTH,
    )

    print("Consolidating metadata...")
    zarr.consolidate_metadata(OUT_ZARR)
    print("Processing Complete.")

