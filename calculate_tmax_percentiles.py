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
from dask.distributed import Client, LocalCluster

# --- CONFIGURATION ---
DPATH = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/rechunked/'
PERCENTILES_DIR = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/percentiles/'
OUT_ZARR = os.path.join(PERCENTILES_DIR, 'tmax_95pc.zarr')
PAD = 7

# Ensure output directory exists
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
def quantile95_over_year_window_skipna(arr):
    # arr shape: (year, doy, lat, lon, window)
    ny, nd, nlat, nlon, nw = arr.shape
    out = np.empty((nd, nlat, nlon), dtype=np.float64)
    for d in prange(nd):
        for i in range(nlat):
            for j in range(nlon):
                vals = np.empty(ny * nw, dtype=np.float64)
                k = 0
                for y in range(ny):
                    for w in range(nw):
                        vals[k] = arr[y, d, i, j, w]
                        k += 1
                out[d, i, j] = _quantile_skipna_1d(vals, 0.95)
    return out

# --- HELPER FUNCTIONS ---

def load_tmax_da():
    # consolidated=False prevents the KeyError: '.zmetadata'
    ds = xr.open_zarr(DPATH + 'tmax_rechunked.zarr', chunks={}, consolidated=False)
    # Rename 'time' to 'year' to match your dimension logic
    da = ds["Ta_TMAX"].rename({"time": "year", "DOY": "doy"})
    # Assign year coordinates (must match dimension name exactly)
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

def iter_lat_lon_chunks(da, lat_dim="lat", lon_dim="lon"):
    lat_axis = da.get_axis_num(lat_dim)
    lon_axis = da.get_axis_num(lon_dim)
    lat_chunks = da.chunks[lat_axis]
    lon_chunks = da.chunks[lon_axis]

    for i, (lat0, lat1) in enumerate(iter_chunk_bounds_1d(lat_chunks)):
        for j, (lon0, lon1) in enumerate(iter_chunk_bounds_1d(lon_chunks)):
            yield {
                "lat": slice(lat0, lat1),
                "lon": slice(lon0, lon1),
            }

def build_wrapped_doy_axis(da, pad):
    left = da.isel(doy=slice(-pad, None)).assign_coords(doy=np.arange(1 - pad, 1))
    middle = da
    right = da.isel(doy=slice(0, pad)).assign_coords(doy=np.arange(366, 366 + pad))
    return xr.concat([left, middle, right], dim='doy')

# --- MAIN EXECUTION ---
#%%
# 1. Load data and Initialize Skeleton
da = load_tmax_da()

shape = (365, len(da.lat), len(da.lon))
chunks = (365, 100, 100) # Ensure doy is in one chunk for rolling window efficiency

# Create a lazy skeleton using dask (zero memory footprint)
dummy_data = da_lazy.full(shape, fill_value=np.nan, chunks=chunks)
full_ds = xr.Dataset(
    data_vars={"Ta_TMAX_p95": (("doy", "lat", "lon"), dummy_data)},
    coords={"doy": np.arange(1, 366), "lat": da.lat, "lon": da.lon}
)

# Initialize Zarr on disk (metadata only)
print(f"Initializing Zarr skeleton at {OUT_ZARR}...")
full_ds.to_zarr(OUT_ZARR, compute=False, mode="w")

# 2. Iterative Tile Processing
print("Starting iterative processing...")
for tile in iter_lat_lon_chunks(da):
    # Load tile into RAM
    cda = da.isel(lat=tile["lat"], lon=tile["lon"]).load()

    # Skip tiles that are all NaN (oceans/empty space)
    if np.isnan(cda).all():
        continue


    # Prep data for Numba
    padded = build_wrapped_doy_axis(cda, PAD)
    windowed = padded.rolling(doy=2 * PAD + 1, center=True).construct('window').sel(doy=slice(1, 365))

    # Calculate Percentiles
    p95_vals = quantile95_over_year_window_skipna(windowed.values)
    
    # Wrap in Dataset with coordinates aligned to the skeleton
    percentiles_ds = xr.DataArray(
        p95_vals.astype(np.float32), # Optional: float32 saves 50% disk space
        dims=('doy', 'lat', 'lon'),
        coords={'doy': np.arange(1, 366), 'lat': cda.lat, 'lon': cda.lon},
        name='Ta_TMAX_p95',
    ).to_dataset()

    # Map the tile slice to the global Zarr indices
    region_dict = {
        "doy": slice(0, 365),
        "lat": tile["lat"],
        "lon": tile["lon"]
    }

    # Write the specific region
    percentiles_ds.to_zarr(OUT_ZARR, region=region_dict)
    print(f"Finished: lat={tile['lat']} lon={tile['lon']}")

# 3. Final Step: Consolidate metadata for faster future reading
print("Consolidating metadata...")
zarr.consolidate_metadata(OUT_ZARR)
print("Processing Complete.")