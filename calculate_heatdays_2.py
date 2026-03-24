#!/usr/bin/env python3
#%%
"""
Unified Dask pipeline:
1. Compute per-chunk 95th percentile climatology (DOY)
2. Compute heatdays (TMAX > threshold and TMAX > percentile)
3. Remove heatdays that are part of runs shorter than min-run length
"""

import argparse
import os
import re
import time
from glob import glob

import numpy as np
import regionmask
import xarray as xr
from dask.distributed import Client, LocalCluster

from constants import LAT_STEP, LON_STEP

DPATH = '/uolstore/Research/b/b0256/Users/earjba/ZhangandZhou/tropical_regions/'
PERCENTILE_DIR = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/percentile_chunks/'
HEATDAYS_DIR = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/heatdays_chunks/'
FINAL_DIR = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/heatwave_chunks/'

PERCENTILE_VAR = 'Ta_TMAX_95pc'
TMAX_VAR = 'Ta_TMAX'
HEATDAYS_VAR = 'heatdays'
FINAL_VAR = 'heatwaves'


def parse_args():
    parser = argparse.ArgumentParser(description='Unified Dask heatwave pipeline')
    parser.add_argument('--input-glob', default=DPATH + 'SouthAmerica_Ta_SVCMsp_*TMAX_daily_tropics30N30S.nc')
    parser.add_argument('--start-year', type=int, default=2006)
    parser.add_argument('--end-year', type=int, default=2017)
    parser.add_argument('--lat-min', type=int, default=-20)
    parser.add_argument('--lat-max', type=int, default=0)
    parser.add_argument('--lon-min', type=int, default=-75)
    parser.add_argument('--lon-max', type=int, default=-45)
    parser.add_argument('--lat-step', type=int, default=LAT_STEP)
    parser.add_argument('--lon-step', type=int, default=LON_STEP)
    parser.add_argument('--pad-days', type=int, default=7)
    parser.add_argument('--heatday-threshold', type=float, default=28.5)
    parser.add_argument('--min-run-days', type=int, default=3)
    parser.add_argument('--skip-land-check', action='store_true')
    parser.add_argument('--overwrite', action='store_true')

    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--threads-per-worker', type=int, default=1)
    parser.add_argument('--memory-limit', default='6GB')
    parser.add_argument('--scheduler-address', default='')

    parser.add_argument('--percentile-dir', default=PERCENTILE_DIR)
    parser.add_argument('--heatdays-dir', default=HEATDAYS_DIR)
    parser.add_argument('--final-dir', default=FINAL_DIR)
    return parser.parse_args()


def get_year_from_path(path):
    match = re.search(r'_(\d{4})TMAX_', os.path.basename(path))
    if not match:
        return None
    return int(match.group(1))


def get_input_paths(input_glob, start_year, end_year):
    all_paths = sorted(glob(input_glob))
    selected = []
    for path in all_paths:
        year = get_year_from_path(path)
        if year is None:
            continue
        if start_year <= year <= end_year:
            selected.append(path)
    return selected


def get_amazon_chunks(lat_min, lat_max, lon_min, lon_max, lat_step, lon_step):
    chunks = []
    for lat in range(lat_min, lat_max, lat_step):
        for lon in range(lon_min, lon_max, lon_step):
            lat_slice = slice(lat + lat_step, lat)
            lon_slice = slice(lon, lon + lon_step)
            chunks.append((lat_slice, lon_slice))
    return chunks


def chunk_name(lat_slice, lon_slice):
    return f'{lat_slice.start}_{lon_slice.start}'


def check_if_land(lat_slice, lon_slice):
    lon = np.arange(-179.5, 180, 0.1)
    lat = np.arange(-89.5, 90, 0.1)

    min_lat, max_lat = lat_slice.stop, lat_slice.start
    min_lon, max_lon = lon_slice.start, lon_slice.stop

    land_mask = regionmask.defined_regions.natural_earth_v4_1_0.land_110.mask(lon, lat)
    local = land_mask.loc[{'lat': slice(min_lat, max_lat), 'lon': slice(min_lon, max_lon)}]
    return local.count() > 0


def preprocess_chunk(lat_slice, lon_slice):
    def pp(ds):
        ds = ds.loc[{'lat': lat_slice, 'lon': lon_slice}]
        if 'spatial_ref' in ds:
            ds = ds.drop_vars('spatial_ref')
        ds = ds.sortby('lat', ascending=False)

        if 'DOY' in ds.dims:
            ds = ds.assign_coords({'DOY': np.arange(1, ds.sizes['DOY'] + 1)})
        elif 'time' in ds.dims:
            ds = ds.rename({'time': 'DOY'})
            ds = ds.assign_coords({'DOY': np.arange(1, ds.sizes['DOY'] + 1)})

        return ds

    return pp


def open_chunked_tmax(input_paths, lat_slice, lon_slice):
    ds = xr.open_mfdataset(
        input_paths,
        preprocess=preprocess_chunk(lat_slice, lon_slice),
        concat_dim='year',
        combine='nested',
        parallel=True,
        chunks={'year': 1, 'DOY': -1, 'lat': 'auto', 'lon': 'auto'},
    )
    return ds[TMAX_VAR]


def compute_percentile_chunk(tmax, pad_days):
    pad_start = tmax.sel(DOY=slice(365 - pad_days + 1, 365)).assign_coords(DOY=np.arange(1 - pad_days, 1))
    pad_end = tmax.sel(DOY=slice(1, pad_days)).assign_coords(DOY=np.arange(366, 366 + pad_days))
    padded = xr.concat([pad_start, tmax, pad_end], dim='DOY')

    rolling = padded.rolling(DOY=2 * pad_days + 1, center=True, min_periods=2 * pad_days + 1).construct('window')
    rolling = rolling.sel(DOY=slice(1, 365))

    percentile = rolling.quantile(0.95, dim=('year', 'window'), skipna=True)
    if 'quantile' in percentile.dims:
        percentile = percentile.squeeze('quantile', drop=True)

    percentile = percentile.chunk({'DOY': -1, 'lat': min(200, percentile.sizes['lat']), 'lon': min(200, percentile.sizes['lon'])})
    percentile.name = PERCENTILE_VAR
    percentile.attrs['units'] = '0.1degC'
    return percentile


def build_heatdays_for_years(input_paths, percentile, years, threshold):
    yearly = []

    for year in years:
        year_path = None
        for path in input_paths:
            if get_year_from_path(path) == year:
                year_path = path
                break

        if year_path is None:
            raise ValueError(f'No TMAX file found for year {year}')

        ds = xr.open_dataset(year_path, chunks={'DOY': -1, 'lat': 'auto', 'lon': 'auto'})
        tmax = ds[TMAX_VAR].sortby('lat', ascending=False)

        tmax = tmax.loc[
            {
                'lat': slice(float(percentile.lat.max()), float(percentile.lat.min())),
                'lon': slice(float(percentile.lon.min()), float(percentile.lon.max())),
            }
        ]

        if 'time' in tmax.dims:
            tmax = tmax.rename({'time': 'DOY'})

        tmax = tmax.chunk({'DOY': -1, 'lat': min(200, percentile.sizes['lat']), 'lon': min(200, percentile.sizes['lon'])})
        pct = percentile.chunk({'DOY': -1, 'lat': min(200, percentile.sizes['lat']), 'lon': min(200, percentile.sizes['lon'])})
        tmax, pct = xr.align(tmax, pct, join='exact')

        heatdays = ((tmax > threshold) & (tmax > pct)).astype('int8')
        heatdays = heatdays.rename({'DOY': 'time'})
        heatdays = heatdays.assign_coords(
            {
                'time': xr.date_range(
                    start=f'{year}-01-01',
                    end=f'{year}-12-31',
                    freq='D',
                    use_cftime=True,
                    calendar='noleap',
                )
            }
        )
        yearly.append(heatdays)

    out = xr.concat(yearly, dim='time')
    out.name = HEATDAYS_VAR
    return out


def remove_short_runs(heatdays, min_run_days):
    heatdays_bool = heatdays.astype(bool)
    is_run_end = heatdays_bool.rolling(time=min_run_days).all()
    mask = is_run_end.rolling(time=min_run_days, center=False).any()
    mask = mask.shift(time=-(min_run_days - 1))
    filtered = heatdays_bool.where(mask.fillna(False), False).astype('int8')
    filtered.name = FINAL_VAR
    return filtered


def build_output_paths(args, lat_slice, lon_slice):
    c_name = chunk_name(lat_slice, lon_slice)

    percentile_path = os.path.join(args.percentile_dir, f'percentiles_{c_name}.nc')
    heatdays_path = os.path.join(args.heatdays_dir, f'heatdays_{c_name}.nc')
    final_path = os.path.join(args.final_dir, f'heatwaves_{c_name}.nc')

    return percentile_path, heatdays_path, final_path


def ensure_dirs(args):
    os.makedirs(args.percentile_dir, exist_ok=True)
    os.makedirs(args.heatdays_dir, exist_ok=True)
    os.makedirs(args.final_dir, exist_ok=True)


def should_skip(path, overwrite):
    return os.path.exists(path) and not overwrite


def process_chunk(args, client, input_paths, years, lat_slice, lon_slice):
    start = time.time()

    percentile_path, heatdays_path, final_path = build_output_paths(args, lat_slice, lon_slice)
    print(f'processing chunk lat={lat_slice.start}:{lat_slice.stop} lon={lon_slice.start}:{lon_slice.stop}')

    if should_skip(final_path, args.overwrite):
        print(f'skipping existing final output: {final_path}')
        return

    tmax_chunk = open_chunked_tmax(input_paths, lat_slice, lon_slice)
    percentile = compute_percentile_chunk(tmax_chunk, args.pad_days)
    percentile_write = percentile.to_netcdf(
        percentile_path,
        encoding={PERCENTILE_VAR: {'zlib': True, 'complevel': 5}},
        compute=False,
    )
    client.compute(percentile_write).result()

    if should_skip(heatdays_path, args.overwrite):
        heatdays = xr.open_dataarray(heatdays_path, chunks={'time': -1, 'lat': 'auto', 'lon': 'auto'})
        if heatdays.name is None:
            heatdays.name = HEATDAYS_VAR
    else:
        heatdays = build_heatdays_for_years(input_paths, percentile, years, args.heatday_threshold)
        heatdays_write = heatdays.to_netcdf(
            heatdays_path,
            encoding={HEATDAYS_VAR: {'zlib': True, 'complevel': 5, 'dtype': 'int8'}},
            compute=False,
        )
        client.compute(heatdays_write).result()

    filtered = remove_short_runs(heatdays, args.min_run_days)
    final_write = filtered.to_netcdf(
        final_path,
        encoding={FINAL_VAR: {'zlib': True, 'complevel': 5, 'dtype': 'int8'}},
        compute=False,
    )
    client.compute(final_write).result()

    elapsed = time.time() - start
    print(f'finished chunk in {elapsed:.1f}s -> {final_path}')


def main():
    args = parse_args()
    ensure_dirs(args)

    input_paths = get_input_paths(args.input_glob, args.start_year, args.end_year)
    if not input_paths:
        raise ValueError('No matching TMAX files found for the requested year range')

    years = list(range(args.start_year, args.end_year + 1))
    chunks = get_amazon_chunks(
        args.lat_min,
        args.lat_max,
        args.lon_min,
        args.lon_max,
        args.lat_step,
        args.lon_step,
    )

    if args.scheduler_address:
        client = Client(args.scheduler_address)
    else:
        cluster = LocalCluster(
            n_workers=args.workers,
            threads_per_worker=args.threads_per_worker,
            memory_limit=args.memory_limit,
        )
        client = Client(cluster)

    print('Dask dashboard:', client.dashboard_link)

    try:
        for lat_slice, lon_slice in chunks:
            if not args.skip_land_check and not check_if_land(lat_slice, lon_slice):
                print(f'skipping ocean chunk lat={lat_slice.start}:{lat_slice.stop} lon={lon_slice.start}:{lon_slice.stop}')
                continue

            process_chunk(args, client, input_paths, years, lat_slice, lon_slice)
    finally:
        client.close()


if __name__ == '__main__':
    main()
