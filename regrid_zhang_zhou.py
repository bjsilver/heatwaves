#%%
import json
import os
import shutil
import time

import dask
import dask.diagnostics
import numpy as np
import xarray as xr
import xesmf as xe
import zarr
from dask.distributed import Client, LocalCluster
from tqdm import tqdm


RECHUNKED_DIR = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/rechunked/'
SOURCE_STORE = RECHUNKED_DIR + 'tmax_annual_chunks.zarr'
OUTPUT_STORE = RECHUNKED_DIR + 'tmax_regridded.zarr'
WEIGHTS_PATH = RECHUNKED_DIR + 'tmax_regridder.nc'
PROGRESS_JSON = RECHUNKED_DIR + 'tmax_regridded_progress.json'

VARNAME = 'Ta_TMAX'
SOURCE_NODATA_THRESHOLD = -30000
SPATIAL_CHUNK = 256
TIME_CHUNK = 365
NUM_PROCESSES = 4
THREAD_WORKERS = 16
WRITE_TILE = 256
RESET_OUTPUT = False

TARGET_RES = .01


def mask_source_nodata(da):
    return da.astype('float32').where(da > SOURCE_NODATA_THRESHOLD)


def build_target_grid(ds):
    step = TARGET_RES  # 0.05
    inv = int(round(1.0 / step))  # 20

    lat_min = np.rint(float(ds[VARNAME].lat.min()) * inv) / inv
    lat_max = np.rint(float(ds[VARNAME].lat.max()) * inv) / inv
    lon_min = np.rint(float(ds[VARNAME].lon.min()) * inv) / inv
    lon_max = np.rint(float(ds[VARNAME].lon.max()) * inv) / inv

    n_lat = int(np.rint((lat_max - lat_min) * inv)) + 1
    n_lon = int(np.rint((lon_max - lon_min) * inv)) + 1

    target_lat = (lat_min + np.arange(n_lat) / inv).astype(np.float64)
    target_lon = (lon_min + np.arange(n_lon) / inv).astype(np.float64)

    # cosmetic cleanup for printing/inspection
    target_lat = np.round(target_lat, 2)
    target_lon = np.round(target_lon, 2)
    return target_lat, target_lon, xr.Dataset({'lat': (['lat'], target_lat), 'lon': (['lon'], target_lon)})


def load_or_create_progress(years):
    if not os.path.exists(PROGRESS_JSON):
        return {'completed_years': []}
    with open(PROGRESS_JSON, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    completed = sorted(set(y for y in data.get('completed_years', []) if y in years))
    return {'completed_years': completed}


def save_progress(progress):
    with open(PROGRESS_JSON, 'w', encoding='utf-8') as fh:
        json.dump(progress, fh, indent=2)


def create_skeleton(time_vals, target_lat, target_lon):
    print('Pre-creating Zarr skeleton...')
    t0 = time.time()

    if RESET_OUTPUT:
        if os.path.exists(OUTPUT_STORE):
            shutil.rmtree(OUTPUT_STORE)
        if os.path.exists(PROGRESS_JSON):
            os.remove(PROGRESS_JSON)

    if not os.path.exists(OUTPUT_STORE):
        root = zarr.open_group(OUTPUT_STORE, mode='w')
        z_data = root.require_dataset(
            VARNAME,
            shape=(len(time_vals), target_lat.size, target_lon.size),
            chunks=(TIME_CHUNK, SPATIAL_CHUNK, SPATIAL_CHUNK),
            dtype='float32',
            overwrite=True,
        )
        z_data.attrs['_ARRAY_DIMENSIONS'] = ['time', 'lat', 'lon']

        ref = time_vals[0]
        time_numeric = np.array([(t - ref).days for t in time_vals], dtype='float64')
        root.array('time', time_numeric, overwrite=True)
        root['time'].attrs['_ARRAY_DIMENSIONS'] = ['time']
        root['time'].attrs['units'] = f'days since {ref.year}-{ref.month:02d}-{ref.day:02d}'
        root['time'].attrs['calendar'] = 'noleap'

        root.array('lat', target_lat.astype('float64'), overwrite=True)
        root['lat'].attrs['_ARRAY_DIMENSIONS'] = ['lat']

        root.array('lon', target_lon.astype('float64'), overwrite=True)
        root['lon'].attrs['_ARRAY_DIMENSIONS'] = ['lon']

    print(f'  Skeleton ready in {time.time() - t0:.1f}s')


def main():
    print('Opening source Zarr (lazy)...')
    # Setup the hybrid pool: 4 processes, 16 threads each = 64 cores total
    with LocalCluster(n_workers=NUM_PROCESSES, threads_per_worker=THREAD_WORKERS, memory_limit='40GB') as cluster:

        with Client(cluster) as client:

            ds = xr.open_zarr(SOURCE_STORE, consolidated=True)

            time_vals_all = ds.time.values
            years_arr = np.array([t.year for t in time_vals_all], dtype=np.int32)
            years = sorted(set(int(y) for y in years_arr))

            print('Building target grid...')
            target_lat, target_lon, target_grid = build_target_grid(ds)
            print(f'  Output shape: ({len(time_vals_all)}, {target_lat.size}, {target_lon.size})')

            print(f'Weights path: {WEIGHTS_PATH}')
            if os.path.exists(WEIGHTS_PATH):
                print('  Reusing existing weights...')
                regridder = xe.Regridder(
                    ds[VARNAME],
                    target_grid,
                    'bilinear',
                    reuse_weights=True,
                    weights=WEIGHTS_PATH,
                )
            else:
                print('  Building new weights...')
                regridder = xe.Regridder(ds[VARNAME], target_grid, 'bilinear')
                regridder.to_netcdf(WEIGHTS_PATH)
                print(f'  Saved weights to {WEIGHTS_PATH}')

            create_skeleton(time_vals_all, target_lat, target_lon)

            year_slices = {}
            for year in years:
                idx = np.where(years_arr == year)[0]
                year_slices[year] = (int(idx[0]), int(idx[-1]) + 1)

            progress = load_or_create_progress(years)
            completed_years = set(progress['completed_years'])

            root = zarr.open_group(OUTPUT_STORE, mode='r+')
            z_data = root[VARNAME]

            total_start = time.time()
            for i, year in enumerate(years, start=1):
                start_idx, end_idx = year_slices[year]
                n_days = end_idx - start_idx

                if year in completed_years:
                    print(f'  [{i}/{len(years)}] {year}: already completed, skipping')
                    continue

                year_start = time.time()
                year_da = ds[VARNAME].isel(time=slice(start_idx, end_idx)).chunk(
                    {'time': -1, 'lat': SPATIAL_CHUNK, 'lon': SPATIAL_CHUNK}
                )
                year_da = mask_source_nodata(year_da)
                regridded = regridder(
                    year_da,
                    skipna=True,
                    na_thres=1.0,
                    output_chunks={'lat': SPATIAL_CHUNK, 'lon': SPATIAL_CHUNK},
                ).astype('float32')

                task_count = len(regridded.data.dask)
                print(
                    f'  [{i}/{len(years)}] {year}: {task_count} tasks, '
                    f'time[{start_idx}:{end_idx}]...',
                    flush=True,
                )

                # Compute and write spatial tiles to keep memory bounded.
                n_lat = target_lat.size
                n_lon = target_lon.size
                tile_total = ((n_lat + WRITE_TILE - 1) // WRITE_TILE) * ((n_lon + WRITE_TILE - 1) // WRITE_TILE)
                tile_done = 0

                for lat0 in tqdm(range(0, n_lat, WRITE_TILE)):
                    lat1 = min(lat0 + WRITE_TILE, n_lat)
                    for lon0 in range(0, n_lon, WRITE_TILE):
                        lon1 = min(lon0 + WRITE_TILE, n_lon)

                        tile_da = regridded.isel(lat=slice(lat0, lat1), lon=slice(lon0, lon1))
                        # with dask.config.set(scheduler='threads', num_workers=THREAD_WORKERS):
                        #     with dask.diagnostics.ProgressBar():
                        #         tile_np = tile_da.data.compute()

                        tile_np = tile_da.data.compute(scheduler='distributed')

                        z_data[start_idx:end_idx, lat0:lat1, lon0:lon1] = tile_np

                        tile_done += 1
                        if tile_done % 20 == 0 or tile_done == tile_total:
                            print(
                                f'    tiles {tile_done}/{tile_total} '
                                f'({100.0 * tile_done / tile_total:.1f}%)',
                                flush=True,
                            )

                completed_years.add(year)
                progress['completed_years'] = sorted(completed_years)
                save_progress(progress)

                print(f'    done in {time.time() - year_start:.1f}s', flush=True)

            print(f'All years written in {time.time() - total_start:.1f}s')
            print('Consolidating metadata...')
            zarr.consolidate_metadata(OUTPUT_STORE)
            print(f'Done. Output: {OUTPUT_STORE}')


#%%
if __name__ == '__main__':
    main()

#%%