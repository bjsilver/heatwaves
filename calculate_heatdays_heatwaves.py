#%%
import os
from distributed import Client, LocalCluster
import xarray as xr
import dask.array as da
from constants import (
    STORAGE_DIR,
    HEATDAY_TEMP_THRESH,
    HEATWAVE_DAY_THRESH,
    LAT_CHUNK_SIZE,
    LON_CHUNK_SIZE,
    TMAX_RECHUNKED_FILE
)
from ufuncs import ensure_match_coords, local_cluster
from dask.diagnostics import ProgressBar
import numpy as np
import shutil
import matplotlib.pyplot as plt

HEATWAVE_DAY_THRESH = 3


def load_data():
    zz = xr.open_zarr(TMAX_RECHUNKED_FILE,
                    consolidated=True)['Ta_TMAX'] / 10
    
    # check if zz time has duplicates
    if len(zz.time) != len(np.unique(zz.time)):
        print("Warning: Duplicate time entries found in zz dataset. Removing duplicates.")
        zz = zz.drop_duplicates('time')

    zz_pct = xr.open_zarr(STORAGE_DIR + 'percentiles/zz_tmax_95pc.zarr',
                        consolidated=True)['Ta_TMAX_p95'] / 10

    zz_pct_gapfilled = xr.open_zarr(STORAGE_DIR + 'percentiles/zz_tmax_95pc_gapfilled.zarr',
                        consolidated=False)['Ta_TMAX_p95'] / 10



    era = xr.open_zarr(STORAGE_DIR + 'regridded/era5_t2m_p95_2003-2020_regridded.zarr',
                        consolidated=True)['heatwaves']
    

    pct = xr.Dataset({'zz_2003-2020': zz_pct,
                      'zz_2003-2020_gapfilled': zz_pct_gapfilled, 'era5_2003-2020': era})
    zz, pct = ensure_match_coords(zz, pct)
    return zz, pct




def remove_previous_zarr(path):
    if os.path.exists(path):
        print(f"Deleting existing store at {path}")
        shutil.rmtree(path)


def safe_zarr_chunks(da_in, year_chunk=1, lat_chunk=LAT_CHUNK_SIZE, lon_chunk=LON_CHUNK_SIZE):
    """Apply stable chunking; keep spatial chunks fixed for output efficiency."""
    chunk_spec = {}
    if 'year' in da_in.dims:
        chunk_spec['year'] = min(year_chunk, da_in.sizes['year'])
    if 'lat' in da_in.dims:
        chunk_spec['lat'] = min(lat_chunk, da_in.sizes['lat'])
    if 'lon' in da_in.dims:
        chunk_spec['lon'] = min(lon_chunk, da_in.sizes['lon'])
    return da_in.chunk(chunk_spec)

def calculate_annual_heatwave_sum(zz, pctda, write_path):

    if 'doy' not in pctda.dims:
        if 'time' in pctda.dims:
            pctda = pctda.rename({'time': 'doy'})
        else:
            raise ValueError(f"Expected percentile dims to include 'doy', got {pctda.dims}")

    pctda = pctda.chunk({'doy': -1, 'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})

    years = np.unique(zz.time.dt.year)

    for i, year in enumerate(years):
        zz_year = zz.sel(time=str(year))

        # noleap calendar: day-of-year must already be in 1..365.
        doy_index = zz_year.time.dt.dayofyear
        doy_min = int(doy_index.min().compute())
        doy_max = int(doy_index.max().compute())
        if doy_min < 1 or doy_max > 365:
            raise ValueError(
                f"Expected noleap day-of-year in 1..365, got range {doy_min}..{doy_max} for year {int(year)}"
            )
        pct_for_time = pctda.sel(doy=doy_index)

        heatdays_year = ((zz_year > HEATDAY_TEMP_THRESH) & (zz_year > pct_for_time))
        heatdays_year.name = 'heatdays'

        heatwaves_year = remove_short_runs(heatdays_year)
        annual_sum = heatwaves_year.sum('time').astype(np.uint16)
        annual_sum = annual_sum.expand_dims(year=[int(year)])
        annual_sum = safe_zarr_chunks(annual_sum)
        annual_sum.name = 'heatwave_days'

        if i == 0:
            annual_sum.to_zarr(write_path, consolidated=True, mode='w')
        else:
            annual_sum.to_zarr(write_path, consolidated=True, mode='a', append_dim='year')

def remove_short_runs(heatdays, thresh=HEATWAVE_DAY_THRESH):
    # 1) True only at run-ends of length >= thresh
    is_run_end = heatdays
    for k in range(1, thresh):
        is_run_end = is_run_end & heatdays.shift(time=k, fill_value=False)

    # 2) Expand each run-end back over the previous thresh-1 days
    mask = is_run_end
    for k in range(1, thresh):
        mask = mask | is_run_end.shift(time=-k, fill_value=False)

    # 3) Keep only days belonging to long-enough runs
    heatwaves = heatdays & mask
    heatwaves.name = 'heatwaves'
    return heatwaves

#%%
if __name__ == '__main__':

    with local_cluster(n_workers=1, threads_per_worker=128, memory_limit='500GB',
                       processes=False,
        dashboard_address=':6767', suppress_dask_warnings=True, use_dask_dashboard=True):

            zz, pct = load_data()

            for pctlkey in ['zz_2003-2020_gapfilled']:
                print(f"Calculating annual heatwave-day sums for {pctlkey}...")
                pctda = pct[pctlkey]
                write_path = STORAGE_DIR + f'heatdays/heatwaves_annual_sum_{pctlkey}.zarr'
                remove_previous_zarr(write_path)
                print(f"Writing annual sums to {write_path} as yearly appends...")
                calculate_annual_heatwave_sum(zz, pctda, write_path)
           
#%%

# heatwaves = xr.open_zarr(STORAGE_DIR + 'heatdays/heatwaves_era_2003-2020.zarr', consolidated=True)['heatwaves']