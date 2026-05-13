#%%
import os
from distributed import Client, LocalCluster
import xarray as xr
import dask.array as da
from constants import STORAGE_DIR
from ufuncs import ensure_match_coords
from dask.diagnostics import ProgressBar
import numpy as np
import shutil
import matplotlib.pyplot as plt

HEATWAVE_DAY_THRESH = 3


def load_data():
    zz = xr.open_zarr(STORAGE_DIR + 'rechunked/zhang_zhou_regridded_rechunked.zarr',
                    consolidated=True)['Ta_TMAX'] / 10
    
    # check if zz time has duplicates
    if len(zz.time) != len(np.unique(zz.time)):
        print("Warning: Duplicate time entries found in zz dataset. Removing duplicates.")
        zz = zz.drop_duplicates('time')

    zz_pct = xr.open_zarr(STORAGE_DIR + 'rechunked/zz_pct_rechunked.zarr',
                        consolidated=True)['Ta_TMAX_p95'] / 10
    zz_pct = zz_pct[:, :-1, :-1]

    era_pct_1970_1999 = xr.open_zarr(STORAGE_DIR + 'rechunked/era5_skt_p95_1970-1999_rechunked.zarr',
                        consolidated=True)['Ta_TMAX']

    era_pct_2003_2020 = xr.open_zarr(STORAGE_DIR + 'rechunked/era5_skt_p95_2003-2020_rechunked.zarr',
                        consolidated=True)['Ta_TMAX']

    pct = xr.Dataset({'era_1970-1999': era_pct_1970_1999, 'era_2003-2020': era_pct_2003_2020})
    
    zz, pct = ensure_match_coords(zz, pct)
    pct['zz_2003-2020'] = zz_pct
    pct.coords['time'] = list(range(1, 366))
    return zz, pct




def remove_previous_zarr(path):
    if os.path.exists(path):
        print(f"Deleting existing store at {path}")
        shutil.rmtree(path)

def calculate_heatdays(zz, pctda, write_path):

    years = np.unique(zz.time.dt.year)

    for i, year in enumerate(years):
        zz_year = zz.sel(time=str(year))
        zz_time_coords = zz_year.coords['time']
        zz_year_doy = zz_year.assign_coords(time=np.arange(1, zz_year.sizes['time'] + 1))

        heatdays_year = ((zz_year_doy > 28.5) & (zz_year_doy > pctda)).assign_coords(time=zz_time_coords)
        heatdays_year.name = 'heatdays'

        if i == 0:
            heatdays_year.to_zarr(write_path, consolidated=True, mode='w')
        else:
            heatdays_year.to_zarr(write_path, consolidated=True, mode='a', append_dim='time')

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

    with LocalCluster(n_workers=32, memory_limit='20GB',
                      threads_per_worker=2, dashboard_address=':8787') as cluster:
        with Client(cluster) as client:
            print(f"Dask Dashboard available at: {client.dashboard_link}")

            zz, pct = load_data()

            for pctlkey in pct.data_vars:
                print(f"Calculating heatdays and heatwaves for {pctlkey}...")
                pctda = pct[pctlkey]
                write_path = STORAGE_DIR + f'heatdays/heatdays_{pctlkey}.zarr'
                remove_previous_zarr(write_path)
                print(f"Writing heatdays to {write_path} as yearly appends...")
                calculate_heatdays(zz, pctda, write_path)

                heatdays = xr.open_zarr(write_path, consolidated=True)['heatdays']
                heatwaves = remove_short_runs(heatdays)
                write_path = STORAGE_DIR + f'heatdays/heatwaves_{pctlkey}.zarr'
                remove_previous_zarr(write_path)
                print(f"Writing heatwaves to {write_path}...")
                heatwaves.to_zarr(write_path, consolidated=True, mode='w')
           