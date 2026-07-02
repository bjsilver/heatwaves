#%%
import os
from glob import glob
import json
import numpy as np
import pandas as pd
from sklearn import cluster
import xarray as xr
import xesmf as xe
from dask.distributed import Client, LocalCluster
from numcodecs import Blosc
from dask.diagnostics import ProgressBar
import dask

from constants import STORAGE_DIR



# ---------------------------
# CONFIG
# ---------------------------
SOURCE_DIR = "/uolstore/Research/b/b0256/Users/earjba/ZhangandZhou/tropical_regions/"
ZARR_DESTINATION = "/uolstore/Research/b/b0256/Users/earbsi/heatwaves/regridded/zhang_zhou_regridded.zarr"
WEIGHTS_FILE = "/uolstore/Research/b/b0256/Users/earbsi/heatwaves/regridded/weights_0.01deg_bilinear.nc"
PROGRESS_JSON = "/uolstore/Research/b/b0256/Users/earbsi/heatwaves/regridded/progress.json"
YEAR_RANGE = range(2003, 2021)

RESET_OUTPUT = True



def get_regridder(source_ds, target_ds):
    if not os.path.exists(WEIGHTS_FILE):
        print("Computing weights...")
        ds_sample = source_ds.isel(time=0)

        regridder = xe.Regridder(
            ds_sample,
            target_ds,
            method="bilinear",
            periodic=False,
            parallel=False,
            unmapped_to_nan=True
        )
        regridder.to_netcdf(WEIGHTS_FILE)
    else:
        print("Reusing existing weights...")
        regridder = xe.Regridder(
                        source_ds,
                        target_ds,
                        method="bilinear",
                        periodic=False,
                        filename=WEIGHTS_FILE,
                        unmapped_to_nan=True,
                        reuse_weights=True
                    )
    return regridder

def fix_time_coord(ds, year):
    ds = ds.rename({'DOY': 'time'})
    tindex = xr.date_range(start=f"{year}-01-01", end=f"{year}-12-31",
                            freq='D', calendar='noleap')
    ds.coords['time'] = tindex
    return ds

def get_regridder_and_data(year):
    # create target grid
    lats = np.arange(-30, 20, .01)
    lons = np.arange(-82, -34.5, .01)
    target_ds = xr.Dataset({
        "lat": (["lat"], lats),
        "lon": (["lon"], lons),
    })

    
    source_ds = xr.open_dataset(f"{SOURCE_DIR}SouthAmerica_Ta_SVCMsp_{year}TMAX_daily_tropics30N30S.nc",
                                chunks={'DOY':1, 'lat':1024, 'lon':1024}).drop_vars('spatial_ref', errors='ignore')
    
    mask = xr.open_zarr(STORAGE_DIR + 'regrid/zz_tmax_nanmask.zarr')['mask'].all('year').astype('int8')
    source_ds['mask'] = mask

    source_ds = fix_time_coord(source_ds, year)
    regridder = get_regridder(source_ds, target_ds)


    regridded_ds = regridder(source_ds['Ta_TMAX'])
    return regridded_ds, regridder

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


#%%
if __name__ == "__main__":

    if RESET_OUTPUT and os.path.exists(ZARR_DESTINATION):
        print(f"Removing existing output at {ZARR_DESTINATION}...")
        os.system(f"rm -rf {ZARR_DESTINATION}")

    if RESET_OUTPUT and os.path.exists(PROGRESS_JSON):
        print(f"Removing existing progress file at {PROGRESS_JSON}...")
        os.remove(PROGRESS_JSON)


    progress = load_or_create_progress(list(YEAR_RANGE))
    completed_years = set(progress['completed_years'])

    with LocalCluster(n_workers=24, memory_limit='15GB',
                      threads_per_worker=16,
                      dashboard_address=':8787') as cluster:
        with Client(cluster) as client:
            print(f"Dask Dashboard available at: {client.dashboard_link}")
            for year in YEAR_RANGE:

                if year in completed_years:
                    print(f'{year}: already completed, skipping')
                    continue
                print(f'  [{year}] Starting regridding...')

                regridded_ds, regridder = get_regridder_and_data(year)

                

                print('saving to zarr...')
                if not os.path.exists(ZARR_DESTINATION):
                    print("Creating new Zarr store...")
                    regridded_ds.astype('float32').to_zarr(ZARR_DESTINATION, mode='w', consolidated=False)
                else:
                    print("Appending to existing Zarr store...")
                    regridded_ds.astype('float32').to_zarr(ZARR_DESTINATION, mode='a', append_dim='time', consolidated=False)

                completed_years.add(year)
                progress['completed_years'] = sorted(completed_years)
                save_progress(progress)


