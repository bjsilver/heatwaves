#%%
import os
from glob import glob

import numpy as np
import pandas as pd
from sklearn import cluster
import xarray as xr
import xesmf as xe
from dask.distributed import Client, LocalCluster
from numcodecs import Blosc
from dask.diagnostics import ProgressBar
import dask



# ---------------------------
# CONFIG
# ---------------------------
SOURCE_DIR = "/uolstore/Research/b/b0256/Users/earjba/ZhangandZhou/tropical_regions/"
ZARR_DESTINATION = "/uolstore/Research/b/b0256/Users/earbsi/heatwaves/regridded/zhang_zhou_regridded.zarr"
WEIGHTS_FILE = "/uolstore/Research/b/b0256/Users/earbsi/heatwaves/regridded/weights_0.01deg_bilinear.nc"

YEAR_RANGE = range(2003, 2021)

#%%
# WEIGHTS

def get_regridder(source_ds, target_ds):
    if not os.path.exists(WEIGHTS_FILE):
        print("Computing weights...")
        ds_sample = source_ds[0]

        regridder = xe.Regridder(
            ds_sample,
            target_ds,
            method="bilinear",
            periodic=False,
            parallel=False
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
                        reuse_weights=True
                    )
    return regridder

def get_regridder_and_data():
    # create target grid
    lats = np.arange(-30, 20, .01)
    lons = np.arange(-82, -34.5, .01)
    target_ds = xr.Dataset({
        "lat": (["lat"], lats),
        "lon": (["lon"], lons),
    })

    # add all-True spatial mask variable
    target_ds["mask"] = xr.DataArray(
        np.ones((len(target_ds.lat), len(target_ds.lon)), dtype=bool),
        dims=["lat", "lon"]
    )
    year = 2003
    source_ds = xr.open_dataset(f"{SOURCE_DIR}SouthAmerica_Ta_SVCMsp_{year}TMAX_daily_tropics30N30S.nc",
                                chunks={'DOY':1, 'lat':1024, 'lon':1024}).drop_vars('spatial_ref', errors='ignore')

    regridder = get_regridder(source_ds, target_ds)




    regridded_ds = regridder(source_ds)
    return regridded_ds, regridder



#%%
if __name__ == "__main__":
    with LocalCluster(n_workers=6, memory_limit='60GB',
                      threads_per_worker=12,
                      dashboard_address=':8787') as cluster:
        with Client(cluster) as client:

            regridded_ds, regridder = get_regridder_and_data()

            print(f"Dask Dashboard available at: {client.dashboard_link}")

            print('saving to zarr...')
            if not os.path.exists(ZARR_DESTINATION):
                regridded_ds.astype('float32').to_zarr(ZARR_DESTINATION, mode='w', consolidated=False)
            else:
                regridded_ds.astype('float32').to_zarr(ZARR_DESTINATION, mode='a', append_dim='DOY', consolidated=False)

