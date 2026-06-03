#%%
import os

import numpy as np
import xarray as xr
import xesmf as xe
import regionmask

from constants import ERA5_PATH, STORAGE_DIR
from calc_era5_percentiles import maybe_local_cluster, save_heatwaves

WEIGHTS_FILE = ERA5_PATH + 'era5_regrid_weights_0.01deg_bilinear.nc'
INPUT_PATH = STORAGE_DIR + 'percentiles/era5_percentile_2003-2020.zarr'
OUTPUT_PATH = STORAGE_DIR + 'regridded/era5_skt_p95_2003-2020_regridded.zarr'


def get_regridder(source_da, target_ds):
    if not os.path.exists(WEIGHTS_FILE):
        print('Computing weights...')
        source_sample = source_da.isel(doy=0)
        regridder = xe.Regridder(
            source_sample,
            target_ds,
            method='bilinear',
            periodic=False,
            parallel=False,
        )
        regridder.to_netcdf(WEIGHTS_FILE)
    else:
        print('Reusing existing weights...')
        regridder = xe.Regridder(
            source_da,
            target_ds,
            method='bilinear',
            periodic=False,
            filename=WEIGHTS_FILE,
            reuse_weights=True,
        )
    return regridder


def regrid_to_zz(source_da):
    lats = np.arange(-30, 20, 0.01)
    lons = np.arange(-82, -34.5, 0.01)
    target_ds = xr.Dataset(
        {
            'lat': (['lat'], lats),
            'lon': (['lon'], lons),
        }
    )

    target_ds['mask'] = xr.DataArray(
        np.ones((len(target_ds.lat), len(target_ds.lon)), dtype=bool),
        dims=['lat', 'lon'],
    )

    source_da = source_da.chunk({'doy': 5, 'lat': -1, 'lon': -1})
    regridder = get_regridder(source_da, target_ds)
    return regridder(source_da)


def load_percentile_data(input_path=INPUT_PATH):
    ds = xr.open_zarr(input_path)
    if 'skt_p95' in ds:
        return ds['skt_p95']
    if 'heatwaves' in ds:
        return ds['heatwaves']
    return ds[list(ds.data_vars)[0]]


#%%
if __name__ == '__main__':
    with maybe_local_cluster():
        p95 = load_percentile_data()
        regridded_p95 = regrid_to_zz(p95)
        save_heatwaves(regridded_p95, output_path=OUTPUT_PATH)
 