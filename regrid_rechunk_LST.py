#%%
import dask
import xarray as xr
import rioxarray as rxr
import xesmf as xe
import numpy as np
import pandas as pd
from glob import glob
import os
from constants import LST_DIR, AREA_BBOX, STORAGE_DIR, HANSEN_DIR, LAT_CHUNK_SIZE, LON_CHUNK_SIZE
from pyproj import Transformer
from functools import lru_cache
from tqdm import tqdm
from ufuncs import local_cluster

TARGET_ZARR = STORAGE_DIR+'lst/zz_lst_regridded_rechunked.zarr'

@lru_cache(maxsize=None)
def get_mask():
    da = rxr.open_rasterio(LST_DIR+'zhang_zhou_tifs/2003/gf_Day2003_001.tif').squeeze()
    x_coords = da.x.values
    y_coords = da.y.values

    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    transformer = Transformer.from_crs(da.rio.crs, "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform(grid_x, grid_y)

    da = da.assign_coords(lon=(('y', 'x'), lons), lat=(('y', 'x'), lats))

    westlimit = AREA_BBOX['westlimit'] -1
    eastlimit = AREA_BBOX['eastlimit'] +1
    southlimit = AREA_BBOX['southlimit'] -1
    northlimit = AREA_BBOX['northlimit'] +1
    mask = (lons >= westlimit) & (lons <= eastlimit) & (lats >= southlimit) & (lats <= northlimit)
    masked_da = da.where(mask).dropna(dim='y', how='all').dropna(dim='x', how='all')

    return masked_da

@lru_cache(maxsize=None)
def get_lons_lats():
    da = rxr.open_rasterio(LST_DIR+'zhang_zhou_tifs/2003/gf_Day2003_001.tif').squeeze()
    x_coords = da.x.values
    y_coords = da.y.values

    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    transformer = Transformer.from_crs(da.rio.crs, "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform(grid_x, grid_y)
    return lons, lats



# open a single LST tif
def open_LST_tif(year, masked_da, lons, lats):
    da = xr.open_mfdataset(LST_DIR+f'zhang_zhou_tifs/{year}/gf_Day{year}_*.tif',
                        chunks={'y': 1000}, concat_dim='doy', combine='nested',
                        compat='override', engine='rasterio', coords='minimal')
    # da = da.chunk({'x': 1000, 'y': 1000})
    da = da.squeeze()



    da = da.assign_coords(lon=(('y', 'x'), lons), lat=(('y', 'x'), lats))

    # clip da to masked_da bounds
    da = da.sel(y=masked_da.y, x=masked_da.x)
    da = da.chunk({'x':1560, 'y':3257})

    da = da.rename({'band_data': 'LST'})
    return da

def get_regridder(lst_1):
    
    target = xr.open_zarr(HANSEN_DIR+'hansen_tc_2000_2024_rechunked.zarr')
    target = target.isel(time=0)

    lst_1['mask'] = lst_1['LST'] > -32768.

    target['mask'] = target['tree_cover'].notnull()

    regrid_path = LST_DIR+'zhang_zhou_tifs/regridder_LST_to_t2m_weights.nc'
    if not os.path.exists(regrid_path):
        regridder = xe.Regridder(lst_1, target, 'bilinear')
        regridder.to_netcdf(regrid_path)
    else:
        regridder = xe.Regridder(lst_1, target, 'bilinear', weights=regrid_path)

    return regridder

#%%

masked_da = get_mask()
lons, lats = get_lons_lats()
lst = open_LST_tif(2003, masked_da, lons, lats)
regridder = get_regridder(lst.isel(doy=0))

#%%
if __name__ == "__main__":
    with local_cluster(n_workers=16, threads_per_worker=1, memory_limit='40GB',
                       suppress_dask_warnings=True) as cluster:
        for year in range(2020, 2021):
            lst = open_LST_tif(year, masked_da, lons, lats)
            lst_regridded = regridder(lst)
            with dask.config.set({"array.rechunk.method": "p2p", "optimization.fuse.active": False}):
                lst_regridded = lst_regridded.chunk({'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})
            lst_regridded.to_zarr(LST_DIR+f'zz_lst_{year}.zarr', mode='w', consolidated=True)
