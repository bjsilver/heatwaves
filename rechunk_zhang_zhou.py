#%%
from rechunker import rechunk
import xarray as xr
from glob import glob
from dask.diagnostics import ProgressBar
import pandas as pd
import re


DPATH = '/uolstore/Research/b/b0256/Users/earjba/ZhangandZhou/tropical_regions/'


ds_source = xr.open_mfdataset(DPATH+'SouthAmerica_Ta_SVCMsp_*TMAX_daily_tropics30N30S.nc', 
                               concat_dim='DOY', combine='nested',
                              chunks = {'lat': 200, 'lon': 200, 'time': -1})
ds_source = ds_source.drop_vars('spatial_ref')

#%%

target_chunks = {
    'DOY': -1,      # All time in one chunk (ideal for rolling means)
    'lat': 40, 
    'lon': 40
}

target_store = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/rechunked/zhang_zhou_rechunked.zarr'
temp_store = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/rechunked/rechunker_tmp.zarr'


array_plan = rechunk(
    ds_source, 
    target_chunks=target_chunks, 
    max_mem='8GB', 
    target_store=target_store, 
    temp_store=temp_store
)

with ProgressBar(): 
    array_plan.execute()


