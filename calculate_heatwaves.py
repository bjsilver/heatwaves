#%%
import xarray as xr
from glob import glob
from dask.diagnostics import ProgressBar
import multiprocessing as mp
from constants import LAT_STEP, LON_STEP
import time
import dask

HEATDAYS_CHUNKS_PATH = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/heatdays_chunks/'
HEATWAVE_DAY_THRESH = 3

#%%

heatdays_chunks_paths = glob(HEATDAYS_CHUNKS_PATH+'heatdays_*.nc')

chunkpath = heatdays_chunks_paths[0]

def load_heatdays_chunk(chunkpath):
    ds = xr.open_dataset(chunkpath)
    if '__xarray_dataarray_variable__' in ds.data_vars:
        ds = ds.rename({'__xarray_dataarray_variable__':'heatdays'})
    da = ds['heatdays']
    return da

#%%
da = load_heatdays_chunk(chunkpath)
da = da.chunk({'time':-1, 'lat':50, 'lon':50})

is_run_end = da.rolling(time=HEATWAVE_DAY_THRESH).all()
mask = is_run_end.rolling(time=HEATWAVE_DAY_THRESH, center=False).any()
mask = mask.shift(time=-(HEATWAVE_DAY_THRESH - 1))
da = da.where(mask.fillna(False), False)

with dask.config.set(scheduler='processes', num_workers=16):
    da.to_netcdf("filtered_output.nc")