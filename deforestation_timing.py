#%%
import xarray as xr
from constants import HANSEN_DIR
from dask.diagnostics import ProgressBar

da = xr.open_zarr(HANSEN_DIR + 'hansen_tc_2000_2024_rechunked.zarr')['tree_cover']

#%%
events = (-da.diff(dim='time') > 0).sum('time')
with ProgressBar():
    events = events.compute(num_workers=16)