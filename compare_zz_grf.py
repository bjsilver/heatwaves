#%%
import xarray as xr
from constants import TMAX_RECHUNKED_FILE, GRF_RESULTS_DIR
from ufuncs import ensure_match_coords
from dask.diagnostics import ProgressBar

zz = xr.open_zarr(TMAX_RECHUNKED_FILE)['Ta_TMAX'] / 10
grf = xr.open_zarr(GRF_RESULTS_DIR+'predictions/grf_tmax_2003_2020.zarr')['tmax'] 



# ensure coords between the two datasets are aligned
grf, zz = ensure_match_coords(grf, zz)

ds = xr.Dataset({'zz': zz, 'grf': grf})
ds['diff'] = ds['zz'] - ds['grf']


#%%
with ProgressBar():
    diff = ds['diff'][:365].mean(dim='time').compute(num_threads=8)
    diff.dropna(dim='lat', how='all').dropna(dim='lon', how='all').plot(cmap='RdBu_r', vmin=-5, vmax=5)