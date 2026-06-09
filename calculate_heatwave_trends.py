#%%
import xarray as xr
from constants import STORAGE_DIR
from dask.diagnostics import ProgressBar
from calculate_tmax_trends import _bootstrap_theilsen_1d_wrapper, maybe_local_cluster
import numpy as np

BOOTSTRAP_REPS = 250
BOOTSTRAP_SEED = 42


#%% resample to annual sum


#%%
def calculate_trend(fname):

    
    heatwaves = xr.open_zarr(STORAGE_DIR + f'heatdays/{fname}.zarr', consolidated=True)['heatwaves']

    heatwaves_annual = heatwaves.resample(time='YE').sum()
    heatwaves_annual = heatwaves_annual.rename({'time':'year'})
    heatwaves_annual = heatwaves_annual.assign_coords(year=heatwaves_annual.year.dt.year)


    heatwaves_annual = heatwaves_annual.chunk({'year': -1})
    trend, p_value = xr.apply_ufunc(
        _bootstrap_theilsen_1d_wrapper,
        heatwaves_annual,
        heatwaves_annual['year'].astype(np.float64),
        kwargs={'n_boot': BOOTSTRAP_REPS, 'seed': BOOTSTRAP_SEED},
        input_core_dims=[['year'], ['year']],
        output_core_dims=[[], []],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.float64, np.float64],
    )

    results = xr.Dataset({
        'heatwaves_trend': trend,
        'heatwaves_pvalue': p_value,
    })

    results.to_zarr(STORAGE_DIR + f'heatwaves/{fname}_trend.zarr', mode='w', consolidated=True)

#%% 

if __name__ == "__main__":
    with maybe_local_cluster(n_workers=64, threads_per_worker=1, memory_limit='6GB'):
        calculate_trend('heatwaves_zz_2003-2020.zarr')
        calculate_trend('heatwaves_zz_2003-2020_smooth')