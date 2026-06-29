#%%
import xarray as xr
from mwa_stats import retrieve_ind_dep_results
from dask.diagnostics import ProgressBar

ind_var_name = 'cc_loss'
dep_var_name = 'tmax'
time_res = 'annual_mean'
coarsen_factor = None
sname = f'{dep_var_name}_vs_{ind_var_name}_{time_res}'
if coarsen_factor:
    sname += f'_coarsened{coarsen_factor}'


ind, dep = retrieve_ind_dep_results(sname, ind_var_name, dep_var_name)
ind = ind.rename({'time': 'year'})
ind.coords['year'] = dep.year

with ProgressBar():

    da = dep.where(ind > 70).mean('year').compute(num_workers=18)

#%%