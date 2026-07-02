#%%
import os
import pickle
import numpy as np
import xarray as xr
from constants import MWA_DAILY_DIR  
from ufuncs import local_cluster
from glob import glob

COARSEN_FACTORS = [None, 2, 5, 10, 25]


westlimit=-69.7; southlimit=-13.7; eastlimit=-46.4; northlimit=5.6
RESULTS_BBOX = {'lat':slice(southlimit, northlimit), 'lon':slice(westlimit, eastlimit)}

def retrieve_ind_dep_results(sname, ind_var_name, dep_var_name):

    # get file paths for all versions of the combined zarr files
    zarr_filepaths = sorted(glob(os.path.join(MWA_DAILY_DIR, f'{sname}_mwa_results*.zarr')))
    if not zarr_filepaths:
        raise FileNotFoundError(f'No combined MWA result stores found for {sname}')

    all_inds = [xr.open_zarr(path, consolidated=False)[ind_var_name + '_cdiff'] for path in zarr_filepaths]
    all_deps = [xr.open_zarr(path, consolidated=False)[dep_var_name + '_cdiff'] for path in zarr_filepaths]


    # Concatenate across versions (if multiple)
    if len(all_inds) > 1:
        ind = xr.concat(all_inds, dim='version', join='outer')
        dep = xr.concat(all_deps, dim='version', join='outer')
    else:
        ind = all_inds[0]
        dep = all_deps[0]

    return ind, dep

def calculate_binned_stats(dep_var_name, ind_var_name, sname=None,
                           save_pickle=True):
    """Calculate binned statistics from moving window analysis results.
    
    Handles both single zarr files and multiple versions (from offset resampling).
    """
    
    ind, dep = retrieve_ind_dep_results(sname, ind_var_name, dep_var_name)
    ind = ind.loc[RESULTS_BBOX]
    dep = dep.loc[RESULTS_BBOX]

    reduce_dims = ['lat', 'lon']
    if 'version' in ind.dims:
        reduce_dims.append('version')

    bins = np.arange(10, 110, 10)
    binned = dep.groupby_bins(ind, bins=bins)
    stats_summed = xr.Dataset({
        'sum': binned.sum(reduce_dims),
        'count': binned.count(reduce_dims),
        'sumsq': (dep**2).groupby_bins(ind, bins=bins).sum(reduce_dims),
    })


    stats_summed['mean'] = stats_summed['sum'] / stats_summed['count']
    stats_summed['std'] = np.sqrt(stats_summed['sumsq'] / stats_summed['count'] - stats_summed['mean']**2)

    stats_computed = stats_summed.compute()

    if save_pickle:
        pickle_path = os.path.join(MWA_DAILY_DIR, f'{sname}_binned_stats.P')
        print(f"Saving binned statistics to as pickle {pickle_path}...")
        with open(pickle_path, 'wb') as f:
            pickle.dump(stats_computed, f)
    else:
        return stats_computed
    
def calc_stats_all_coarsens(dep_var_name, ind_var_names, driver_ind_var_name,
                            ind_var_name_for_bins, aggregation='annual_mean',
                            coarsen_factors=COARSEN_FACTORS):
    """Calculate binned statistics for all specified coarsen factors."""
    ind_tag = '+'.join(ind_var_names)
    for coarsen_factor in coarsen_factors:
        sname = f'{dep_var_name}_vs_{ind_tag}_driver_{driver_ind_var_name}_{aggregation}'
        if coarsen_factor:
            sname += f'_coarsened{coarsen_factor}'
        print(f"Calculating binned statistics for {sname}...")
        calculate_binned_stats(
            dep_var_name=dep_var_name,
            ind_var_name=ind_var_name_for_bins,
            sname=sname,
            save_pickle=True,
        )
#%%


# dep_var_name = 'tmax'
# ind_var_name = 'cc_loss'
# aggregation = 'annual_95pct'
# coarsen_factor = 25
# sname = f'{dep_var_name}_vs_{ind_var_name}_{aggregation}'

# if coarsen_factor:
#     sname += f'_coarsened{coarsen_factor}'


# # for coarsen_factor in COARSEN_FACTORS:
# #     # check if output zarr already exists

    

# calculate_binned_stats(dep_var_name=dep_var_name, ind_var_name=ind_var_name, sname=sname, save_pickle=True)

# #     break

#%%
if __name__ == "__main__":
    with local_cluster(n_workers=24, threads_per_worker=1, memory_limit='6GB',
                       suppress_dask_warnings=False, use_dask_dashboard=True) as client:
        pass
        # calc_stats_all_coarsens(dep_var_name='tmax', ind_var_names=['cc_loss'],
        #                        driver_ind_var_name='cc_loss', ind_var_name_for_bins='cc_loss',
        #                        aggregation='annual_mean', coarsen_factors=COARSEN_FACTORS)
        calc_stats_all_coarsens(dep_var_name='tmax', ind_var_names=['cc_loss', 'forest_loss_age'],
                    driver_ind_var_name='cc_loss', ind_var_name_for_bins='cc_loss',
                    aggregation='annual_95pct', coarsen_factors=COARSEN_FACTORS)
        # calc_stats_all_coarsens(dep_var_name='heatwaves', ind_var_names=['cc_loss'],
        #                        driver_ind_var_name='cc_loss', ind_var_name_for_bins='cc_loss',
        #                        aggregation='annual_sum', coarsen_factors=COARSEN_FACTORS)
        # calc_stats_all_coarsens(dep_var_name='heatwaves_smoothed', ind_var_names=['cc_loss'],
        #                        driver_ind_var_name='cc_loss', ind_var_name_for_bins='cc_loss',
        #                        aggregation='annual_sum', coarsen_factors=COARSEN_FACTORS)
