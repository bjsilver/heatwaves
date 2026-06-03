#%%
import rioxarray as rxr
import xarray as xr
import pandas as pd
import numpy as np
import os
import dask.array as da
from functools import partial
from ufuncs import load_heat
from constants import HANSEN_DIR, STORAGE_DIR
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt

def open_hansen_loss(start_year, end_year):
    HANSEN_DIR = '/uolstore/Research/b/b0256/Users/earbsi/hansen/'

    ds = xr.open_zarr(HANSEN_DIR + 'hansen_tc_2000_2024_rechunked.zarr')
    tc = ds['tree_cover']
    tc = tc.sel(time=slice(str(start_year), str(end_year)))
    loss = tc.diff('time').sum('time')
    loss = -loss
    loss = loss.assign_coords(lat=np.round(loss.lat, 2), lon=np.round(loss.lon, 2))
    return loss

def calculate_heatwaves_trend(heatwaves):
    heatwaves_yearly = heatwaves.groupby('time.year').sum('time')
    heatwaves_trend = heatwaves_yearly.polyfit(dim='year', deg=1)
    trends = heatwaves_trend['polyfit_coefficients'].sel(degree=1)
    return trends




def _mwa_kernel(block, window_radius):
    """Numpy kernel for moving_window_analysis, called by dask.array.map_overlap.

    block: (2, H+2wr, W+2wr) — channel 0 = d_cc, channel 1 = dep_var.
    Returns: same shape, with results embedded in the centre (H, W) region;
    map_overlap trims the border automatically (trim=True).
    """
    wr = window_radius
    d_cc = block[0]
    dep  = block[1]
    H = d_cc.shape[0] - 2 * wr
    W = d_cc.shape[1] - 2 * wr

    center_cc  = d_cc[wr:wr+H, wr:wr+W]
    center_dep = dep [wr:wr+H, wr:wr+W]

    max_cc_diff  = np.full((H, W), -np.inf, dtype=d_cc.dtype)
    best_dep_diff = np.zeros((H, W), dtype=dep.dtype)
    has_higher   = np.zeros((H, W), dtype=bool)

    for di in range(-wr, wr + 1):
        for dj in range(-wr, wr + 1):
            if di == 0 and dj == 0:
                continue
            ri, rj = wr + di, wr + dj
            n_cc  = d_cc[ri:ri+H, rj:rj+W]
            n_dep = dep [ri:ri+H, rj:rj+W]
            diff  = n_cc - center_cc          # NaN if neighbour is out-of-bounds
            better = diff > max_cc_diff       # NaN comparisons are False — safe
            max_cc_diff   = np.where(better, diff, max_cc_diff)
            best_dep_diff = np.where(better, n_dep - center_dep, best_dep_diff)
            has_higher   |= (n_cc > center_cc)

    valid = (center_cc >= 0) & has_higher & ~np.isnan(center_dep)

    # embed results in centre of same-shape output; border is NaN (trimmed by map_overlap)
    out = np.full_like(block, np.nan)
    out[0, wr:wr+H, wr:wr+W] = np.where(valid, max_cc_diff,  np.nan)
    out[1, wr:wr+H, wr:wr+W] = np.where(valid, best_dep_diff, np.nan)
    return out


def moving_window_analysis(ds, window_radius, d_cc_var_name='d_cc',
                           dep_var_name='d_temp'):

    d_cc_arr = ds[d_cc_var_name].data  # dask (lat, lon)
    dep_arr  = ds[dep_var_name].data   # dask (lat, lon)

    # stack into (2, lat, lon); rechunk so both channels are in one block per tile
    stacked = da.stack([d_cc_arr, dep_arr]).rechunk({0: -1})

    result = da.map_overlap(
        partial(_mwa_kernel, window_radius=window_radius),
        stacked,
        depth={0: 0, 1: window_radius, 2: window_radius},
        boundary=np.nan,
        dtype=d_cc_arr.dtype,
    )

    dims   = ds[d_cc_var_name].dims
    coords = ds[d_cc_var_name].coords
    results = xr.Dataset(coords=ds.coords)
    results['cc_loss']    = xr.DataArray(result[0], dims=dims, coords=coords)
    results[f'{dep_var_name}_diff'] = xr.DataArray(result[1], dims=dims, coords=coords)
    return results

def calculate_binned_stats(results, csv_path=None,
                           dep_var_name='heatwaves_trend',
                           bin_width=10):
    """Calculate binned statistics and save to CSV.
    
    Parameters
    ----------
    results : xr.Dataset
        Moving window analysis results with 'cc_loss' and '{dep_var_name}_diff'.
    csv_path : str
        Path to save the CSV file with bin statistics.
    dep_var_name : str
        Name of the dependent variable (e.g., 'heatwaves_trend').
    bin_width : int
        Width of each bin in percent.
    """
    # Auto-detect max bin limit: find last bin with n > 3 observations
    preliminary_bins = np.arange(0, 110, bin_width)
    preliminary_counts = results[f'{dep_var_name}_diff'].groupby_bins(results['cc_loss'], preliminary_bins).count().compute()
    
    # Find the last bin with count > 3
    valid_bins = preliminary_counts[preliminary_counts > 3]
    valid_bins_pd = valid_bins.to_pandas()
    if len(valid_bins_pd) > 0:
        last_bin_upper = valid_bins_pd.index[-1].right
        max_bin_limit = last_bin_upper + bin_width  # Include one more bin
    else:
        max_bin_limit = bin_width  # Fallback if no bins have n > 3
    
    # Create final bins and compute statistics
    bins = np.arange(0, max_bin_limit, bin_width)
    binned = results[f'{dep_var_name}_diff'].groupby_bins(results['cc_loss'], bins).mean() 
    bin_counts = results[f'{dep_var_name}_diff'].groupby_bins(results['cc_loss'], bins).count()
    bin_stds = results[f'{dep_var_name}_diff'].groupby_bins(results['cc_loss'], bins).std()
    bin_sems = bin_stds / np.sqrt(bin_counts)
    bin_means = binned.values
    ci_lower = bin_means - 1.96 * bin_sems
    ci_upper = bin_means + 1.96 * bin_sems
    
    # Save the bin means, counts and CIs to a CSV for later use
    bin_dim = next(dim for dim in binned.dims if dim.endswith('_bins'))
    bin_stats = pd.DataFrame({
        'bin': [f'{int(interval.left)}-{int(interval.right)}' for interval in binned[bin_dim].values],
        'mean_diff': bin_means,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'count': bin_counts.values
    })
    if csv_path is not None:
        bin_stats.to_csv('./data/'+csv_path, index=False)
        print(f"Saved bin statistics to {csv_path}")
    else:
        return bin_stats


def slice_elev_mask(da):
    westlimit=-81.8; southlimit=-17.7; eastlimit=-41.6; northlimit=12.9
    da = da.sel(lat=slice(southlimit, northlimit), lon=slice(westlimit, eastlimit))
    elev = xr.open_dataset(STORAGE_DIR + '../elevation/elevation_regridded.nc')['elevation']
    elev = elev.reindex_like(da, method='nearest')
    da_elevmask = da.where(elev<1000)
    return da_elevmask

#%%

def calc_moving_window(coarsen_factor=None):

    elev = xr.open_dataset(STORAGE_DIR + '../elevation/elevation_regridded.nc')['elevation']


    loss = open_hansen_loss(2003, 2020)
    heatwaves = load_heat('heatwaves')['zz_2003-2020']
    heatwaves_trend = calculate_heatwaves_trend(heatwaves)
    ds = xr.Dataset({'d_cc': loss, 'heatwaves_trend': heatwaves_trend})

    ds = ds.where(elev < 1000)  # mask out high elevation areas

    if coarsen_factor:
        ds = ds.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary='trim').mean()

    results = moving_window_analysis(ds, window_radius=1, 
                                    d_cc_var_name='d_cc',
                                    dep_var_name='heatwaves_trend')

    results = results.compute()
    # heatwaves_trend.to_zarr(STORAGE_DIR + 'heatwaves/heatwaves_trend_2003_2020.zarr', mode='w')
    # results = results.to_zarr(STORAGE_DIR + f'heatwaves/moving_window_results_coarsened{coarsen_factor}.zarr', mode='w')

    # results = xr.open_zarr(STORAGE_DIR + f'heatwaves/moving_window_results_coarsened{coarsen_factor}.zarr')
    if coarsen_factor:
        csv_path = f'heatwave_trend_vs_cc_loss_bins_coarsened{coarsen_factor}_stats.csv'
    else:
        csv_path = f'heatwave_trend_vs_cc_loss_bins_coarsened_stats.csv'
    calculate_binned_stats(results, csv_path=csv_path, dep_var_name='heatwaves_trend',
                           bin_width=10)
 

#%%
if __name__ == "__main__":
    

    with LocalCluster(n_workers=32, threads_per_worker=1,
                            memory_limit='20GB') as cluster:
        with Client(cluster) as client:
            print(f"Dask Dashboard available at: {client.dashboard_link}")
            #calc_moving_window()
            # for coarsen_factor in [2, 5, 10, 15, 20, 25]:
            for coarsen_factor in [2, 3, 4]:
                calc_moving_window(coarsen_factor=coarsen_factor)

#%% moving window analysis for Tmax trends

def calc_moving_window_tmax_trends(trend_path, var_name='Ta_TMAX_trend',
                                   coarsen_factor=None):
    elev = xr.open_dataset(STORAGE_DIR + '../elevation/elevation_regridded.nc')['elevation']


    loss = open_hansen_loss(2003, 2020)
    trend = xr.open_zarr(trend_path)[var_name]
    ds = xr.Dataset({'d_cc': loss, 'trend': trend})

    ds = ds.where(elev < 1000)  # mask out high elevation areas

    if coarsen_factor:
        ds = ds.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary='trim').mean()

    results = moving_window_analysis(ds, window_radius=1, 
                                    d_cc_var_name='d_cc',
                                    dep_var_name='trend')

    return results

if __name__ == "__main__":
    with LocalCluster(n_workers=32, threads_per_worker=1,
                                memory_limit='20GB') as cluster:
            with Client(cluster) as client:
                print(f"Dask Dashboard available at: {client.dashboard_link}")

                # results = calc_moving_window_tmax_trends(STORAGE_DIR + 'zz_tmax_ts_trend.zarr')
                # results = results.compute()
                # calculate_binned_stats(results, csv_path='tmax_trend_vs_cc_loss_bins_stats.csv', 
                #                        dep_var_name='trend', bin_width=10)
                
                # results = calc_moving_window_tmax_trends(STORAGE_DIR + 'percentiles/zz_tmax_ts_trend_JJA.zarr', var_name='Ta_TMAX_trend_JJA')
                # results = results.compute()
                # calculate_binned_stats(results, csv_path='tmax_trend_jja_vs_cc_loss_bins_stats.csv', 
                #                        dep_var_name='trend', bin_width=10)
                
                results = calc_moving_window_tmax_trends(STORAGE_DIR + 'percentiles/zz_tmax_ts_trend_JJA.zarr', var_name='Ta_TMAX_trend_JJA',
                                                         coarsen_factor=10)
                results = results.compute()
                calculate_binned_stats(results, csv_path='tmax_trend_jja_vs_cc_loss_bins_coarsened10_stats.csv', 
                                       dep_var_name='trend', bin_width=10)

#%%
stats = calculate_binned_stats(results, dep_var_name='trend',
                           bin_width=10, csv_path='tmax_trend_vs_cc_loss_bins_stats.csv')