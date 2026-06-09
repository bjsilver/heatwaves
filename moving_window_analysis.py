#%%
import rioxarray as rxr
import xarray as xr
import pandas as pd
import numpy as np
import os
import dask.array as da
from functools import partial
from ufuncs import load_heat
from constants import HANSEN_DIR, PERCENTILES_DIR, STORAGE_DIR
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt

MWA_DIR = STORAGE_DIR + 'moving_window_analysis/'

def open_hansen_loss(start_year, end_year):
    ds = xr.open_zarr(HANSEN_DIR + 'hansen_tc_2000_2024_rechunked.zarr')
    tc = ds['tree_cover']
    tc = tc.sel(time=slice(str(start_year), str(end_year)))
    loss = tc.diff('time').sum('time')
    loss = -loss
    loss = loss.assign_coords(lat=np.round(loss.lat, 2), lon=np.round(loss.lon, 2))
    loss = loss.drop_vars('spatial_ref')
    return loss

def load_tmax_trends(mask_insignificant=True):
    dpath = os.path.join(PERCENTILES_DIR, 'zz_tmax_ts_trend.zarr')
    ds = xr.open_zarr(dpath)
    trends = ds['Ta_TMAX_trend']
    if mask_insignificant:
        pvals = ds['Ta_TMAX_trend_pvalue']
        trends = trends.where(pvals < 0.05)
    return trends

def combine_dataarrays_match_coords(da1, da2, da1_name='d_cc', da2_name='d_temp'):
    # round all coords to 2 decimals to ensure alignment
    da1 = da1.assign_coords(lat=np.round(da1.lat, 2), lon=np.round(da1.lon, 2))
    da2 = da2.assign_coords(lat=np.round(da2.lat, 2), lon=np.round(da2.lon, 2))
    # check if coords match
    if not da1.lat.equals(da2.lat) or not da1.lon.equals(da2.lon):
        raise ValueError("Latitude and longitude coordinates do not match between the two DataArrays.")
    # combine into a Dataset
    ds = xr.Dataset({da1_name: da1, da2_name: da2})
    return ds

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
        bin_stats.to_csv(MWA_DIR+csv_path, index=False)
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

def calc_moving_window(ind_var, dep_var, ind_var_name, dep_var_name, coarsen_factor=None):

    # generate file save name based on variable names and coarsen factor
    sname = f'{dep_var_name}_vs_{ind_var_name}'
    if coarsen_factor:
        sname += f'_coarsened{coarsen_factor}'

    elev = xr.open_dataset(STORAGE_DIR + '../elevation/elevation_regridded.nc')['elevation']

    ds = combine_dataarrays_match_coords(ind_var, dep_var, da1_name=ind_var_name, da2_name=dep_var_name)

    ds = ds.where(elev < 1000)  # mask out high elevation areas

    if coarsen_factor:
        ds = ds.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary='trim').mean()

    results = moving_window_analysis(ds, window_radius=1, 
                                    d_cc_var_name=ind_var_name,
                                    dep_var_name=dep_var_name)

    results.chunk({'lat': -1, 'lon': -1}).to_zarr(os.path.join(MWA_DIR, f'{sname}_mwa_results.zarr'),
                    mode='w', consolidated=True)
    
    results = xr.open_zarr(os.path.join(MWA_DIR, f'{sname}_mwa_results.zarr'), consolidated=True)
    
    csv_path = f'{sname}_stats.csv'
    calculate_binned_stats(results, csv_path=csv_path, dep_var_name=dep_var_name, bin_width=10)



#%% DO FOR TRENDS
if __name__ == "__main__":
    
    with LocalCluster(n_workers=16, threads_per_worker=1,
                            memory_limit='20GB') as cluster:
        with Client(cluster) as client:
            print(f"Dask Dashboard available at: {client.dashboard_link}")

            loss = open_hansen_loss(2003, 2020)
            tmax_trend = load_tmax_trends(mask_insignificant=False)
            for coarsen_factor in [None, 2, 4, 8, 16]:
                print(f"Running moving window analysis with coarsen_factor={coarsen_factor}...")
                calc_moving_window(ind_var=loss, dep_var=tmax_trend, 
                                   ind_var_name='d_cc', dep_var_name='tmax_trend',
                                   coarsen_factor=coarsen_factor)
                
#%% DO FOR SMOOTH TRENDS TOO

#%% do for heatwaves
if __name__ == "__main__":
    

    with LocalCluster(n_workers=16, threads_per_worker=1,
                                memory_limit='20GB') as cluster:
            with Client(cluster) as client:
                print(f"Dask Dashboard available at: {client.dashboard_link}")

                heatwaves = xr.open_zarr(STORAGE_DIR + 'heatdays/heatwaves_zz_2003-2020.zarr', consolidated=True)['heatwaves']
                heatwaves_annual = heatwaves.resample(time='YE').sum()
                heatwaves_diff = heatwaves_annual[-3:].sum('time') - heatwaves_annual[:3].sum('time')
                heatwaves_diff.name = 'd_heatwaves'

                loss = open_hansen_loss(2003, 2020)
                for coarsen_factor in [None, 2, 4, 8, 16]:
                    print(f"Running moving window analysis with coarsen_factor={coarsen_factor}...")
                    calc_moving_window(ind_var=loss, dep_var=heatwaves_diff, 
                                    ind_var_name='d_cc', dep_var_name='d_heatwaves',
                                    coarsen_factor=coarsen_factor)
                    
#%% do for mean temp diff
if __name__ == "__main__":
    

    with LocalCluster(n_workers=16, threads_per_worker=1,
                                memory_limit='20GB') as cluster:
            with Client(cluster) as client:
                print(f"Dask Dashboard available at: {client.dashboard_link}")
                tmax = xr.open_zarr(STORAGE_DIR + 'rechunked/zhang_zhou_regridded_rechunked.zarr')['Ta_TMAX']
                tmax_annual = tmax.resample(time='YE').mean()
                tmax_diff = tmax_annual[-5:].mean('time') - tmax_annual[:5].mean('time')


                loss = open_hansen_loss(2003, 2020)
                for coarsen_factor in [None, 2, 4, 8, 16]:
                    print(f"Running moving window analysis with coarsen_factor={coarsen_factor}...")
                    calc_moving_window(ind_var=loss, dep_var=tmax_diff, 
                                    ind_var_name='d_cc', dep_var_name='d_temp',
                                    coarsen_factor=coarsen_factor)
                    
#%% do for heatwave trends (raw and smoothed)
if __name__ == "__main__":
    
    with LocalCluster(n_workers=16, threads_per_worker=1,
                                memory_limit='20GB') as cluster:
            with Client(cluster) as client:
                print(f"Dask Dashboard available at: {client.dashboard_link}")
                heatwave_trends = xr.open_zarr(STORAGE_DIR + 'heatwaves/heatwaves_trend.zarr')['heatwaves_trend']
                heatwave_trends_smooth = xr.open_zarr(STORAGE_DIR + 'heatwaves/heatwaves_zz_2003-2020_smooth_trend.zarr')['heatwaves_trend']

                loss = open_hansen_loss(2003, 2020)
                for dep_var, dep_var_name in zip([heatwave_trends, heatwave_trends_smooth],
                                                 ['heatwave_trends', 'heatwave_trends_smooth']):
                    for coarsen_factor in [None, 2, 4, 8, 16]:
                        print(f"Running moving window analysis for {dep_var_name} with coarsen_factor={coarsen_factor}...")
                        calc_moving_window(ind_var=loss, dep_var=dep_var, 
                                        ind_var_name='d_cc', dep_var_name=dep_var_name,
                                        coarsen_factor=coarsen_factor)