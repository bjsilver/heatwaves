#%%
from itertools import count

import xarray as xr
import pandas as pd
import numpy as np
from dask.diagnostics import ProgressBar
from calculate_tmax_percentiles import load_tmax_da
from constants import MODIS_PATH, HANSEN_PATH, ERA5_PATH, AMAZON_BBOX, FIGDIR
import matplotlib.pyplot as plt
from ufuncs import lon_to_180




def safe_compute_dataset(ds, memory_budget_gb=70, safety_factor=2.0, num_workers=4):
    estimated_bytes = ds.nbytes
    estimated_peak_bytes = int(estimated_bytes * safety_factor)
    budget_bytes = int(memory_budget_gb * (1024 ** 3))

    if estimated_peak_bytes > budget_bytes:
        raise MemoryError(
            f"Estimated peak memory ({estimated_peak_bytes / (1024 ** 3):.1f} GB) exceeds "
            f"budget ({memory_budget_gb} GB). Reduce chunks/workers or compute in stages."
        )

    computed = {}
    for name, da in ds.data_vars.items():
        with ProgressBar():
            computed[name] = da.astype(np.float32).compute(
                scheduler='threads', num_workers=num_workers
            )

    return xr.Dataset(computed, coords=ds.coords)


def get_modis_temperature_change(year_window):
    # NOTE: does not account for missing data in any way

    ds = xr.open_dataset(MODIS_PATH+'MOD11A2_LST_0.25deg_2001–2021.nc')['LST_Day_1km']
    years = np.unique(ds.time.dt.year)
    start_years = years[:year_window]
    end_years = years[-year_window:]
    modis_start = ds.sel(time=ds.time.dt.year.isin(start_years)).mean('time')
    modis_end = ds.sel(time=ds.time.dt.year.isin(end_years)).mean('time')
    temperature_change = modis_end - modis_start
    return temperature_change

def get_hansen_deforestation(res, type='loss'):
    ds = xr.open_mfdataset(
        HANSEN_PATH+f'treeFrac*_{res}_hansen-landsat_mon_{res}deg_conserv.nc',
        compat='override',
        chunks={'latitude': 200, 'longitude': 200},
        parallel=False,
    )
    ds = ds.sel(AMAZON_BBOX)
    # round coords and ensure float32
    ds['latitude'] = np.round(ds['latitude'], 2).astype(np.float32)
    ds['longitude'] = np.round(ds['longitude'], 2).astype(np.float32)
    if type == 'loss':
        defor = ds['treeFrac2003_0_05'] - ds['treeFrac2020_0_05']
        return defor.astype(np.float32)
    else:
        return ds

def ensure_match_coords(ds1, ds2):
    lat_coords = np.intersect1d(ds1.latitude.values, ds2.latitude.values)
    lon_coords = np.intersect1d(ds1.longitude.values, ds2.longitude.values)
    ds1 = ds1.sel(latitude=lat_coords, longitude=lon_coords)
    ds2 = ds2.sel(latitude=lat_coords, longitude=lon_coords)
    return ds1, ds2

def prepare_modis_data():
    modis = get_modis_temperature_change(year_window=3)
    hansen = get_hansen_deforestation(res='0.25')

    modis = modis.sel(AMAZON_BBOX)

    modis, hansen = ensure_match_coords(modis, hansen)

    ds = xr.Dataset({
        'd_temp': modis,
        'd_cc': hansen
    })
    with ProgressBar():
        return ds.compute()



def moving_window_analysis(ds, window_radius, d_cc_var_name='d_cc', 
                           dep_var_name='d_temp'):

    # prep output dataset for results 
    results = xr.Dataset(coords=ds.coords)

    # gets all co-ordinate combinations except centre
    shifts = []
    for i in range(-window_radius, window_radius+1):
        for j in range(-window_radius, window_radius+1):
            if i == 0 and j == 0: 
                continue
            shifts.append((i, j))

    # create a ds with the window as a new dimension
    shifted_dss = [ds.shift(latitude=dx, longitude=dy) for dx, dy in shifts]
    nds = xr.concat(shifted_dss, dim=pd.RangeIndex(len(shifts), name='neighbour'))

    # only keep pixels where:
    valid = [
        ds[d_cc_var_name] > 0, #  CC loss data present and >0
        ds[d_cc_var_name] < nds[d_cc_var_name].max('neighbour'), # at least one neighbour with higher CC loss
        ds[dep_var_name].notnull(), # temperature change data present
        # how about edge pixels?
    ]
    valid_mask = np.logical_and.reduce(valid)

    # CC loss difference from centre for each neighbour
    cc_loss_diff = nds[d_cc_var_name] - ds[d_cc_var_name]
    # get chance in CC for the neighbour with greatest CC loss difference
    cc_loss_diff_max = cc_loss_diff.max('neighbour')
    # get indices of neighbour with greatest CC loss difference
    max_diff_idx = cc_loss_diff.fillna(-np.inf).argmax('neighbour')
    # get temperature change for neighbour with greatest CC loss difference
    temp_change_diff = nds[dep_var_name].isel(neighbour=max_diff_idx) - ds[dep_var_name]

    results['cc_loss'] = cc_loss_diff_max.where(valid_mask)
    results['temp_change'] = temp_change_diff.where(valid_mask)
    return results

def plot_binned_results(results, spath=None,
                        ylabel='neighbour mean temperature change (°C)',
                        xlabel='canopy cover loss bin (%)',
                        bin_lims=(0, 60), bin_width=5):
    # group by cc_loss_diff_max bins
    bins = np.arange(bin_lims[0], bin_lims[1], bin_width)
    binned = results['temp_change'].groupby_bins(results['cc_loss'], bins).mean() 
    n_in_bins = results['temp_change'].groupby_bins(results['cc_loss'], bins).count()
    # compute 95% CI for mean in each bin
    bin_counts = results['temp_change'].groupby_bins(results['cc_loss'], bins).count()
    bin_stds = results['temp_change'].groupby_bins(results['cc_loss'], bins).std()
    bin_sems = bin_stds / np.sqrt(bin_counts)
    bin_means = binned.values
    ci_lower = bin_means - 1.96 * bin_sems
    ci_upper = bin_means + 1.96 * bin_sems      


    fig, ax = plt.subplots(figsize=(10, 6))
    binned.to_series().plot(kind='bar', ax=ax, color='steelblue',
                            yerr=[bin_means - ci_lower, ci_upper - bin_means], capsize=4)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    bin_dim = next(dim for dim in binned.dims if dim.endswith('_bins'))
    ax.set_xticklabels(
        [f'{int(interval.left)}-{int(interval.right)}' for interval in binned[bin_dim].values],
        rotation=0,
    )
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add pixel counts on top of bars
    _, yuplim = ax.get_ylim()
    for i, (bar, count) in enumerate(zip(ax.patches, n_in_bins.values)):

        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., yuplim*.96,
                f'n={int(count)}',
                ha='center', va='bottom', fontsize=9)
    if spath:
        fig.savefig(spath, dpi=400, bbox_inches='tight')

def get_era5_temperature_change(year_window):
    ds = xr.open_mfdataset(
        ERA5_PATH+'era5_land_skinT_on_single_level_daily_*.nc',
        chunks={},
        parallel=False,
    )['skt'].astype(np.float32)
    ds = lon_to_180(ds)
    ds = ds.sortby('latitude')
    ds = ds.sel(AMAZON_BBOX)
    ds = ds.rename({'valid_time': 'time'})
    # round coords and ensure float32
    ds['latitude'] = np.round(ds['latitude'], 2).astype(np.float32)
    ds['longitude'] = np.round(ds['longitude'], 2).astype(np.float32)
    years = np.unique(ds.time.dt.year)
    start_years = years[:year_window]
    end_years = years[-year_window:]
    era_start = ds.sel(time=ds.time.dt.year.isin(start_years)).mean('time')
    era_end = ds.sel(time=ds.time.dt.year.isin(end_years)).mean('time')
    temperature_change = (era_end - era_start).astype(np.float32)
    return temperature_change

#%% analyse modis data
ds = prepare_modis_data()
results = moving_window_analysis(ds, window_radius=1)
plot_binned_results(results,
                    spath=FIGDIR+'moving_window_tests/modis_dtemp.png')


#%%


def prepare_era5_hansen_data():
    era = get_era5_temperature_change(year_window=3)
    
    hansen = get_hansen_deforestation(res=0.1)
    
    era, hansen = ensure_match_coords(era, hansen)
    ds = xr.Dataset({
        'd_temp': era,
        'd_cc': hansen
    })

    ds = safe_compute_dataset(ds, memory_budget_gb=70, safety_factor=2.0, num_workers=1)

    return ds

ds = prepare_era5_hansen_data()
results = moving_window_analysis(ds, window_radius=1)
plot_binned_results(results,
                    spath=FIGDIR+'moving_window_tests/era5_dtemp.png')

#%% same for era5 heatwaves data

def load_era_heatwaves_count_data(heatwave_year_range=None):
    OUTPUT_PATH = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/era5_heatwaves.zarr'
    da = xr.open_zarr(OUTPUT_PATH, consolidated=True)['heatwaves']
    if heatwave_year_range:
        da = da.sel(time=slice(f'{heatwave_year_range[0]}-01-01', f'{heatwave_year_range[1]}-12-31'))
    count = da.sum('time').compute()
    start_year, end_year = da.time.dt.year.min().item(), da.time.dt.year.max().item()
    count.attrs['units'] = f'n days during {start_year}-{end_year}'
    count.attrs['long_name'] = 'heatwave days'
    count.name = 'heatwave_day_count'
    count = count.rename({'lat': 'latitude', 'lon': 'longitude'})
    return count

def prepare_era5_heatwaves_hansen_data():
    heatwave_count = load_era_heatwaves_count_data(heatwave_year_range=(2015,2020))
    hansen = get_hansen_deforestation(res=0.25, type='loss')
    heatwave_count, hansen = ensure_match_coords(heatwave_count, hansen)
    ds = xr.Dataset({
        'heatwave_count': heatwave_count,
        'd_cc': hansen
    })
    return ds.compute()

ds = prepare_era5_heatwaves_hansen_data()
results = moving_window_analysis(ds, window_radius=1, d_cc_var_name='d_cc', dep_var_name='heatwave_count')
plot_binned_results(results,
                    xlabel='paired pixel difference in canopy cover bin (%)',
                    bin_lims=(0,55), bin_width=5,
                    ylabel='difference in deforestation (%)',
                    spath=FIGDIR+'moving_window_tests/era5_heatwave_count.png',)

#%% test with zz heatwaves data




def prepare_zz_heatwaves_hansen_data(heatwave_year_range=None):
    ZZ_HEATWAVES_PATH = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/heatdays/heatwaves.zarr'
    da = xr.open_zarr(ZZ_HEATWAVES_PATH, consolidated=True)['heatwaves']
    if heatwave_year_range:
        da = da.sel(time=slice(f'{heatwave_year_range[0]}-01-01', f'{heatwave_year_range[1]}-12-31'))
    count = da.sum('time')
    count.attrs['units'] = f'n days during {da.time.dt.year.min().item()}-{da.time.dt.year.max().item()}'
    count.attrs['long_name'] = 'heatwave days'
    count = count.sortby('lat')
    count = count.rename({'lat': 'latitude', 'lon': 'longitude'})
    count = count.sel(AMAZON_BBOX)
    count.coords['latitude'] = np.round(count.latitude, 3).astype(np.float32)
    count.coords['longitude'] = np.round(count.longitude, 3).astype(np.float32)


    hansen = get_hansen_deforestation(res=0.05, type='loss')
    # hansen = hansen['treeFrac2017_0_05']

    count, hansen = ensure_match_coords(count, hansen)

    ds = xr.Dataset({
        'heatwave_count': count,
        'd_cc': hansen
    })
    return ds.compute()

ds = prepare_zz_heatwaves_hansen_data(heatwave_year_range=(2015,2020))
results = moving_window_analysis(ds, window_radius=1, d_cc_var_name='d_cc', dep_var_name='heatwave_count')
plot_binned_results(results,
                    xlabel='difference in deforestation (%)',
                    bin_lims=(0,65), bin_width=5,
                    ylabel='mean difference in heatwave count (days)',
                    spath=FIGDIR+'moving_window_tests/zz_heatwave_count.png')

#%%
def prepare_zz_heatdays_hansen_data():
    ZZ_HEATDAYS_PATH = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/heatdays/heatwaves.zarr'
    da = xr.open_zarr(ZZ_HEATDAYS_PATH, consolidated=True)['heatwaves']
    start_years = [2003, 2004, 2005]
    end_years = [2018, 2019, 2020]
    start = da.sel(time=da.time.dt.year.isin(start_years)).sum('time')
    end = da.sel(time=da.time.dt.year.isin(end_years)).sum('time')
    change = end - start
    change.attrs['units'] = f'n days during {da.time.dt.year.min().item()}-{da.time.dt.year.max().item()}'
    change.attrs['long_name'] = 'heatdays'
    change = change.sortby('lat')
    change = change.rename({'lat': 'latitude', 'lon': 'longitude'})
    change = change.sel(AMAZON_BBOX)
    change.coords['latitude'] = np.round(change.latitude, 3).astype(np.float32)
    change.coords['longitude'] = np.round(change.longitude, 3).astype(np.float32)

    hansen = get_hansen_deforestation(res=0.05, type='loss')
    # hansen = hansen['treeFrac2017_0_05']

    change, hansen = ensure_match_coords(change, hansen)

    ds = xr.Dataset({
        'heatdays_count': change,
        'd_cc': hansen
    })
    return ds.compute()
ds = prepare_zz_heatdays_hansen_data()
results = moving_window_analysis(ds, window_radius=1, d_cc_var_name='d_cc', dep_var_name='heatdays_count')
plot_binned_results(results,
                    xlabel='difference in deforestation (%)',
                    bin_lims=(0,65), bin_width=5,
                    ylabel='mean difference in heatdays count (days)',
                    spath=FIGDIR+'moving_window_tests/zz_heatdays_count.png')

