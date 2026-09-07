#%%
import xarray as xr
from constants import STORAGE_DIR
from ufuncs import local_cluster
from dask.diagnostics import ProgressBar
import numpy as np

HEATDAY_ABS_THRESH = 37.5
HEATWAVE_DAY_THRESH = 3
WARMEST_DAYS_NDAYS = 5


def load_data():
    zz = xr.open_zarr(STORAGE_DIR + 'rechunked/zhang_zhou_regridded_rechunked.zarr',
                    consolidated=True)['Ta_TMAX'] / 10
    return zz

tmax = load_data()


def calc_absthresh_heatwaves(tmax, abs_thresh, day_thresh):

    heatdays = tmax > abs_thresh

    is_run_end = heatdays
    for k in range(1, day_thresh):
        is_run_end = is_run_end & heatdays.shift(time=k, fill_value=False)

    mask = is_run_end
    for k in range(1, day_thresh):
        mask = mask | is_run_end.shift(time=-k, fill_value=False)

    heatwaves = heatdays & mask
    heatwaves_annual = heatwaves.resample(time='YS').sum(dim='time')
    heatwaves_annual = heatwaves_annual.rename({'time': 'year'})
    heatwaves_annual.coords['year'] = heatwaves_annual.year.dt.year
    heatwaves_annual.name= 'absthresh_heatwaves'
    return heatwaves_annual



def calc_warmest_days(tmax, ndays):
    years = np.unique(tmax.time.dt.year)
    das = []
    for year in years:
        tmax_year = tmax.sel(time=str(year))
        tmax_year_sorted = tmax_year.sortby('time', ascending=False)
        warmest_5days = tmax_year_sorted.isel(time=slice(0, ndays))
        mean_warmest = warmest_5days.groupby('time.year').mean(dim='time')
        das.append(mean_warmest)

    warmest_days = xr.concat(das, dim='year')
    warmest_days.name = f'warmest_days'
    return warmest_days

#%%

if __name__ == '__main__':

    with local_cluster(n_workers=32, threads_per_worker=1,
                       memory_limit='16GB') as cluster:
        ds = xr.Dataset({
            'absthresh_heatwaves': calc_absthresh_heatwaves(tmax, HEATDAY_ABS_THRESH, HEATWAVE_DAY_THRESH),
            'warmest_days': calc_warmest_days(tmax, WARMEST_DAYS_NDAYS)
        })
        spath = STORAGE_DIR + 'heat_metrics/heat_metrics.zarr'
        ds.to_zarr(spath, mode='w', consolidated=True, zarr_format=2)

#%% plot heat metrics
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

ds = xr.open_zarr(STORAGE_DIR + 'heat_metrics/heat_metrics.zarr', consolidated=True)

abs_thresh_heatwaves = ds['absthresh_heatwaves'].sum('year').compute()
mean_temp = ds['warmest_days'].max('year').compute()
#%%

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(projection=ccrs.PlateCarree())
abs_thresh_heatwaves.plot(cmap='magma', vmin=0, vmax=500,
                          ax=ax, transform=ccrs.PlateCarree(),
                          cbar_kwargs={'label': 'Number of heatwave days',
                                       'orientation': 'horizontal', 'shrink': 0.5,
                                       'pad': 0.05})
ax.coastlines(color='white')

#%%
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(projection=ccrs.PlateCarree())
mean_temp.plot(cmap='magma', vmin=30, vmax=40,
               ax=ax, transform=ccrs.PlateCarree(),
                cbar_kwargs={'label': 'Mean temperature of warmest 5 days (°C)',
                             'orientation': 'horizontal', 'shrink': 0.5,
                             'pad': 0.05})