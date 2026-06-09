#%%
import matplotlib.pyplot as plt
import xarray as xr
from dask.diagnostics import ProgressBar
import numpy as np
import cartopy.crs as ccrs
from constants import AMAZON_BBOX, FIGDIR, STORAGE_DIR


HEATWAVES_PATH = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/'
heatwaves = xr.open_zarr(HEATWAVES_PATH+'heatdays/heatwaves_zz_2003-2020.zarr')['heatwaves']
hw_smooth = xr.open_zarr(HEATWAVES_PATH+'heatdays/heatwaves_zz_2003-2020_smooth.zarr')['heatwaves']

ds = xr.Dataset({
    'heatwaves': heatwaves,
    'heatwaves_smooth': hw_smooth,
})

#%%
with ProgressBar():
    ds = ds.sum('time').compute(num_workers=64, num_threads=1)

#%% plot heatwaves days

levels = np.arange(0, 600, 50)

fig = plt.figure(figsize=[10,8])
ax = fig.add_subplot(projection=ccrs.PlateCarree())
ds['heatwaves'].plot.contourf(ax=ax, levels=levels, cmap='magma',
                                cbar_kwargs={'label':'Number of heatwave days',
                                             'shrink':0.8})
fig.savefig(FIGDIR+'heatwaves/heatwave_days_new.png', dpi=400, bbox_inches='tight')
#%%
fig = plt.figure(figsize=[10,8])
ax = fig.add_subplot(projection=ccrs.PlateCarree())
ds['heatwaves_smooth'].plot.contourf(ax=ax, levels=levels, cmap='magma',
                                cbar_kwargs={'label':'Number of heatwave days',
                                             'shrink':0.8})
fig.savefig(FIGDIR+'heatwaves/heatwave_days_new_smoothed.png', dpi=400, bbox_inches='tight')

#%% compare raw vs smoothed
extent = [-4, -3, -54, -53]

fig = plt.figure(figsize=[12,6])
ax = fig.add_subplot(1,2,1, projection=ccrs.PlateCarree())
ds['heatwaves'].sel(lat=slice(extent[0], extent[1]), lon=slice(extent[2], extent[3])).plot.imshow(ax=ax,
                                                                                                  cmap='magma', vmin=0, vmax=300,
                                cbar_kwargs={'label':'Number of heatwave days',
                                             'shrink':0.6})
ax.set_title('Raw')
ax = fig.add_subplot(1,2,2, projection=ccrs.PlateCarree())
ds['heatwaves_smooth'].sel(lat=slice(extent[0], extent[1]), lon=slice(extent[2], extent[3])).plot.imshow(ax=ax,
                                                                                                  cmap='magma', vmin=0, vmax=500,
                                cbar_kwargs={'label':'Number of heatwave days',
                                             'shrink':0.6})
ax.set_title('Smoothed')
fig.savefig(FIGDIR+'/heatwaves/heatwave_days_comparison.png', dpi=400, bbox_inches='tight')

#%% plot heatwave trends
heatwave_trends = xr.open_zarr(STORAGE_DIR+'heatwaves/heatwaves_trend.zarr')['heatwaves_trend']
heatwave_trends_smooth = xr.open_zarr(STORAGE_DIR+'heatwaves/heatwaves_zz_2003-2020_smooth_trend.zarr')['heatwaves_trend']

ds_trends = xr.Dataset({
    'heatwave_trends': heatwave_trends,
    'heatwave_trends_smooth': heatwave_trends_smooth,
})
fig = plt.figure(figsize=[10,8])
ax = fig.add_subplot(projection=ccrs.PlateCarree())
ds_trends['heatwave_trends'].plot.imshow(ax=ax, vmin=-4, vmax=4, cmap='bwr',
                                cbar_kwargs={'label':'Trend in heatwave days per year',
                                             'shrink':0.8})


fig.savefig(FIGDIR+'heatwaves/heatwave_trends.png', dpi=400, bbox_inches='tight')   

fig = plt.figure(figsize=[10,8])
ax = fig.add_subplot(projection=ccrs.PlateCarree())
ds_trends['heatwave_trends_smooth'].plot.imshow(ax=ax, vmin=-4, vmax=4, cmap='bwr',
                                cbar_kwargs={'label':'Trend in heatwave days per year',
                                             'shrink':0.8})
fig.savefig(FIGDIR+'heatwaves/heatwave_trends_smoothed.png', dpi=400, bbox_inches='tight')

#%% plot diff in heatwaves trend
heatwave_trends_diff = ds_trends['heatwave_trends_smooth'] - ds_trends['heatwave_trends']
fig = plt.figure(figsize=[10,8])
ax = fig.add_subplot(projection=ccrs.PlateCarree())
heatwave_trends_diff.plot.imshow(ax=ax, vmin=-2, vmax=2, cmap='bwr',
                                cbar_kwargs={'label':'Difference in trend (smoothed - raw)',
                                             'shrink':0.8})
fig.savefig(FIGDIR+'heatwaves/heatwave_trends_diff.png', dpi=400, bbox_inches='tight')