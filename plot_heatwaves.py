#%%
import matplotlib.pyplot as plt
import xarray as xr
from dask.diagnostics import ProgressBar
import numpy as np
import cartopy.crs as ccrs


SFIGPATH = '/uolstore/home/users/earbsi/figs/heatwaves/'
HEATWAVES_PATH = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/'

heatwaves = xr.open_zarr(HEATWAVES_PATH+'heatdays/heatwaves.zarr')['heatwaves']

#%%
with ProgressBar():
    heatwaves_timesum = heatwaves.sum('time').compute(num_workers=16, num_threads=2)

#%% plot heatwaves days

levels = np.arange(0, 130, 10)

fig = plt.figure(figsize=[10,8])
ax = fig.add_subplot(projection=ccrs.PlateCarree())
heatwaves_timesum.plot.contourf(ax=ax, levels=levels, cmap='magma',
                                cbar_kwargs={'label':'Number of heatwave days',
                                             'shrink':0.8})
fig.savefig(SFIGPATH+'heatwave_days.png', dpi=400, bbox_inches='tight')

#%% plot heatwave days in amazon area
amazon_bbox = [-75, -30, -25, 10]
heatwaves_timesum_amazon = heatwaves_timesum.sel(lon=slice(amazon_bbox[0], amazon_bbox[1]),
                                                  lat=slice(amazon_bbox[3], amazon_bbox[2]))

fig = plt.figure(figsize=[10,8])
ax = fig.add_subplot(projection=ccrs.PlateCarree())
heatwaves_timesum_amazon.plot.contourf(ax=ax, levels=levels, cmap='magma',
                                        cbar_kwargs={'label':'Number of heatwave days',
                                                     'shrink':0.8})
fig.savefig(SFIGPATH+'heatwave_days_amazon.png', dpi=400, bbox_inches='tight')