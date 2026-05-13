#%%
import matplotlib.pyplot as plt
import xarray as xr
from dask.diagnostics import ProgressBar
import numpy as np
import cartopy.crs as ccrs
from constants import AMAZON_BBOX, FIGDIR


HEATWAVES_PATH = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/'

heatwaves = xr.open_zarr(HEATWAVES_PATH+'heatdays/heatwaves.zarr')['heatwaves']

#%%
with ProgressBar():
    heatwaves_timesum = heatwaves.sum('time').compute(num_workers=16, num_threads=2)

#%% plot heatwaves days

levels = np.arange(0, 220, 20)

fig = plt.figure(figsize=[10,8])
ax = fig.add_subplot(projection=ccrs.PlateCarree())
heatwaves_timesum.plot.contourf(ax=ax, levels=levels, cmap='magma',
                                cbar_kwargs={'label':'Number of heatwave days',
                                             'shrink':0.8})
fig.savefig(FIGDIR+'heatwave_days.png', dpi=400, bbox_inches='tight')

#%% plot heatwave days in amazon area
amazon_bbox = [-75, -30, -25, 10]
heatwaves_timesum = heatwaves_timesum.rename({'lat':'latitude', 'lon':'longitude'})
heatwaves_timesum_amazon = heatwaves_timesum.sel(longitude=slice(amazon_bbox[0], amazon_bbox[1]),
                                                latitude=slice(amazon_bbox[2], amazon_bbox[3]))

fig = plt.figure(figsize=[10,8])
ax = fig.add_subplot(projection=ccrs.PlateCarree())
heatwaves_timesum_amazon.plot.contourf(ax=ax, levels=levels, cmap='magma',
                                        cbar_kwargs={'label':'Number of heatwave days',
                                                     'shrink':0.8})
fig.savefig(FIGDIR+'heatwave_days_amazon.png', dpi=400, bbox_inches='tight')

#%% compare heatwave days with ERA5- and ZZ-derived percentile threshold
zz = xr.open_zarr(HEATWAVES_PATH+'heatdays/heatdays.zarr')['heatdays']
era = xr.open_zarr(HEATWAVES_PATH+'heatdays/heatdays_era5pctl.zarr')['heatdays']

ds = xr.Dataset({'zz':zz, 'era':era})

with ProgressBar():
    timesum = ds.sum(['year', 'doy']).compute(num_workers=16, num_threads=4)