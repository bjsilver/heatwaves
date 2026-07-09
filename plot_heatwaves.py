#%%
import matplotlib.pyplot as plt
import xarray as xr
from dask.diagnostics import ProgressBar
import numpy as np
import cartopy.crs as ccrs
from constants import AMAZON_BBOX, FIGDIR, STORAGE_DIR
from ufuncs import _add_gridlines
import contextily as ctx    


HEATWAVES_PATH = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/'


ds = xr.Dataset({
    'heatwaves': xr.open_zarr(HEATWAVES_PATH+'heatdays/heatwaves_annual_sum_zz_2003-2020.zarr')['heatwave_days'],
    'heatwaves_smooth': xr.open_zarr(HEATWAVES_PATH+'heatdays/heatwaves_annual_sum_zz_2003-2020_smooth.zarr')['heatwave_days'],
    'heatwaves_gapfilled': xr.open_zarr(HEATWAVES_PATH+'heatdays/heatwaves_annual_sum_zz_2003-2020_gapfilled.zarr')['heatwave_days'],
    # 'heatwaves_era5-p95': xr.open_zarr(HEATWAVES_PATH+'heatdays/heatwaves_annual_sum_era5_2003-2020.zarr')['heatwave_days'],
    'heatwaves_era5-p95-zzadjusted': xr.open_zarr(HEATWAVES_PATH+'heatdays/heatwaves_annual_sum_era5_2003-2020_zzadjusted.zarr')['heatwave_days'],
})

#%%
with ProgressBar():
    ds = ds.sum('year').compute(num_workers=64, num_threads=1)

#%% plot comparison

def plot_heatwave_maps(ds, extent, satellite_plot=False,
                       tick_interval=2):
    minlat, maxlat, minlon, maxlon = extent

    fig, axes_all = plt.subplots(2, 2, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(12, 10)) 
    if satellite_plot:
        axes = axes_all.flatten()[:-1]
    else:
        axes = axes_all.flatten()
    for ax, var in zip(axes, ds.data_vars):
        da = ds[var].sel(lat=slice(minlat, maxlat), lon=slice(minlon, maxlon))
        im = da.plot.imshow(ax=ax, cmap="magma", vmin=0, vmax=350, add_colorbar=False)
        ax.set_title(var)
        _add_gridlines(ax, tick_interval=tick_interval, alpha=0.2)

    if satellite_plot:
        sat = axes_all.flatten()[-1]
        sat.set_xlim(minlon, maxlon)
        sat.set_ylim(minlat, maxlat)
        sat.set_aspect('equal', adjustable='box')
        ctx.add_basemap(sat, crs="EPSG:4326", source=ctx.providers.Esri.WorldImagery,
        attribution='')
        _add_gridlines(sat, tick_interval=tick_interval, alpha=0.2)


    # add colorbar
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Total Heatwave Days (2003-2020)', rotation=270, labelpad=15)
    plt.tight_layout(rect=[0, 0, 0.9, 1])

plot_heatwave_maps(ds, extent=(-20, 10, -65, -35),
                   tick_interval=5)

#%%
plot_heatwave_maps(ds, extent=(-11, -7, -65, -61), tick_interval=2,
                   satellite_plot=True)

#%%
#westlimit=-56.4652; southlimit=-7.5912; eastlimit=-52.0267; northlimit=-3.867
plot_heatwave_maps(ds, extent=(-7.6, -3.6, -56, -52), tick_interval=2,
                   satellite_plot=True)

#%%
westlimit=-60.8982; southlimit=-15.1557; eastlimit=-56.4597; northlimit=-11.5137
plot_heatwave_maps(ds, extent=(southlimit, northlimit, westlimit, eastlimit), tick_interval=2,
                   satellite_plot=True)