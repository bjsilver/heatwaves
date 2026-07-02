#%%
import xarray as xr
from constants import STORAGE_DIR
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import contextily as ctx
import numpy as np
from ufuncs import _add_gridlines
from dask.diagnostics import ProgressBar

ERA5_PATH = STORAGE_DIR + 'rechunked/era5_skt_p95_2003-2020_rechunked.zarr'
ZZ_PATH = STORAGE_DIR + 'rechunked/zz_pct_rechunked.zarr'
era = xr.open_zarr(ERA5_PATH, consolidated=True)['Ta_TMAX']
zz = xr.open_zarr(ZZ_PATH, consolidated=True)['Ta_TMAX_p95'] / 10
zz = zz[:, :-1, :-1]  # remove last row and column to match ERA5 grid
era = era.assign_coords({'lat': zz.lat, 'lon': zz.lon})

ds = xr.Dataset({'era5': era, 'zz': zz})

#%%
with ProgressBar():
    ds = ds.mean('time').compute(num_workers=32, memory_limit='12GB')


#%%


def plot_satellite_comparison(ds, extent, tick_interval=2, cmap='magma', vmin=32, vmax=36):
    minlat, maxlat, minlon, maxlon = extent

    ds_clipped = ds.sel(lat=slice(minlat, maxlat), lon=slice(minlon, maxlon))
    extent = [minlon, maxlon, minlat, maxlat]

    fig, axes = plt.subplots(2, 2, figsize=(12,10), subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()

    im = ds_clipped['zz'].plot.imshow(ax=axes[0], cmap=cmap, add_colorbar=False, vmin=vmin, vmax=vmax)
    axes[0].set_extent(extent, crs=ccrs.PlateCarree())
    axes[0].set_aspect('equal', adjustable='box')
    _add_gridlines(axes[0], tick_interval=tick_interval)
    # add title in a box
    axes[0].set_title('Zhang and Zhou tmax 95th percentile annual mean', 
                    fontsize=10, fontweight='bold', pad=10, loc='center', y=.92, 
                    bbox=dict(facecolor='white', edgecolor='None', boxstyle='round,pad=0.3', alpha=.7))

    im = ds_clipped['era5'].plot.imshow(ax=axes[1], cmap=cmap, add_colorbar=False, vmin=vmin, vmax=vmax)
    axes[1].set_extent(extent, crs=ccrs.PlateCarree())
    axes[1].set_aspect('equal', adjustable='box')
    _add_gridlines(axes[1], tick_interval=tick_interval)
    axes[1].set_title('ERA5 (regridded) tmax 95th percentile annual mean', 
                    fontsize=10, fontweight='bold', pad=10, loc='center', y=.92, 
                    bbox=dict(facecolor='white', edgecolor='None', boxstyle='round,pad=0.3', alpha=.7))

    axes[2].set_extent(extent, crs=ccrs.PlateCarree())
    ctx.add_basemap(axes[2], crs="EPSG:4326", source=ctx.providers.Esri.WorldImagery,
    attribution='')
    _add_gridlines(axes[2], tick_interval=tick_interval)

    diff =  ds_clipped['era5'] - ds_clipped['zz']
    diff_im = diff.plot.imshow(ax=axes[3], cmap='RdBu_r', add_colorbar=False, vmin=-3, vmax=3)
    axes[3].set_extent(extent, crs=ccrs.PlateCarree())
    axes[3].set_aspect('equal', adjustable='box')
    _add_gridlines(axes[3], tick_interval=tick_interval)
    axes[3].set_title('Difference (ERA5 - Zhang and Zhou)', 
                    fontsize=10, fontweight='bold', pad=10, loc='center', y=.92, 
                    bbox=dict(facecolor='white', edgecolor='None', boxstyle='round,pad=0.3', alpha=.7))

    # Lay out first, then add a dedicated colorbar axis 
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.canvas.draw()
    left_pos = axes[1].get_position()
    cbar_width = 0.025
    cbar_pad = 0.02
    cax = fig.add_axes([left_pos.x1 + cbar_pad, left_pos.y0, cbar_width, left_pos.height])
    fig.colorbar(im, cax=cax, orientation='vertical', label='Temperature (°C)')

    left_pos = axes[3].get_position()
    cbar_width = 0.025
    cbar_pad = 0.02
    diffcax = fig.add_axes([left_pos.x1 + cbar_pad, left_pos.y0, cbar_width, left_pos.height])
    fig.colorbar(diff_im, cax=diffcax, orientation='vertical', label='Temperature Difference (°C)')
#%%

plot_satellite_comparison(ds, extent=(-12, -6, -61, -53), tick_interval=2, cmap='magma', vmin=32, vmax=36)

#%%
westlimit=-55.23; southlimit=-11.0; eastlimit=-48.29; northlimit=-4.04
plot_satellite_comparison(ds, extent=(southlimit, northlimit, westlimit, eastlimit), tick_interval=2, cmap='magma', vmin=32, vmax=38)