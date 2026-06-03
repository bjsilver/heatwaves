#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:01:05 2024

@author: ben
"""
#%%
import xarray as xr
from dask.diagnostics import ProgressBar
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from constants import FIGDIR, STORAGE_DIR, PERCENTILE_KEYS
from calculate_heatdays_heatwaves import load_data as load_zz_era_pct
import cartopy.crs as ccrs
import contextily as ctx
from matplotlib.colors import LinearSegmentedColormap
import cartopy.feature as cfeature
from ufuncs import load_heat

CPU_COUNT = 32

zz, pctl = load_zz_era_pct()

pretty_percentile = {
    "zz_2003-2020": "Zhang and Zhou 2003-2020",
    "era_2003-2020": "ERA5 2003-2020",
    "era_1970-1999": "ERA5 1970-1999"}

#%%
with ProgressBar():
    pctlmean = pctl.mean("time").compute(num_workers=CPU_COUNT, memory_limit="120GB")

#%%

cmap_lims = {
    "zz_2003-2020": (28, 40),
    "era_2003-2020": (28, 40),
    "era_1970-1999": (22, 32),
}


for pctlkey in PERCENTILE_KEYS:

    fig, ax = plt.subplots()
    pctlmean[pctlkey].plot.imshow(ax=ax, cmap="magma", vmin=cmap_lims[pctlkey][0], vmax=cmap_lims[pctlkey][1],
                                cbar_kwargs={'shrink': 0.8})
    ax.set_title(f"Annual mean {pctlkey} daily max temperature 95th percentile")
    fig.savefig(FIGDIR + f"percentiles/{pctlkey}_mean.png", dpi=300, bbox_inches="tight")

#%% amazon zoom 
minlat, maxlat, minlon, maxlon = -15, 5, -70, -45
xticks = np.arange(minlon, maxlon, 5)
yticks = np.arange(minlat, maxlat, 5)
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
pctlmean['zz_2003-2020'].sel(lat=slice(minlat, maxlat), lon=slice(minlon, maxlon)).plot.imshow(
    ax=ax, origin='lower', cmap='magma', vmin=30, vmax=37,
    cbar_kwargs={'shrink': 0.5})
ax.set_xticks(xticks, crs=ccrs.PlateCarree())
ax.set_yticks(yticks, crs=ccrs.PlateCarree())
fig.savefig(FIGDIR + "percentiles/zz_2003-2020_amazon_zoom.png", dpi=300, bbox_inches="tight")

#%% zoom area satellite comparison

def _add_gridlines(ax, tick_interval, alpha=0.5):
    minlon, maxlon, minlat, maxlat = ax.get_extent(crs=ccrs.PlateCarree())
    eps = tick_interval * 1e-9
    xstart = np.ceil((minlon - eps) / tick_interval) * tick_interval
    xend = np.floor((maxlon + eps) / tick_interval) * tick_interval
    ystart = np.ceil((minlat - eps) / tick_interval) * tick_interval
    yend = np.floor((maxlat + eps) / tick_interval) * tick_interval

    xticks = np.arange(xstart, xend + eps, tick_interval)
    yticks = np.arange(ystart, yend + eps, tick_interval)

    # ensure ticks stay strictly within the original extent
    xticks = xticks[(xticks >= minlon - eps) & (xticks <= maxlon + eps)]
    yticks = yticks[(yticks >= minlat - eps) & (yticks <= maxlat + eps)]
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        xlocs=xticks,
        ylocs=yticks,
        linewidth=0.6,
        color='white',
        alpha=alpha,
        linestyle='--',
    )

def plot_satellite_comparison(data, extent, tick_interval=2,
                              save_fig=False, cmap="magma", cbar_label="°C",
                              vmin=None, vmax=None, prefix="zz_2003-2020"):

    minlat, maxlat, minlon, maxlon = extent

    da = data.sel(lat=slice(minlat, maxlat), lon=slice(minlon, maxlon))
    extent = [minlon, maxlon, minlat, maxlat]
    xticks = np.arange(minlon, maxlon, tick_interval)
    yticks = np.arange(minlat, maxlat, tick_interval)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})

    im = da.plot.imshow(ax=axes[0], cmap=cmap, add_colorbar=False, vmin=vmin, vmax=vmax)
    axes[0].set_extent(extent, crs=ccrs.PlateCarree())
    axes[0].set_aspect('equal', adjustable='box')
    _add_gridlines(axes[0], tick_interval=tick_interval)

    # add contextily satellite plot as second axis
    fig.delaxes(axes[1])
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlim(extent[0], extent[1])
    ax2.set_ylim(extent[2], extent[3])
    ax2.set_xticks(xticks)
    ax2.set_yticks(yticks)
    ax2.set_aspect('equal', adjustable='box')
    ctx.add_basemap(ax2, crs="EPSG:4326", source=ctx.providers.Esri.WorldImagery,
    attribution='')

    ax2.grid(True, linewidth=0.6, color='white', alpha=0.5, linestyle='--')

    # Lay out first, then add a dedicated colorbar axis under the left panel.
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.canvas.draw()
    left_pos = axes[0].get_position()
    cbar_height = 0.025
    cbar_pad = 0.1
    cax = fig.add_axes([left_pos.x0, left_pos.y0 - cbar_pad, left_pos.width, cbar_height])
    fig.colorbar(im, cax=cax, orientation='horizontal', label=cbar_label)

    if save_fig:
        area_string = f"{minlat}_{maxlat}_{minlon}_{maxlon}"
        fig.savefig(FIGDIR + f"percentiles/{prefix}_satcomp_{area_string}.png", dpi=300, bbox_inches="tight")
#%%
plot_satellite_comparison(pctlmean['zz_2003-2020'], extent=(0, 10, -68, -58), tick_interval=2, save_fig=True)
plot_satellite_comparison(pctlmean['zz_2003-2020'], extent=(4, 6, -68, -66), tick_interval=.4, save_fig=True)

#%%
plot_satellite_comparison(pctlmean['zz_2003-2020'], extent=(-16, -14, -61, -59), tick_interval=.4, save_fig=True)

#%%
plot_satellite_comparison(pctlmean, extent=(-11, -9, -56, -54), tick_interval=.4, save_fig=False)

#%% plot heatwaves



heatdays = load_heat('heatdays')
heatwaves = load_heat('heatwaves')

with ProgressBar():
    heatwaves_sum = heatwaves.sum(['time']).compute(num_workers=CPU_COUNT, memory_limit="120GB")

with ProgressBar():
    heatdays_sum = heatdays.sum(['time']).compute(num_workers=CPU_COUNT, memory_limit="120GB")

#%%
# calculate heatwaves per year
with ProgressBar():
    heatwaves_yearly = heatwaves.groupby('time.year').sum('time').compute(num_workers=CPU_COUNT, memory_limit="360GB")

#%%
def plot_heatwaves_sum(heatwave_da):
    plot_bounds = {'lat': slice(-25, 15), 'lon': slice(-80, -30)}

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    heatwave_da.loc[plot_bounds]\
    .plot.imshow(ax=ax, cmap='magma', vmin=0, vmax=heatwave_da.quantile(.9999).round(-2),
                                cbar_kwargs={'shrink': 0.6, 'label': 'heatwave day count'})
    _add_gridlines(ax, tick_interval=5, alpha=0.2)
    ax.set_title(f"{pretty_percentile[heatwave_da.name]} percentile\ntotal heatwave days 2003-2020")
    fig.savefig(FIGDIR + f"heatwaves/heatwaves_sum_{heatwave_da.name}.png", dpi=300, bbox_inches="tight")

for key in PERCENTILE_KEYS:
    plot_heatwaves_sum(heatwaves_sum[key])
#%% plot heatwaves - heatdays sum
heatwave_excess = heatwaves_sum - heatdays_sum
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
heatwave_excess['zz_2003-2020'].plot.imshow(ax=ax, cmap='Blues', vmin=0, vmax=-400,
                                cbar_kwargs={'shrink': 0.6, 'label': 'heatwave day count excess over heatdays'})
_add_gridlines(ax, tick_interval=5, alpha=0.2)
ax.set_title(f"Zhang and Zhou 2003-2020 percentile\nexcess heatwave days over heatdays 2003-2020")



#%% plot distribution of temperatures
bins = np.arange(20, 45, 1)
# dist = zz.groupby_bins(zz, bins).count()
# with ProgressBar():
#     dist = dist.compute(num_workers=CPU_COUNT, memory_limit="120GB")
# do for only single cell
cell_lat = slice(-10, -9)
cell_lon = slice(-56, -55)
dist = zz.sel(lat=cell_lat, lon=cell_lon).groupby_bins(zz.sel(lat=cell_lat, lon=cell_lon), bins=np.arange(20, 45, 1)).count().compute()

#%% plot distribution of temperatures
fig, ax = plt.subplots(figsize=(10, 6))
dist.plot(ax=ax)
# calculate skew and kurtosis
skew = dist.skew().values
kurtosis = dist.kurtosis().values

#%%
# calculate theilsen trend and significance 
years = heatwaves_yearly.year.values
heatwaves_trend = heatwaves_yearly['zz_2003-2020'].polyfit(dim='year', deg=1)
trends = heatwaves_trend['polyfit_coefficients'].sel(degree=1)

#%% trend plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
trends.plot.imshow(ax=ax, cmap='RdBu_r', vmin=-6, vmax=6, add_colorbar=True, cbar_kwargs={'label': 'Trend in heatwave days/year'})
ax.set_title("Trend in heatwave frequency 2003-2020\nZhang and Zhou 2003-2020 percentile")
_add_gridlines(ax, tick_interval=5, alpha=0.2)
fig.savefig(FIGDIR + "percentiles/heatwaves_trend_zz_2003-2020.png", dpi=300, bbox_inches="tight")

#%%

plot_satellite_comparison(trends, extent=(-17, -15, -62, -60), tick_interval=.4, save_fig=True,
                          cmap='RdBu_r', vmin=-5, vmax=5, cbar_label='Trend in heatwaves/year',
                          prefix='heatwaves_trend_')

plot_satellite_comparison(trends, extent=(-20, -0, -70, -40), tick_interval=5, save_fig=True,
                          cmap='RdBu_r', vmin=-7, vmax=7, cbar_label='Trend in heatwaves/year',
                          prefix='heatwaves_trend_')

#%% 
hansen = xr.open_dataarray(STORAGE_DIR + "../hansen/hansen_tc_2000_2024.nc")
hansen = hansen.chunk({'lat':200, 'lon':200})
# calculate loss
tcloss = hansen.sel(time=slice("2003", "2020")).diff('time').sum('time')
with ProgressBar():
    tcloss = tcloss.compute(num_workers=CPU_COUNT, memory_limit="120GB")

#%% plot hansen loss
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
# create a cmap that goes from transparent to green
cmap_green = LinearSegmentedColormap.from_list(
    'transparent_yellow_green',
    [
        (0.0, (1.0, 1.0, 0.8, 0.0)),   # transparent
        (0.1, (1.0, 1.0, 0.8, 0.35)),  # start pale yellow (~10)
        (0.3, (1.0, 1.0, 0.8, 1.0)),   # pale yellow (~30)
        (1.0, (0.0, 0.35, 0.0, 1.0)) # darker forest green (30-100)
    ],
    N=256
)
hansen.sel(time=2003).plot.imshow(ax=ax, cmap=cmap_green, vmin=0, vmax=100,
                                  add_colorbar=False)

# add ocean
ax.add_feature(cfeature.OCEAN, zorder=-1, facecolor='lightblue')
# add land
ax.add_feature(cfeature.LAND, zorder=-1, facecolor='cornsilk')

# create a cmap that goes from transparent to red
cmap = LinearSegmentedColormap.from_list('transparent_red', [(0, (1, 0, 0, 0)), (1, (1, 0, 0, 1))], N=256)
(-tcloss).plot.imshow(ax=ax, cmap=cmap, vmin=0, vmax=100, add_colorbar=False)
# add both colourbars
cax1 = fig.add_axes([0.1, 0.06, 0.35, 0.02])
cax2 = fig.add_axes([0.55, 0.06, 0.35, 0.02])
ax.set_ylabel(""); ax.set_xlabel("")
fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_green, norm=plt.Normalize(vmin=0, vmax=100)), cax=cax1, orientation='horizontal', label='Tree cover % in 2003')
fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=100)), cax=cax2, orientation='horizontal', label='Tree cover loss % 2003-2020')

_add_gridlines(ax, tick_interval=5, alpha=0.5)

fig.savefig(FIGDIR + "hansen/hansen_loss.png", dpi=300, bbox_inches="tight")