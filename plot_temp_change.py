#%%
import xarray as xr
from constants import STORAGE_DIR, FIGDIR
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar
import numpy as np
import cartopy.crs as ccrs
from matplotlib import gridspec

DPATH = STORAGE_DIR + 'rechunked/'  

def load_tmax_da():
    print(f"Opening input Zarr: {DPATH}zhang_zhou_regridded_rechunked.zarr")
    # Use consolidated=False to support stores without .zmetadata.
    ds = xr.open_zarr(DPATH + 'zhang_zhou_regridded_rechunked.zarr', chunks={}, consolidated=False)
    da = ds['Ta_TMAX']
    # check if zz time has duplicates
    if len(da.time) != len(np.unique(da.time)):
        print("Warning: Duplicate time entries found in zz dataset. Removing duplicates.")
        da = da.drop_duplicates('time')

    da.coords['lat'] = np.where(np.isnan(da.lat.values), 0, da.lat.values)

    if 'time' in da.dims:
        year_vals = da['time'].dt.year.data
        doy_vals = da['time'].dt.dayofyear.data
        da = da.assign_coords(year=('time', year_vals), doy=('time', doy_vals))
        da = da.set_index(time=['year', 'doy']).unstack('time').transpose('year', 'doy', 'lat', 'lon')

    if 'doy' not in da.coords:
        da = da.assign_coords(doy=np.arange(1, da.sizes['doy'] + 1))

    da = da.transpose('year', 'doy', 'lat', 'lon')

    print('Using native chunks from source/transform pipeline:')
    print(dict(zip(da.dims, da.chunks)))

    
    da = da / 10.0  # Convert from tenths of degrees to degrees Celsius.

    return da

tmax = load_tmax_da()
start_period = tmax[:3].mean('year').mean('doy')
end_period = tmax[-3:].mean('year').mean('doy')

with ProgressBar():
    diff = (end_period - start_period).compute(num_workers=32)

#%%

fig = plt.figure(figsize=(16, 10),
                 constrained_layout=True)
gs = fig.add_gridspec(20, 20)
# create a map axes on left half of the figure
ax_map = fig.add_subplot(gs[0:18, 0:9], projection=ccrs.PlateCarree())

cax = fig.add_subplot(gs[18:19, 2:8])
diff.plot.imshow(ax=ax_map, cmap='RdBu_r', vmin=-3, vmax=3,
                 cbar_kwargs={'label': '°C', 'shrink': 1,
                              'orientation':'horizontal'}, add_colorbar=True, cbar_ax=cax)

txt = 'Change in Annual Mean Tmax (°C)\n2018-2020 minus 2003-2005'
ax_map.text(0.5, .9, txt, transform=ax_map.transAxes, ha='center', fontsize=16)

# plot timeseries for sub area 
def add_sub_area_timeseries(ax, tmax_subarea, label):
    sub_area = tmax_subarea.mean(dim=['lat', 'lon'])
    # calc annual means and slope of annual means
    with ProgressBar():
        sub_area_annual = sub_area.mean(dim='doy').compute(num_workers=8)
    x = sub_area_annual.year
    y = sub_area_annual.values
    # calc slope
    slope = np.polyfit(x, y, 1)[0]
    sub_area_annual.plot(ax=ax, color='black', marker='o', label='Annual Mean Tmax', linewidth=.5)
    # plot trend slope as line
    ax.plot(x, slope * (x - x[0]) + y[0], color='red', linestyle='--', label=f'Trend: {slope:.2f} °C/year')
    ax.legend()

    # use sub_area to get lims and plot box on map
    lat_min = tmax_subarea.lat.min().item()
    lat_max = tmax_subarea.lat.max().item()
    lon_min = tmax_subarea.lon.min().item()
    lon_max = tmax_subarea.lon.max().item()
    ax_map.plot([lon_min, lon_max, lon_max, lon_min, lon_min], 
                [lat_min, lat_min, lat_max, lat_max, lat_min], 
                color='black', linewidth=1.5)
    # add label text in top corner of box
    ax_map.text(lon_min + 0.13, lat_max - 1, label, color='black', fontsize=12, fontweight='bold')
    # add label text to ax bottom right
    ax.text(0.95, 0.05, label, color='black', fontsize=12, fontweight='bold',
            transform=ax.transAxes, ha='right', va='bottom')
    

add_sub_area_timeseries(fig.add_subplot(gs[7:13, 10:20]),
                         tmax[:,:,1400:2000, 2000:2800],
                         label='B')
add_sub_area_timeseries(fig.add_subplot(gs[14:20, 10:20]),
                         tmax[:,:,200:1400, 2900:4100],
                         label='C')

add_sub_area_timeseries(fig.add_subplot(gs[0:6, 10:20]),
                         tmax[:,:,2200:3200, 1900:3400],
                         label='A')

fig.savefig(FIGDIR + '/trends/tmax_change_map_timeseries.png', dpi=300)


#%% make plot comparing diff to trends

trend = xr.open_zarr(STORAGE_DIR + 'percentiles/zz_tmax_ts_trend.zarr')['Ta_TMAX_trend']
fig = plt.figure(figsize=(16, 10), constrained_layout=True)
(ax1, ax2) = fig.subplots(1, 2, sharey=True, subplot_kw={'projection': ccrs.PlateCarree()})
diff.plot.imshow(ax=ax1, cmap='RdBu_r', vmin=-3, vmax=3, add_colorbar=False)
ax1.set_title('Change in Annual Mean Tmax (°C)\n2018-2020 minus 2003-2005', fontsize=16)
(trend*18).plot.imshow(ax=ax2, cmap='RdBu_r', vmin=-3, vmax=3, add_colorbar=False)
ax2.set_title('Trend in Annual Mean Tmax (°C/year)\n2003-2020', fontsize=16)
fig.savefig(FIGDIR + '/trends/tmax_change_vs_trend.png', dpi=300)

# trend diff diff = diff - (trend * 18)
trend_diff = diff - (trend * 18)
fig = plt.figure(figsize=(8, 10), constrained_layout=True)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
trend_diff.plot.imshow(ax=ax, cmap='RdBu_r', vmin=-2, vmax=2
, add_colorbar=True, cbar_kwargs={'label': 'Difference (°C)', 'shrink': 0.5})
ax.set_title('Difference between Observed Change and Trend Projection\n(Observed Change minus Trend Projection)', fontsize=16)
fig.savefig(FIGDIR + '/trends/tmax_change_vs_trend_diff.png', dpi=300)