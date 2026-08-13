"""
        year   lat    lon  dep_centre  dep_neighbour  ind_centre  ind_neighbour   ind_diff   dep_diff
20608   2015 -0.63 -78.13   11.546314      28.053053   30.856319      89.980331  59.124012 -16.506739
110045  2015 -0.58 -78.08   11.273528      27.585978   35.718719      90.786636  55.067917 -16.312450
85432   2012 -0.63 -78.13   11.720901      27.820055   30.860001      90.498878  59.638878 -16.099154
70665   2009 -0.63 -78.13   11.725979      27.690807   30.881916      90.951836  60.069920 -15.964828
174755  2012 -0.58 -78.08   11.476858      27.414602   35.722401      91.428017  55.705616 -15.937744
...      ...   ...    ...         ...            ...         ...            ...        ...        ...
100268  2013  2.92 -75.33   31.792444      21.017218   15.022282      78.467033  63.444752  10.775227
95413   2004  2.92 -75.33   31.679743      20.827450   15.790761      79.253113  63.462353  10.852293
105263  2014  2.92 -75.33   31.998085      21.002037   15.001631      78.439682  63.438049  10.996048
20829   2015  2.87 -75.38   32.587738      21.379774   15.767383      77.972328  62.204945  11.207964
110267  2015  2.92 -75.33   32.668358      20.986412   14.977716      78.426727  63.449013  11.681946
"""
#%% 

import matplotlib.ticker as mticker

import xarray as xr
from mwa import load_ind_dep_data, _generate_coarsened_versions, aggregate_dependent_variable
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import contextily as ctx

ind_var_name = 'cc'
dep_var_name = 'tmax'

ind, dep = load_ind_dep_data(ind_var_name=ind_var_name, dep_var_name=dep_var_name)
dep = aggregate_dependent_variable(dep, 'annual_mean')
dss = _generate_coarsened_versions(ind, dep, coarsen_factor=25)
#%%
year = 2015; lat = 2.92; lon = -75.33

for ind_da, dep_da in dss:
    if (lat in ind_da.lat.round(2)) and (lon in ind_da.lon.round(2)):
        print('found version with coords')
        break
lat_res = np.unique(np.diff(ind_da.lat).round(2))[0]
lon_res = np.unique(np.diff(ind_da.lon).round(2))[0]

ind_slice = ind_da.loc[{'lat':slice(lat-lat_res*2, lat+lat_res*2+lat_res), 
                        'lon':slice(lon-lon_res*2, lon+lon_res*2+lon_res),
                        'year':year}]
dep_slice = dep_da.loc[{'lat':slice(lat-lat_res*2, lat+lat_res*2+lat_res), 
                        'lon':slice(lon-lon_res*2, lon+lon_res*2+lon_res),
                        'year':year}]

# identify neighbour cell with max ind diff
centre = ind_slice.sel(lat=lat, lon=lon, method='nearest').compute().item()
max_diff = -np.inf
for i in range(-1, 2):
    for j in range(-1, 2):
        if i==0 and j==0:
            continue
        n_lat = lat+(lat_res*i)
        n_lon = lon+(lon_res*j)
        neighbour = ind_slice.sel(lat=n_lat, lon=n_lon, method='nearest').compute().item()
        diff = neighbour - centre
        if diff > max_diff:
            max_diff = diff
            max_neighbour_lat = n_lat
            max_neighbour_lon = n_lon
#%%

def format_ax(ax, max_neighbour_lat, max_neighbour_lon, lat, lon):
    ax.plot([lon-lon_res*1.5, lon+lon_res*1.5, lon+lon_res*1.5, lon-lon_res*1.5, lon-lon_res*1.5],
          [lat-lat_res*1.5, lat-lat_res*1.5, lat+lat_res*1.5, lat+lat_res*1.5, lat-lat_res*1.5],
          color='red', linewidth=2, transform=ccrs.PlateCarree())
    # add white cross with black outline using patheffects
    ax.plot(max_neighbour_lon, max_neighbour_lat, marker='x',
            markersize=10, color='white', markeredgewidth=3, transform=ccrs.PlateCarree(),
            path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=7, foreground='black')])
    # add black circle to centre cell
    ax.plot(lon, lat, marker='o', markersize=5, color='black', transform=ccrs.PlateCarree(),
            path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=4, foreground='white')])

def add_colourbar(fig, ax, im, label):
    cax = ax.inset_axes([0.1, -0.1, 0.8, 0.045])
    fig.colorbar(im, cax=cax, orientation='horizontal', 
                 label=label)

fig = plt.figure(figsize=(12, 6), layout='constrained')
grid = fig.add_gridspec(1, 3)

iax = fig.add_subplot(grid[0, 0], projection=ccrs.PlateCarree())
ind_plot = ind_slice.plot(ax=iax, cmap='Greens', vmin=0, vmax=100,
                          add_colorbar=False)
iax.set_title(ind_var_name)
format_ax(iax, max_neighbour_lat, max_neighbour_lon, lat, lon)
add_colourbar(fig, iax, ind_plot, f'{ind_var_name} (%)')


dax = fig.add_subplot(grid[0, 1], projection=ccrs.PlateCarree())
dep_plot = dep_slice.plot(ax=dax, cmap='magma',
                          vmin=dep_slice.min().compute(),
                          vmax=dep_slice.max().compute(),
                          add_colorbar=False)
dax.set_title(dep_var_name)
format_ax(dax, max_neighbour_lat, max_neighbour_lon, lat, lon)
add_colourbar(fig, dax, dep_plot, f'{dep_var_name} (°C)')
sax = fig.add_subplot(grid[0, 2], projection=ccrs.PlateCarree())
sax.set_extent([lon-lon_res*2.5, lon+lon_res*1.5+lon_res, 
                lat-lat_res*2.5, lat+lat_res*1.5+lat_res])


ctx.add_basemap(sax, crs="EPSG:4326", source=ctx.providers.Esri.WorldImagery,
        attribution='')
sax.set_title('Satellite Imagery')
format_ax(sax, max_neighbour_lat, max_neighbour_lon, lat, lon)
# add a grid that matches the ind_slice grid
gl = sax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=False,
    linewidth=0.8,
    linestyle='--',
    alpha=0.7,
)
gl.xlocator = mticker.FixedLocator(dep_slice.lon.values-lon_res/2)
gl.ylocator = mticker.FixedLocator(dep_slice.lat.values-lat_res/2)


