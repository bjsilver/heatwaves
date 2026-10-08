#%%

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import contextily as ctx
from mwa import load_ind_dep_data
from constants import FIGDIR, HEATDAY_TEMP_THRESH, HEATWAVE_DAY_THRESH, STORAGE_DIR
from calculate_heatdays_heatwaves import remove_short_runs
import imageio
import io
from tqdm import tqdm
import numpy as np
import cartopy.feature as cf
import os

def calculate_doy_percentile(tmax, percentile=0.95, window=15):
    """Calculate a circular day-of-year percentile using a ±7-day window."""
    years = np.unique(tmax.time.dt.year)
    tmax = tmax.values.reshape(len(years), 365)

    pad = window // 2
    padded = np.concatenate([tmax[:, -pad:], tmax, tmax[:, :pad]], axis=1)

    windows = np.lib.stride_tricks.sliding_window_view(
        padded, window, axis=1
    )

    pctl = np.nanpercentile(windows, percentile * 100, axis=(0, 2))

    return xr.DataArray(
        pctl,
        dims='dayofyear',
        coords={'dayofyear': np.arange(1, 366)},
        name='doy_percentile'
    )

# load data
WLAT = -65
SLICER = {'lat': slice(-10.5, -9.5), 'lon': slice(WLAT, WLAT+1), 'year': 2020}
ind_var, cc_loss, dep_var = load_ind_dep_data(ind_var_name='cc', dep_var_name='tmax', include_cc_loss=True)
dep_var_annual = dep_var.groupby('time.year').mean(dim='time')
tmax = dep_var_annual.loc[{s: k for s, k in SLICER.items() if s != 'year'}].compute()
elev = xr.open_dataset(os.path.join(STORAGE_DIR, '..', 'elevation/orig_resolution_regridded', 'elevation_regridded.nc'))['elevation']

#%% background img
OUTER_SLICER = {'lat': slice(-15, -8), 'lon': slice(-54 - 12.44444444, -54), 'year': 2020}
fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([OUTER_SLICER['lon'].start, OUTER_SLICER['lon'].stop, OUTER_SLICER['lat'].start, OUTER_SLICER['lat'].stop], crs=ccrs.PlateCarree())
ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Esri.WorldImagery, attribution='')
# ensure no whitespace around the map
ax.margins(0)

fig.savefig(FIGDIR +'presentation_plots/background.png', 
            dpi=400, bbox_inches='tight')


#%% overview plots

mean_tc = ind_var.mean(dim=['lat', 'lon']).compute(num_workers=len(ind_var.year))


fig = plt.figure(figsize=(12, 12))
map_ax = fig.add_subplot( projection=ccrs.PlateCarree())
map_ax.add_feature(cf.OCEAN)
ind_var[0].plot.imshow(ax=map_ax, cmap='Greens', vmin=0, vmax=100,
                    cbar_kwargs={'label': 'Tree cover [%]', 'orientation': 'horizontal',
                                 'pad': 0.02, 'shrink': 0.5})
map_ax.set_title('')

fig.savefig(FIGDIR +'presentation_plots/tc_map.png', 
            dpi=400, bbox_inches='tight')

#%%
fig = plt.figure(figsize=(12, 12))
map_ax = fig.add_subplot( projection=ccrs.PlateCarree())
map_ax.add_feature(cf.OCEAN)

dep_var_annual[0].plot.imshow(ax=map_ax, cmap='magma', vmin=5, vmax=35,
                    cbar_kwargs={'label': 'Tmax [°C]', 'orientation': 'horizontal',
                                 'pad': 0.02, 'shrink': 0.5})
map_ax.set_title('')
fig.savefig(FIGDIR +'presentation_plots/tmax_map.png', 
            dpi=400, bbox_inches='tight')

#%% elevation plot
fig = plt.figure(figsize=(12, 12))
map_ax = fig.add_subplot( projection=ccrs.PlateCarree())
map_ax.add_feature(cf.OCEAN)
elev.plot.imshow(ax=map_ax, cmap='terrain', vmin=-10, vmax=1000, transform=ccrs.PlateCarree(),
                    cbar_kwargs={'label': 'Elevation [m]', 'orientation': 'horizontal',
                                 'pad': 0.02, 'shrink': 0.5})
map_ax.set_title('')
fig.savefig(FIGDIR +'presentation_plots/elevation_map.png', 
            dpi=400, bbox_inches='tight')

#%%

fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([SLICER['lon'].start, SLICER['lon'].stop, SLICER['lat'].start, SLICER['lat'].stop], crs=ccrs.PlateCarree())
ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Esri.WorldImagery, attribution='')
ax.margins(0)
fig.savefig(FIGDIR +'presentation_plots/satellite.png', 
            dpi=400, bbox_inches='tight')

#%%
def make_annual_gif(data, years, output_path, cmap, vmin, vmax, cbar_label, duration=500):
    frames = []

    for year in tqdm(years, desc='creating frames'):
        fig, ax = plt.subplots(figsize=(12, 12),
                               subplot_kw={'projection': ccrs.PlateCarree()})

        data.loc[{'year': year}].plot.imshow(
            ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
            cbar_kwargs={'label': cbar_label, 'orientation': 'horizontal',
                         'pad': 0.01, 'shrink': 0.5},
        )

        ax.set_extent(
            [SLICER['lon'].start, SLICER['lon'].stop,
             SLICER['lat'].start, SLICER['lat'].stop],
            crs=ccrs.PlateCarree(),
        )
        ax.margins(0)
        ax.set_title(f'{year}', fontsize=16, y=.95, x=.05,
                     fontweight='bold', color='white')

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        frames.append(imageio.v2.imread(buf))
        plt.close(fig)
        buf.close()

    imageio.mimsave(output_path, frames, backend='pillow', palettesize=256,
                    subrectangles=True, quantizer='nq', duration=duration, loop=0)

make_annual_gif(data=ind_var, 
                years=range(2003, 2021),
                output_path=FIGDIR + 'presentation_plots/tc.gif',
                cmap='Greens',
                vmin=0,
                vmax=100,
                cbar_label='Tree cover [%]')


# calc annual dep var

make_annual_gif(data=tmax,
                years=range(2003, 2021),
                output_path=FIGDIR + 'presentation_plots/tmax.gif',
                cmap='magma',
                vmin=31,
                vmax=34,
                cbar_label='Tmax [°C]')



#%%
fig, ax = plt.subplots(figsize=(12, 12),
                               subplot_kw={'projection': ccrs.PlateCarree()})
ind_var.loc[SLICER].plot.imshow(
    ax=ax, cmap='Greens', vmin=0, vmax=100,
    cbar_kwargs={'label': 'Tree cover [%]', 'orientation': 'horizontal',
                    'pad': 0.01, 'shrink': 0.5},
)
ax.set_title('2020', fontsize=16, y=.95, x=.05,
                     fontweight='bold', color='white')

fig.savefig(FIGDIR +'presentation_plots/tc_2020.png',
            dpi=400, bbox_inches='tight')

#%%
# plot tmax 2020
fig, ax = plt.subplots(figsize=(12, 12),
                               subplot_kw={'projection': ccrs.PlateCarree()})
dep_var_annual.loc[SLICER].plot.imshow(
    ax=ax, cmap='magma', vmin=33.5, vmax=34.75,
    cbar_kwargs={'label': 'Tmax [°C]', 'orientation': 'horizontal',
                    'pad': 0.01, 'shrink': 0.5, 'extend':'neither'},
)
ax.set_title('2020', fontsize=16, y=.95, x=.05,
                     fontweight='bold', color='white')

# save lims for later use
SNIP_LIMS = ax.get_xlim(), ax.get_ylim()

fig.savefig(FIGDIR +'presentation_plots/tmax_2020.png',
            dpi=400, bbox_inches='tight')

#%%
COARSEN_FACTORS = [2, 4, 8, 16, 32]
def coarsen_da(da, cf, shift=0):
    da = da.isel(lat=slice(shift, None), lon=slice(shift, None))
    da = da.coarsen({'lat': cf, 'lon': cf}, boundary='trim').mean()
    return da





def make_coarsened_gif(data, output_prefix, cmap, vmin, vmax, cbar_label,
                       year=2020, duration=700):

    frames = []
    for cf in COARSEN_FACTORS:
        data_coarse = coarsen_da(data, cf).loc[{
            'lat': slice(SLICER['lat'].start - 1, SLICER['lat'].stop + 1),
            'lon': slice(SLICER['lon'].start - 1, SLICER['lon'].stop + 1),
            'year': year,
        }]

        fig, ax = plt.subplots(figsize=(12, 12),
                               subplot_kw={'projection': ccrs.PlateCarree()})

        data_coarse.plot.imshow(
            ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
            cbar_kwargs={'label': cbar_label, 'orientation': 'horizontal',
                         'pad': 0.01, 'shrink': 0.5, 'extend': 'neither'},
        )

        ax.set_title(f'{year}', fontsize=16, y=.95, x=.05,
                     fontweight='bold', color='white')
        ax.set_xlim(SNIP_LIMS[0])
        ax.set_ylim(SNIP_LIMS[1])
        ax.text(0.85, 0.03, f'res ≈ {cf} km', transform=ax.transAxes,
                fontsize=12, color='white',
                bbox=dict(facecolor='black', alpha=0.5))

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        frames.append(imageio.v2.imread(buf)[..., :3])
        plt.close(fig)
        buf.close()

    imageio.mimsave(
        FIGDIR + f'presentation_plots/{output_prefix}.gif',
        frames,
        backend='pillow',
        palettesize=256,
        subrectangles=False,
        quantizer='nq',
        duration=duration,
        loop=0,
    )

make_coarsened_gif(
    data=dep_var_annual,
    output_prefix='tmax_2020_coarse',
    cmap='magma',
    vmin=33.5,
    vmax=34.75,
    cbar_label='Tmax [°C]',
    duration=2000
)

make_coarsened_gif(
    data=ind_var,
    output_prefix='tc_2020_coarse',
    cmap='Greens',
    vmin=0,
    vmax=100,
    cbar_label='Tree cover [%]',
    duration=2000
)

#%% illustrate coarsening shift



frames = []
for shift in [0, 5, 10, 15, 20]:

    ind_var_coarse = coarsen_da(ind_var, 25, shift=shift)

    data_coarse = ind_var_coarse.loc[{
                'lat': slice(SLICER['lat'].start - 1, SLICER['lat'].stop + 1),
                'lon': slice(SLICER['lon'].start - 1, SLICER['lon'].stop + 1),
                'year': 2020,
            }]

    fig, axes = plt.subplots(1, 2 ,figsize=(24, 12),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    ind_var.loc[SLICER].plot.imshow(
        ax=axes[0], cmap='Greens', vmin=0, vmax=100,
        cbar_kwargs={'label': 'Tree cover [%]', 'orientation': 'horizontal',
                        'pad': 0.01, 'shrink': 0.5},
    )
    axes[0].set_title('2020', fontsize=16, y=.95, x=.05,
                        fontweight='bold', color='white')

    # add 25*25 grid to show coarsening
    axes[0].set_xticks(data_coarse.lon.values-.25/2, minor=True)
    axes[0].set_yticks(data_coarse.lat.values-.25/2, minor=True)
    axes[0].grid(which='minor', color='red', linestyle='-', linewidth=3, alpha=1)
    # remove tick marks and labels
    axes[0].tick_params(axis='both', which='both', length=0)
    axes[0].set_xticklabels([])
    axes[0].set_yticklabels([])
    axes[0].set_xlabel(''); axes[0].set_ylabel('')
    axes[0].set_xlim(SNIP_LIMS[0])
    axes[0].set_ylim(SNIP_LIMS[1])

    data_coarse.plot.imshow(
        ax=axes[1], cmap='Greens', vmin=0, vmax=100,
        cbar_kwargs={'label': 'Tree cover [%]', 'orientation': 'horizontal',
                        'pad': 0.01, 'shrink': 0.5},
    )

    axes[1].set_xlim(SNIP_LIMS[0])
    axes[1].set_ylim(SNIP_LIMS[1])
    axes[1].set_title('2020', fontsize=16, y=.95, x=.05,
                        fontweight='bold', color='white')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    frames.append(imageio.v2.imread(buf)[..., :3])
    plt.close(fig)
    buf.close()

imageio.mimsave(
    FIGDIR + f'presentation_plots/tc_2020_coarse_shift.gif',
    frames,
    backend='pillow',
    palettesize=256,
    subrectangles=False,
    quantizer='nq',
    duration=1000,
    loop=0,
)
#%% visualise percentile calculation
def tmax_map(sname, labelcoords=[]):
    fig, ax = plt.subplots(figsize=(8, 8),
                                subplot_kw={'projection': ccrs.PlateCarree()})
    dep_var_annual.loc[SLICER].plot.imshow(
        ax=ax, cmap='magma', vmin=33.5, vmax=34.75,
        cbar_kwargs={'label': 'Tmax [°C]', 'orientation': 'horizontal',
                        'pad': 0.01, 'shrink': 0.5, 'extend':'neither'},
    )
    ax.set_title('2020', fontsize=12, y=.95, x=.05,
                        fontweight='bold', color='white')

    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i, coords in enumerate(labelcoords):
        ix,iy = coords
        pixel = dep_var_annual.loc[SLICER].isel(lat=iy, lon=ix)
        x, y = pixel.lon.values, pixel.lat.values
        ax.annotate(alphabet[i], xy=(x, y), xytext=(x+0.05, y+0.05),
                    fontsize=16, fontweight='bold', color='white',
                    arrowprops=dict(facecolor='white', shrink=0.05, width=1, headwidth=5,
                                    headlength=5, color='white'))

    fig.savefig(FIGDIR +f'presentation_plots/{sname}.png',
                dpi=400, bbox_inches='tight')

tmax_map('tmax_2020_labeled_a', labelcoords=[(54, 50)])
tmax_map('tmax_2020_labeled_ab', labelcoords=[(54, 50), (30, 70)])

#%%

def tmax_tseries(sname_suffix, labelcoords=[], heatwaves=None,
                 line_alpha=1, col_list=None, i_base=0,
                 percentile_window_doy=None):
    if col_list is None:
        col_list = [('limegreen', 'forestgreen'),
                    ('darkorange', 'orange'),
                    ('dodgerblue', 'navy'),
                    ('magenta', 'darkviolet')]

    fig, ax  = plt.subplots(figsize=(9.5, 6))
    dfig, dax  = plt.subplots(figsize=(5, 6))
    slicer = {s: k for s, k in SLICER.items() if s != 'year'}
    alphabet= 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i, coords in enumerate(labelcoords):
        ix, iy = coords
        axtext = alphabet[i+i_base]
        tmax_smean = dep_var.loc[slicer].isel(lat=iy, lon=ix).squeeze().compute()
        tmax_smean.plot(ax=ax, color=col_list[i][0], label='daily mean',
                        lw=.5, alpha=line_alpha)

        if heatwaves is not None:
            hwdays = tmax_smean.where(heatwaves)
            hwdays.plot(ax=ax, color='red', ls='None', marker='.', markersize=2, label='heatwave days')
            total_hw_days = heatwaves.sum().item()
            axtext += f': {total_hw_days} ({round(total_hw_days / (len(tmax_smean)/365))} d/yr)'
        
        ax.text(0.02, 0.93-0.04*i, axtext,
                transform=ax.transAxes, fontsize=12, color=col_list[i][1],
                fontweight='bold')
        dax.text(0.02, 0.93-0.04*i, f'{alphabet[i]}',
                transform=dax.transAxes, fontsize=12, color=col_list[i][1],
                fontweight='bold')

        doy_percentile = calculate_doy_percentile(tmax_smean)
        doy_percentile.plot(ax=dax, color=col_list[i][1], label='95th percentile')

    ax.margins(x=0)
    ax.set_xlabel('year')
    ax.set_ylabel('Tmax [°C]')
    ax.set_title('')

    tseries_lims  = ax.get_xlim(), ax.get_ylim()

    # ax.set_ylim(None, 50)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.savefig(FIGDIR +f'presentation_plots/tseries_{sname_suffix}.png',
                dpi=400, bbox_inches='tight')

    dax.set_xlabel('Day of year')
    dax.margins(0)
    dax.set_ylim(tseries_lims[1])
    dax.set_title('')
    dax.grid(axis='y', linestyle='--', alpha=0.5)
    dfig.savefig(FIGDIR +f'presentation_plots/doy_percentile_{sname_suffix}.png',
                dpi=400, bbox_inches='tight')

tmax_tseries('a', labelcoords=[(54, 50)])
tmax_tseries('ab', labelcoords=[(54, 50), (30, 70)],
             line_alpha=0.5)


#%%
ix = 54; iy = 50
slicer = {s: k for s, k in SLICER.items() if s != 'year'}
tmax_smean = dep_var.loc[slicer].isel(lat=iy, lon=ix).squeeze().compute()
doy_percentile = calculate_doy_percentile(tmax_smean)
doy_percentile_tseries = np.tile(doy_percentile, reps=(len(tmax_smean)//len(doy_percentile)))
heatdays_pctl = (tmax_smean > doy_percentile_tseries)
heatdays = heatdays_pctl & (tmax_smean > HEATDAY_TEMP_THRESH)
heatwaves = remove_short_runs(heatdays, thresh=HEATWAVE_DAY_THRESH)

tmax_tseries('a', labelcoords=[(54, 50)], 
             heatwaves=heatdays_pctl)
tmax_tseries('a', labelcoords=[(54, 50)],
             heatwaves=heatdays)
tmax_tseries('a', labelcoords=[(54, 50)], 
             heatwaves=heatwaves)
ix = 30; iy = 70
slicer = {s: k for s, k in SLICER.items() if s != 'year'}
tmax_smean = dep_var.loc[slicer].isel(lat=iy, lon=ix).squeeze().compute()
doy_percentile = calculate_doy_percentile(tmax_smean)
doy_percentile_tseries = np.tile(doy_percentile, reps=(len(tmax_smean)//len(doy_percentile)))
heatdays_pctl = (tmax_smean > doy_percentile_tseries)
heatdays = heatdays_pctl & (tmax_smean > HEATDAY_TEMP_THRESH)
heatwaves = remove_short_runs(heatdays, thresh=HEATWAVE_DAY_THRESH)
             
tmax_tseries('b', labelcoords=[(46, 75)], 
             heatwaves=heatwaves, col_list=[('darkorange', 'orange')], i_base=1)

#%% heatwaves map
heatwaves = xr.open_zarr(STORAGE_DIR + 'heatdays/heatwaves_annual_sum_zz_2003-2020.zarr', consolidated=True)['heatwave_days']
heatwaves = heatwaves.loc[{s: k for s, k in SLICER.items() if s != 'year'}].sum('year').compute()
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})
heatwaves.plot.imshow(ax=ax, cmap='viridis', vmin=100, vmax=250,
                    cbar_kwargs={'label': 'Heatwave days [2003-2020]', 'orientation': 'horizontal',
                                 'pad': 0.02, 'shrink': 0.5})
ax.set_title('')
fig.savefig(FIGDIR +'presentation_plots/heatwaves_map.png',
            dpi=400, bbox_inches='tight')
#%%
heatwaves = xr.open_zarr(STORAGE_DIR + 'heatdays/heatwaves_annual_sum_zz_2003-2020_gapfilled.zarr', consolidated=True)['heatwave_days']
heatwaves = heatwaves.loc[{s: k for s, k in SLICER.items() if s != 'year'}].sum('year').compute()
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})
heatwaves.plot.imshow(ax=ax, cmap='viridis', vmin=100, vmax=250,
                    cbar_kwargs={'label': 'Heatwave days [2003-2020]', 'orientation': 'horizontal',
                                 'pad': 0.02, 'shrink': 0.5})
ax.set_title('')
fig.savefig(FIGDIR +'presentation_plots/heatwaves_gapfilled_map.png',
            dpi=400, bbox_inches='tight')

#%%
eracc, cc_loss, eratmax = load_ind_dep_data(ind_var_name='cc_era5', dep_var_name='tmax_era5', include_cc_loss=True)
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})
eracc[0].plot.imshow(ax=ax, cmap='Greens', vmin=0, vmax=100,
                    cbar_kwargs={'label': 'Tree cover [%]', 'orientation': 'horizontal',
                                 'pad': 0.02, 'shrink': 0.5})
ax.set_title('')
fig.savefig(FIGDIR +'presentation_plots/eracc_map.png',
            dpi=400, bbox_inches='tight')
#%%
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})
eratmax.groupby('time.year').mean().loc[{'year':2003}].plot.imshow(ax=ax, cmap='magma', vmin=5, vmax=35,
                    cbar_kwargs={'label': 'Tmax [°C]', 'orientation': 'horizontal',
                                 'pad': 0.02, 'shrink': 0.5})
ax.set_title('')
fig.savefig(FIGDIR +'presentation_plots/eratmax_map.png',
            dpi=400, bbox_inches='tight')