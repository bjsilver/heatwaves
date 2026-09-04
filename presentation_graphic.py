#%%
import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import contextily as ctx
from constants import FIGDIR
import xarray as xr
import matplotlib.ticker as mticker
from glob import glob

# Target output settings
TARGET_WIDTH, TARGET_HEIGHT, DPI = 1920, 1080, 100

def render_scene(scene_cfg, slicer):
    """Renders a single scene to a 1080p RGB NumPy array."""
    fig = plt.figure(figsize=(TARGET_WIDTH / DPI, TARGET_HEIGHT / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
    
    ax.set_extent(
        [slicer['lon'].start, slicer['lon'].stop, slicer['lat'].start, slicer['lat'].stop],
        crs=ccrs.PlateCarree()
    )
    ax.set_aspect('auto')

    # 1. Base map layer (optional)
    if scene_cfg.get('basemap', False):
        ctx.add_basemap(
            ax, crs="EPSG:4326", 
            source=scene_cfg.get('basemap_source', ctx.providers.Esri.WorldImagery), 
            attribution='', zoom=scene_cfg.get('zoom', 10)
        )

    # 2. Data layers (optional list of dicts to allow overlaying multiple layers)
    for layer in scene_cfg.get('layers', []):
        layer['data'].plot.imshow(
            ax=ax, 
            cmap=layer.get('cmap', 'viridis'), 
            vmin=layer.get('vmin'), 
            vmax=layer.get('vmax'), 
            add_colorbar=layer.get('add_colorbar', False), 
            transform=ccrs.PlateCarree()
        )

    ax.set_axis_off()
    fig.canvas.draw()
    
    rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return rgb


def generate_wipe_frame(old_rgb, new_rgb, frame_idx, n_frames, fade_width=0.6):
    """Generates a single West -> East sweep transition frame."""
    _, width, _ = old_rgb.shape
    x = np.linspace(0, 1, width, dtype=np.float32)

    t_linear = frame_idx / (n_frames - 1)
    t_smooth = t_linear * t_linear * (3.0 - 2.0 * t_linear)
    t = -fade_width + t_smooth * (1.0 + 2 * fade_width)

    alpha = np.clip((t + fade_width - x) / (2.0 * fade_width), 0.0, 1.0)[None, :, None]
    frame = old_rgb.astype(np.float32) * (1.0 - alpha) + new_rgb.astype(np.float32) * alpha
    return frame.astype(np.uint8)


def generate_scene_transition_video(
    scenes_config, sequence, slicer, output_path, 
    fps=25, pause_seconds=3, wipe_seconds=2, fade_width=0.6
):
    """Renders scenes and compiles wipe transitions into an MP4 video."""
    print("Rendering scene frames...")
    rendered_maps = {name: render_scene(cfg, slicer) for name, cfg in scenes_config.items()}

    pause_frames = int(pause_seconds * fps)
    wipe_frames = int(wipe_seconds * fps)

    print(f"Encoding video to: {output_path}")
    with imageio.get_writer(
        output_path, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1,
        ffmpeg_params=[
            "-crf", "12", "-preset", "slow", "-profile:v", "high",
            "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
        ]
    ) as writer:
        for old_key, new_key in sequence:
            old_map, new_map = rendered_maps[old_key], rendered_maps[new_key]
            
            # Static hold
            for _ in range(pause_frames):
                writer.append_data(old_map)
                
            # Transition
            for idx in range(wipe_frames):
                writer.append_data(generate_wipe_frame(old_map, new_map, idx, wipe_frames, fade_width))

    print("Video generation complete.")


# ============================================================
# Usage Example
# ============================================================
#%%
if __name__ == "__main__":
    from mwa import load_ind_dep_data

    SLICER = {'lat': slice(-15, -8), 'lon': slice(-54 - 12.44444444, -54), 'year': 2020}

    # Data loading and preprocessing outside
    _, _, dep_var = load_ind_dep_data(ind_var_name='cc', dep_var_name='tmax', include_cc_loss=True)
    dep_var_sliced = dep_var[-730:].loc[{'lat': SLICER['lat'], 'lon': SLICER['lon']}].mean('time').compute().fillna(0)

    scenes = {
        'satellite': {
            'basemap': True
        },
        'dep': {
            'basemap': False,
            'layers': [
                {'data': dep_var_sliced, 'cmap': 'magma', 'vmin': 31, 'vmax': 35.3}
            ]
        }
    }

    sequence = [
        ('satellite', 'dep'),
        ('dep', 'satellite')
    ]

    # 3. Generate video
    output_mp4 = os.path.join(FIGDIR, "rolling_transition.mp4")
    generate_scene_transition_video(scenes, sequence, SLICER, output_mp4)
#%%

    dep_coarse = dep_var_sliced.coarsen(lat=25, lon=25, boundary='pad').mean()
    dep_upsamp = dep_coarse.interp_like(dep_var_sliced, method='nearest')

    scenes = {
        'dep': {
            'basemap': False,
            'layers': [
                {'data': dep_var_sliced, 'cmap': 'magma', 'vmin': 31, 'vmax': 35.3}
            ]
        },
        'dep_upsamp': {
            'basemap': False,
            'layers': [
                {'data': dep_upsamp, 'cmap': 'magma', 'vmin': 31, 'vmax': 35.3}
            ]
        }
    }

    sequence = [
        ('dep', 'dep_upsamp'),
        ('dep_upsamp', 'dep')
    ]

    output_mp4 = os.path.join(FIGDIR, "coarsening.mp4")
    generate_scene_transition_video(scenes, sequence, SLICER, output_mp4)
#%% 

def _add_colorbar(ax, im, label):
    cax = ax.inset_axes([0.1, -0.1, 0.8, 0.045])
    fig.colorbar(im, cax=cax, orientation='horizontal', label=label)

def _format_ax(ax, isliced, res):
    lons = isliced.lon.values
    lats = isliced.lat.values
    gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=False,
    linewidth=0.8,
    linestyle='--',
    alpha=0.7,
    )
    gl.xlocator = mticker.FixedLocator(lons-res/2)
    gl.ylocator = mticker.FixedLocator(lats-res/2)
    ax.plot([lons[1]-res/2, lons[1]-res/2, lons[3]+res/2, lons[3]+res/2, lons[1]-res/2],
            [lats[1]-res/2, lats[3]+res/2, lats[3]+res/2, lats[1]-res/2, lats[1]-res/2], 
            color='black', linewidth=1.5, transform=ccrs.PlateCarree())


                  
SLICER = {'lat': slice(-11.5, -11), 'lon': slice(-59.5, -59), 'year': 2020}
ind_var, cc_loss, dep_var = load_ind_dep_data(ind_var_name='cc', dep_var_name='tmax', include_cc_loss=True)

dep_var = dep_var[-465:].loc[{'lat': SLICER['lat'], 'lon': SLICER['lon']}].mean('time').compute()

ind_var = ind_var.loc[SLICER].compute().drop_vars('year')
cc_loss = cc_loss.loc[SLICER].compute().drop_vars('year')

ds = xr.Dataset({'dep_var': dep_var, 'ind_var': ind_var, 'cc_loss': cc_loss})


i = 1
j = -1

lstart = 26
islicer = {'lat':slice(lstart,lstart+5), 'lon':slice(32,37)}
isliced = ds['dep_var'].isel(islicer)
res = 0.01
def _add_mwa(da, ax, i, j, round_decimals=1, show_diff=True,
             show_x90=False, cc_diff=None):
    lats = da.lat.values
    lons = da.lon.values
    cvalue = da.isel({'lat': 2, 'lon': 2}).item()
    ax.text(lons[2], lats[2], s=round(cvalue, round_decimals), ha='center', va='center',
            path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=3, foreground='white')],
            transform=ccrs.PlateCarree())
    if show_diff:
        nvalue = da.isel({'lat': 2+j, 'lon': 2+i}).item()
        ax.text(lons[2+i], lats[2+j], s=round(nvalue, round_decimals), ha='center', va='center',
                path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=3, foreground='white')],
                transform=ccrs.PlateCarree())
        diff = nvalue-cvalue
        ax.text(lons[2], lats[4], s=f'Δ={round(diff, round_decimals)}', ha='center', va='center',
                path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=3, foreground='white')],
                transform=ccrs.PlateCarree())
        return diff
    if show_x90:
        ax.text(lons[2], lats[4], s=f'90% cc diff: {round(cc_diff*.9, round_decimals)}',
                ha='center', va='center',
                path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=3, foreground='white')],
                                transform=ccrs.PlateCarree())
        if cc_diff*.9 <= cvalue:
            threshtext = '✔ at least 90% of\ncc diff due to cc loss'
            threshpass = True
        else:
            threshtext = '✘ at least 90% of\ncc diff due to cc loss'
            threshpass = False
        ax.text(lons[2], lats[0], s=threshtext,
                        ha='center', va='center',
                        path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=3, foreground='white')],
                                        transform=ccrs.PlateCarree())
        return threshpass


    
##%% standalone fig
fig, axes = plt.subplots(2,2,figsize=(8,8), dpi=400,
                         subplot_kw={'projection': ccrs.PlateCarree()})
minlat, maxlat = ds['dep_var'].isel(islicer).lat[[0, -1]].values
minlon, maxlon = ds['dep_var'].isel(islicer).lon[[0, -1]].values

##### canopy cover axis #####
im = ds['ind_var'].isel(islicer).plot.imshow(ax=axes[0,0], cmap='Greens', 
                                             vmin=0, vmax=100, add_colorbar=False)
_add_colorbar(axes[0,0], im, 'Canopy Cover (%)')
_format_ax(axes[0,0], isliced, res)
diff = _add_mwa(ds['ind_var'].isel(islicer), axes[0,0], i, j)

##### canopy loss axis #####
im = ds['cc_loss'].isel(islicer).plot.imshow(ax=axes[1,0], cmap='Reds', 
                                             vmin=0, vmax=100, add_colorbar=False)
_add_colorbar(axes[1,0], im, 'Canopy Loss (%)')
_format_ax(axes[1,0], isliced, res)
_add_mwa(ds['cc_loss'].isel(islicer), axes[1,0], i, j, show_diff=False,
         show_x90=True, cc_diff=diff)


##### dependent variable axis #####
im = ds['dep_var'].isel(islicer).plot.imshow(ax=axes[1,1], cmap='magma', 
                                             vmin=34.6, vmax=34.9, add_colorbar=False)
_add_colorbar(axes[1,1], im, 'Tmax (°C)')
_format_ax(axes[1,1], isliced, res)
_add_mwa(ds['dep_var'].isel(islicer), axes[1,1], i, j, round_decimals=2)

##### satellite imagery axis #####
sat_ax = axes[0,1]
sat_ax.set_extent([minlon-res/2, maxlon+res/2, minlat-res/2, maxlat+res/2], crs=ccrs.PlateCarree())
ctx.add_basemap(sat_ax, crs="EPSG:4326", source=ctx.providers.Esri.WorldImagery, attribution='')
_format_ax(sat_ax, isliced, res)


fig.tight_layout()
fig.subplots_adjust(hspace=0.3)


#%%

                  
SLICER = {'lat': slice(-11.5, -11), 'lon': slice(-59.5, -57), 'year': 2020}
ind_var, cc_loss, dep_var = load_ind_dep_data(ind_var_name='cc', dep_var_name='tmax', include_cc_loss=True)

dep_var = dep_var[-465:].loc[{'lat': SLICER['lat'], 'lon': SLICER['lon']}].mean('time').compute()

ind_var = ind_var.loc[SLICER].compute().drop_vars('year')
cc_loss = cc_loss.loc[SLICER].compute().drop_vars('year')

ds = xr.Dataset({'dep_var': dep_var, 'ind_var': ind_var, 'cc_loss': cc_loss})


xpoints = []
ypoints = []


def plot_lonstart(lonstart, xpoints, ypoints, dpi=400, sfig=False):

    islicer = {'lat':slice(26,31), 'lon':slice(lonstart, lonstart+5)}
    isliced = ds['ind_var'].isel(islicer)

    # search for highest neightbour
    best_diff = -np.inf
    best_ij = (None, None)
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i==0) & (j==0):
                continue
            cvalue = isliced.isel({'lat': 2, 'lon': 2}).item()
            nvalue = isliced.isel({'lat': 2+j, 'lon': 2+i}).item()
            diff = nvalue - cvalue
            if diff > best_diff:
                best_diff = diff
                best_ij = (i, j)


    i, j = best_ij

    fig, axes = plt.subplots(2,2,figsize=(8,8), dpi=dpi,
                            subplot_kw={'projection': ccrs.PlateCarree()})
    minlat, maxlat = ds['dep_var'].isel(islicer).lat[[0, -1]].values
    minlon, maxlon = ds['dep_var'].isel(islicer).lon[[0, -1]].values

    ##### canopy cover axis #####
    im = ds['ind_var'].isel(islicer).plot.imshow(ax=axes[0,0], cmap='Greens', 
                                                vmin=0, vmax=100, add_colorbar=False)
    _add_colorbar(axes[0,0], im, 'Canopy Cover (%)')
    _format_ax(axes[0,0], isliced, res)
    cc_diff = _add_mwa(ds['ind_var'].isel(islicer), axes[0,0], i, j)

    ##### canopy loss axis #####
    im = ds['cc_loss'].isel(islicer).plot.imshow(ax=axes[1,0], cmap='Reds', 
                                                vmin=0, vmax=100, add_colorbar=False)
    _add_colorbar(axes[1,0], im, 'Canopy Loss (%)')
    _format_ax(axes[1,0], isliced, res)
    threshpass = _add_mwa(ds['cc_loss'].isel(islicer), axes[1,0], i, j, show_diff=False,
            show_x90=True, cc_diff=cc_diff)


    ##### dependent variable axis #####
    im = ds['dep_var'].isel(islicer).plot.imshow(ax=axes[1,1], cmap='magma', 
                                                vmin=34.6, vmax=34.9, add_colorbar=False)
    _add_colorbar(axes[1,1], im, 'Tmax (°C)')
    _format_ax(axes[1,1], isliced, res)
    dep_diff = _add_mwa(ds['dep_var'].isel(islicer), axes[1,1], i, j, round_decimals=2)

    ##### satellite imagery axis #####
    sat_ax = axes[0,1]
    sat_ax.set_extent([minlon-res/2, maxlon+res/2, minlat-res/2, maxlat+res/2], crs=ccrs.PlateCarree())
    ctx.add_basemap(sat_ax, crs="EPSG:4326", source=ctx.providers.Esri.WorldImagery, attribution='')
    _format_ax(sat_ax, isliced, res)

    if (threshpass) and (cc_diff > 0):
        xpoints.append(cc_diff)
        ypoints.append(-dep_diff)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3)

    # add graph axes to the side
    gax = fig.add_axes([1.1, .3, .6, .4])
    gax.plot(xpoints, ypoints, ls='', marker='.', color='red')
    gax.set_xlim(0, 100)
    gax.set_ylim(-0.1, .4)
    gax.axhline(0, color='black')
    gax.grid(alpha=.7, ls='--')
    gax.set_xlabel('canopy cover difference')
    gax.set_ylabel('Tmax response')

    if sfig:
        fig.savefig(f"/tmp/gif_frames/{str(lonstart).zfill(3)}.png",
                    bbox_inches='tight')
        plt.close()

    return xpoints, ypoints
#%%
for lonstart in range(32, 37):
    xpoints, ypoints =  plot_lonstart(lonstart, xpoints, ypoints)

#%%
for lonstart in range(37, 137):
    print(lonstart)
    xpoints, ypoints = plot_lonstart(lonstart, xpoints, ypoints, dpi=200, sfig=True)
#%%
frame_paths = sorted(glob('/tmp/gif_frames/*.png'))
images = [imageio.imread(p) for p in frame_paths]
imageio.mimsave(
        FIGDIR+'mwa.gif',
        images,
        fps=15,
        plugin="pillow",
        quantizer="neuquant",  # High-quality color quantization
        dither=1,              # Reduces color banding across heatmaps/maps
        subrectangles=True     # Optimizes file size by only saving modified pixels
    )