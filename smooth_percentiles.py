#%% add gaussian smoothing to percentiles
import xarray as xr
from constants import STORAGE_DIR, FIGDIR
import matplotlib.pyplot as plt
import contextily as ctx
import cartopy.crs as ccrs
import numpy as np
from scipy.ndimage import gaussian_filter
from dask.diagnostics import ProgressBar
from ufuncs import local_cluster



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
                              vmin=None, vmax=None, sfigname=None):

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
    if save_fig and sfigname is not None:
        fig.savefig(FIGDIR+'percentiles/'+sfigname+'.png', dpi=300, bbox_inches='tight')


def _gaussian_nan_safe(block, sigma):
    block = np.asarray(block, dtype=np.float64)
    valid = np.isfinite(block)

    filled = np.where(valid, block, 0.0)
    weights = valid.astype(np.float64)

    smoothed = gaussian_filter(filled, sigma=sigma, mode="nearest")
    weight_sum = gaussian_filter(weights, sigma=sigma, mode="nearest")

    out = np.full_like(smoothed, np.nan, dtype=np.float64)
    np.divide(smoothed, weight_sum, out=out, where=weight_sum > 0)
    return out


def smooth_latlon_chunked(da, sigma=1.0):
    # Axis indices for named dims
    lat_ax = da.get_axis_num("lat")
    lon_ax = da.get_axis_num("lon")

    # Only smooth in lat/lon; keep other dims untouched
    sigma_all = [0.0] * da.ndim
    sigma_all[lat_ax] = sigma
    sigma_all[lon_ax] = sigma

    # Halo size per chunk (rule of thumb: ~4*sigma)
    halo = int(np.ceil(4 * sigma))
    depth_all = [0] * da.ndim
    depth_all[lat_ax] = halo
    depth_all[lon_ax] = halo

    sigma_all = tuple(sigma_all)

    out = da.data.map_overlap(
        _gaussian_nan_safe,
        depth=tuple(depth_all),
        boundary="nearest",
        trim=True,
        dtype=np.float64,
        sigma=sigma_all,
    )

    return xr.DataArray(
        out.astype(da.dtype, copy=False),
        coords=da.coords,
        dims=da.dims,
        attrs=da.attrs,
        name=da.name,
    )

def plot_smoothing_examples(pctls, pctls_smooth, extent, tick_interval=.25):
    plot_satellite_comparison(
        pctls,
        extent,
        tick_interval=tick_interval,
        save_fig=True,
        sfigname='percentile_satellite_unsmoothed',
        vmin=35.6,
        vmax=36.8,
    )
    plot_satellite_comparison(
        pctls_smooth,
        extent,
        tick_interval=tick_interval,
        save_fig=True,
        sfigname='percentile_satellite_smoothed',
        vmin=35.6,
        vmax=36.8,
    )

def main(plot_examples=True):
 
    with local_cluster(n_workers=10, threads_per_worker=4, memory_limit='50GB',
                       suppress_dask_warnings=False, use_dask_dashboard=True) as client:
        # area to test: [2600:2700, 2800:2900]
        pctls = xr.open_zarr(STORAGE_DIR + 'percentiles/zz_tmax_95pc.zarr', consolidated=True)['Ta_TMAX_p95']

        pctls_smooth = smooth_latlon_chunked(pctls, sigma=30.0)
        # pctls_smooth = pctls_smooth.interp(lat=pctls.lat, lon=pctls.lon, method='nearest')

        out_zarr_smooth = STORAGE_DIR + 'percentiles/zz_tmax_95pc_smooth.zarr'
        print(f"Computing and saving smoothed percentiles to {out_zarr_smooth}...")
        ds_out = pctls_smooth.to_dataset(name='Ta_TMAX_p95_smooth')
        ds_out.to_zarr(out_zarr_smooth, mode='w', consolidated=True)

    if plot_examples:
        extent = [-4, -3, -54, -53]
        plot_smoothing_examples(pctls, pctls_smooth, extent, tick_interval=.25)

#%%
if __name__ == '__main__':
    main(plot_examples=False)

#%%
pctl_smoothed = xr.open_zarr(STORAGE_DIR + 'percentiles/zz_tmax_95pc_smooth.zarr', consolidated=True)['Ta_TMAX_p95_smooth']