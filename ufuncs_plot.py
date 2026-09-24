import numpy as np
import cartopy.crs as ccrs

def _add_gridlines(ax, tick_interval, alpha=0.5,
                   color='white'):
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
        color=color,
        alpha=alpha,
        linestyle='--',
    )