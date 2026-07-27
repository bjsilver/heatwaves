#%%
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
from constants import LAT_CHUNK_SIZE, LON_CHUNK_SIZE, FIGDIR
from mwa import _load_single_ind_var
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from ufuncs import _add_gridlines
from dask.diagnostics import ProgressBar


def generate_coarsened_versions(ind_var, coarsen_factor):
    """Generate coarsened variable triplets, with offset versions if coarsen_factor >= 10.

    Returns a list of (ind_var_coarse, driver_ind_var_coarse, dep_var_coarse) tuples.
    For coarsen_factor >= 10, creates offset versions with 5-pixel intervals.
    """
    dss = []

    # First version: standard coarsening
    ind_coarse = ind_var.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary='trim').mean()
    dss.append(ind_coarse)

    # Add offset versions if coarsen_factor >= 10
    if coarsen_factor >= 10:
        num_offsets = (coarsen_factor - 10) // 5 + 1
        for offset_idx in range(num_offsets):
            offset = (offset_idx + 1) * 5

            # Create shifted versions with offset slicing
            ind_shifted = ind_var.loc[
                {
                    'lat': slice(ind_var.lat.values[offset], ind_var.lat.values[-offset]),
                    'lon': slice(ind_var.lon.values[offset], ind_var.lon.values[-offset]),
                }
            ]


            # Coarsen the shifted versions
            ind_shifted_coarse = ind_shifted.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary='trim').mean()
   
            # rechunk for the coarsened versions
            ind_shifted_coarse = ind_shifted_coarse.chunk({'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})

            dss.append(ind_shifted_coarse)

    return dss

def plot_binned_map_with_histogram_colorbar(da, levels, cmap='viridis',
                                            extent=None, fig=None,
                                            map_ax=None, hist_ax=None,
                                            title=''):
    """Plot a 2D field with a histogram-style colorbar showing pixels per bin."""
    values = np.asarray(da.values)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError('DataArray contains no finite values to plot.')

    cmap_obj = plt.get_cmap(cmap, len(levels) - 1)
    norm = BoundaryNorm(levels, cmap_obj.N, clip=True)
    counts, _ = np.histogram(finite_values, bins=levels)
    bin_centers = 0.5 * (levels[:-1] + levels[1:])
    bin_heights = np.diff(levels)

    if fig is None:
        fig = plt.figure(
            figsize=(10, 6),
            constrained_layout=False,
        )

    if map_ax is None or hist_ax is None:
        grid = fig.add_gridspec(1, 2, width_ratios=[1, 0.3], wspace=-.1)
        map_ax = fig.add_subplot(grid[0, 0], projection=ccrs.PlateCarree())
        hist_ax = fig.add_subplot(grid[0, 1])

    image = da.plot(
        ax=map_ax,
        cmap=cmap_obj,
        norm=norm,
        add_colorbar=False,
        transform=ccrs.PlateCarree(),
    )
    map_ax.set_extent([float(da.lon.min()), float(da.lon.max()), float(da.lat.min()), float(da.lat.max())], crs=ccrs.PlateCarree())
    map_ax.set_aspect('equal', adjustable='box')
    map_ax.set_title(title, color='white', y=.92, fontweight='bold')


    hist_ax.barh(
        bin_centers,
        counts,
        height=bin_heights,
        color=cmap_obj(np.arange(len(counts))),
        edgecolor='black',
        linewidth=0.4,
        align='center',
    )
    hist_ax.set_ylim(levels[0], levels[-1])
    hist_ax.set_yticks(levels)
    hist_ax.yaxis.set_label_position('left')
    hist_ax.set_xlabel('pixel count')
    hist_ax.set_ylabel('')
    # make hist log scale
    # hist_ax.set_xscale('log')
    
    if extent is not None:
        map_ax.set_extent(extent, crs=ccrs.PlateCarree())

    map_ax.set_xlabel('')
    map_ax.set_ylabel('')

    _add_gridlines(map_ax, tick_interval=.5, alpha=0.5, color='white')

    return fig, (map_ax, hist_ax), image

def plot_coarsening_visualisation():
   
    LAT_START = -13.6
    LON_START = -56.6
    TEST_AREA = {'lat':slice(LAT_START, LAT_START+2), 'lon':slice(LON_START, LON_START+2)}
    da = _load_single_ind_var('cc_loss')[-1].loc[TEST_AREA]
    EXTENT = [float(da.lon.min()), float(da.lon.max()), float(da.lat.min()), float(da.lat.max())]




    levels = np.arange(0, 110, 10)
    da_coarse = _coarsen_keep_edges(da, 25)

    OFFSET = .1
    TEST_AREA_OFFSET = {'lat':slice(LAT_START - 1, LAT_START + 3), 'lon':slice(LON_START -1 , LON_START + 3)}
    da_offset = _load_single_ind_var('cc_loss')[-1].loc[TEST_AREA_OFFSET]
    offset_slicer = {'lat':slice(da_offset.lat.values[0] + OFFSET, da_offset.lat.values[-1] - OFFSET),
                    'lon':slice(da_offset.lon.values[0] + OFFSET, da_offset.lon.values[-1] - OFFSET)}
    da_offset = da_offset.loc[offset_slicer]
    da_offset_coarse = _coarsen_keep_edges(da_offset, 25)



    fig = plt.figure(figsize=(10, 15), constrained_layout=False)
    grid = fig.add_gridspec(3, 2, width_ratios=[1, 0.3], hspace=0.2, wspace=-0.25)

    plot_binned_map_with_histogram_colorbar(
        da,
        levels,
        fig=fig,
        map_ax=fig.add_subplot(grid[0, 0], projection=ccrs.PlateCarree()),
        hist_ax=fig.add_subplot(grid[0, 1]),
        title='Original',
    )
    plot_binned_map_with_histogram_colorbar(
        da_coarse,
        levels,
        extent=EXTENT,
        fig=fig,
        map_ax=fig.add_subplot(grid[1, 0], projection=ccrs.PlateCarree()),
        hist_ax=fig.add_subplot(grid[1, 1]),
        title='Coarsened',
    )
    plot_binned_map_with_histogram_colorbar(
        da_offset_coarse,
        levels,
        extent=EXTENT,
        fig=fig,
        map_ax=fig.add_subplot(grid[2, 0], projection=ccrs.PlateCarree()),
        hist_ax=fig.add_subplot(grid[2, 1]),
        title='Coarsened with offset',
    )

    fig.savefig(FIGDIR + 'coarsening_visualisation.png',
                dpi=300, bbox_inches='tight')

#%%

res = {}
for coarsen_factor in [None, 2, 5, 10, 10,  25]:
    ind_var = _load_single_ind_var('cc_loss')[-1]
    if coarsen_factor is None:
        
        res[1] = ind_var
    else:
        ind_var_coarse = generate_coarsened_versions(ind_var, coarsen_factor)
        res[coarsen_factor] = ind_var_coarse


    print(coarsen_factor, 'done')

#%%
ds = xr.Dataset()
for coarsen_factor, das in res.items():
    if type(das) is not list:
        das = [das]
    binned_das = []
    for da in das:
        binned = da.groupby_bins(da, np.arange(0, 110, 10)).count()
        binned_das.append(binned)

    summed_binned = sum(binned_das)
    ds[f'{coarsen_factor}km_res'] = summed_binned

with ProgressBar():
    ds = ds.compute(num_workers=16, memory_limit='16GB')

#%%
cmap = plt.get_cmap('Blues', 25)
norm = plt.Normalize(vmin=-20, vmax=30)

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(1, 1, 1)
for data_var in ds.data_vars:
    da = ds[data_var]
    ax.bar(x = np.arange(len(da.values)), height=da.values, width=1,
           color=cmap(norm(int(data_var.split('km')[0]))),
           label=f'{data_var.split("km")[0]} km resolution',
           edgecolor='grey', linewidth=0.5, alpha=0.7)

# make pretty x labels from bins
xlabels = [f'{int(b.left)}-{int(b.right)}' for b in da.cc_loss_bins.values]

ax.set_yscale('log')
ax.margins(x=0)
ax.set_xticks(np.arange(len(xlabels)))
ax.set_xticklabels(xlabels, rotation=0, ha='center')
ax.legend(loc='upper right')
ax.set_xlabel('forest loss bin (%)')
ax.set_ylabel('pixel count (log scale)')
fig.savefig(FIGDIR + 'coarsening_histogram.svg', dpi=300, bbox_inches='tight')