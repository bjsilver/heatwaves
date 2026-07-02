#%%
import xarray as xr
import numpy as np
import os
from constants import HANSEN_DIR, FIGDIR
from ufuncs import local_cluster




def compute_age_block(loss_block):
    """
    loss_block is a DataArray with dimensions
    (time, lat, lon)
    """

    loss = loss_block.values      # NumPy array
    age = np.zeros_like(loss)

    # First year
    age[0] = np.where(loss[0] > 0, 1.0, 0.0)

    # Remaining years
    for t in range(1, loss.shape[0]):

        L_prev = loss[t-1]
        L_curr = loss[t]

        new_loss = L_curr - L_prev

        mask = L_curr > 0

        age[t][mask] = (
            L_prev[mask] * (age[t-1][mask] + 1)
            + new_loss[mask]
        ) / L_curr[mask]

    return xr.DataArray(
        age,
        coords=loss_block.coords,
        dims=loss_block.dims,
        attrs=loss_block.attrs
    )
#%%
if __name__ == "__main__":
    with local_cluster(n_workers=4, threads_per_worker=1, memory_limit="40GB") as cluster:
        

        ds = xr.open_zarr(os.path.join(HANSEN_DIR, 'hansen_tc_2000_2024_rechunked.zarr'))
        tc = ds['tree_cover']
        loss = -(tc - tc[0])
        loss = loss.chunk({'time': -1})
        template = xr.zeros_like(loss)

        loss_age = xr.map_blocks(
            compute_age_block,
            loss,
            template=template,
        )
        loss_age.name = 'tc_loss_age'
        loss.name = 'tc_loss'
        loss.to_zarr(os.path.join(HANSEN_DIR, 'hansen_tc_2000_2024_loss.zarr'), mode='w', consolidated=True)
        loss_age.to_zarr(os.path.join(HANSEN_DIR, 'hansen_tc_2000_2024_loss_age.zarr'), mode='w', consolidated=True)

#%% make animation gif of loss age over time
import matplotlib.pyplot as plt
import imageio
westlimit=-69.7; southlimit=-13.7; eastlimit=-46.4; northlimit=5.6
RESULTS_BBOX = {'lat':slice(southlimit, northlimit), 'lon':slice(westlimit, eastlimit)}

loss_age = xr.open_zarr(os.path.join(HANSEN_DIR, 'hansen_tc_2000_2024_loss_age.zarr'))['tree_cover']
loss_age = loss_age.loc[RESULTS_BBOX]

for year in range(2000, 2025):
    print(f'Creating plot for year {year}')
    fig, ax = plt.subplots(figsize=(10, 5))
    loss_age.sel(time=year).plot.imshow(ax=ax, cmap='viridis_r',
                                        vmin=0, vmax=24, cbar_kwargs={'label': 'Loss Age (years)',
                                                                      'extend': 'neither'})
    ax.set_title(f'Loss Age in {year}')
    fig.savefig(FIGDIR+f'hansen/frames/loss_age_{year}.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

#%%
# make the gif
with imageio.get_writer(FIGDIR+'hansen/loss_age_animation.gif', mode='I', duration=300,
                        loop=0) as writer:
    for year in range(2000, 2025):
        image = imageio.imread(FIGDIR+f'hansen/frames/loss_age_{year}.png')
        writer.append_data(image)


    