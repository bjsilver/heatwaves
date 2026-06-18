#%%
import rioxarray as rxr
import xarray as xr
from constants import HANSEN_DIR
from dask.distributed import Client, LocalCluster
import os
import shutil
import dask
from ufuncs import add_metadata_to_zarr

TARGET_STORE = HANSEN_DIR + 'hansen_tc_2000_2024_rechunked.zarr'

def open_hansen():
    da1 = rxr.open_rasterio(HANSEN_DIR + 'hansen_tc_2000_2024-0000000000-0000000000.tif',
                            chunks='auto').squeeze()
    da2 = rxr.open_rasterio(HANSEN_DIR + 'hansen_tc_2000_2024-0000004864-0000000000.tif',
                            chunks='auto').squeeze()

    da = xr.concat([da1, da2], dim='y')

    da.name = 'tree_cover'
    da.attrs['units'] = '%'
    da.attrs['long_name'] = 'Tree Cover'
    da = da.rename({'x': 'lon', 'y': 'lat'})
    da = da.rename({'band': 'time'})
    da = da.assign_coords(time=list(range(2000, 2025)))
    da = da.sortby('lat', ascending=True)
    da = da[:, :-1, :-1]
    return da

#%%

if __name__ == '__main__':
    hansen = open_hansen()

    # change lat lon coords to float32 and round to 2dp
    hansen = hansen.assign_coords(lat=hansen.lat.astype('float32').round(2), lon=hansen.lon.astype('float32').round(2))
    hansen = hansen.drop_vars('spatial_ref')
    
    # round to 0dp and convert to int16 to save space
    hansen = hansen.round(0)
    hansen = hansen.astype('uint8')

    hansen.encoding = {}

    da = hansen.chunk().data
    with dask.config.set({"array.rechunk.method": "p2p", "optimization.fuse.active": False}):
        rechunked = da.rechunk((1, 250, 250))

    with LocalCluster(n_workers=4, threads_per_worker=8, memory_limit='128GB') as cluster:
            with Client(cluster) as client:
                print(f"Dask Dashboard available at: {client.dashboard_link}")
                rechunked.to_zarr(TARGET_STORE, mode='w', consolidated=False, align_chunks=True)

        
    add_metadata_to_zarr(
        TARGET_STORE,
        var_name="tree_cover",
        dims=("time", "lat", "lon"),
        consolidate=True,
        orig_ds=hansen,
    )

    ds = xr.open_zarr(TARGET_STORE, consolidated=False)
    print(ds)