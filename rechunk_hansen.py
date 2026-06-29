#%%
import rioxarray as rxr
import xarray as xr
from constants import HANSEN_DIR
from dask.distributed import Client, LocalCluster
import os
import shutil
import dask
from ufuncs import add_metadata_to_zarr
import regionmask

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


# mask out non-land areas using the regionmask package
def mask_non_land(da):
    land_mask = regionmask.defined_regions.natural_earth_v5_1_2.land_10.mask(da[0].lon, da[0].lat)
    da = da.where(~land_mask.isnull())
    return da

#%%


if __name__ == '__main__':
    hansen = open_hansen()

    # change lat lon coords to float32 and round to 2dp
    # hansen = hansen.assign_coords(lat=hansen.lat.astype('float32').round(2), lon=hansen.lon.astype('float32').round(2))
    hansen = hansen.drop_vars('spatial_ref')
    
    hansen = hansen.round(1)
    # hansen = hansen.astype('float32')
    hansen = hansen.fillna(0)
    hansen = mask_non_land(hansen)

    hansen.encoding = {}

    # da = hansen.chunk().data
    with dask.config.set({"array.rechunk.method": "p2p", "optimization.fuse.active": False}):
        rechunked = hansen.chunk({'time': 1, 'lat': 250, 'lon': 250})

    with LocalCluster(n_workers=4, threads_per_worker=8, memory_limit='128GB') as cluster:
            with Client(cluster) as client:
                print(f"Dask Dashboard available at: {client.dashboard_link}")
                rechunked.to_zarr(TARGET_STORE, mode='w')

        
    add_metadata_to_zarr(
        TARGET_STORE,
        var_name="tree_cover",
        dims=("time", "lat", "lon"),
        consolidate=True,
        orig_ds=hansen,
    )

    ds = xr.open_zarr(TARGET_STORE, consolidated=False)
    print(ds)
