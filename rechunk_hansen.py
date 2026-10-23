#%%
import rioxarray as rxr
import xarray as xr
import numpy as np
from constants import HANSEN_DIR, LON_CHUNK_SIZE, LAT_CHUNK_SIZE, TMAX_RECHUNKED_PATH
from dask.distributed import Client, LocalCluster
import dask
from ufuncs import get_target_lat_lons, clip_to_even_chunks
import regionmask

TARGET_STORE = HANSEN_DIR + 'hansen_tc_2000_2024_rechunked.zarr'



def open_hansen():
    da1 = rxr.open_rasterio(HANSEN_DIR + 'hansen_tc_2000_2024-0000000000-0000000000.tif',
                            chunks='auto').squeeze()
    da2 = rxr.open_rasterio(HANSEN_DIR + 'hansen_tc_2000_2024-0000004864-0000000000.tif',
                            chunks='auto').squeeze()

    da_1 = xr.concat([da1, da2], dim='y')

    da1 = rxr.open_rasterio(HANSEN_DIR + 'hansen_tc_2000_2024-0000000000-0000004864.tif',
                            chunks='auto').squeeze()
    da2 = rxr.open_rasterio(HANSEN_DIR + 'hansen_tc_2000_2024-0000004864-0000004864.tif',
                            chunks='auto').squeeze()

    da_2 = xr.concat([da1, da2], dim='y')

    da = xr.concat([da_1, da_2], dim='x')


    da.name = 'tree_cover'
    da.attrs['units'] = '%'
    da.attrs['long_name'] = 'Tree Cover'
    da = da.rename({'x': 'lon', 'y': 'lat'})
    da = da.rename({'band': 'time'})
    da = da.assign_coords(time=list(range(2000, 2025)))
    da = da.sortby('lat', ascending=True)
    return da


# mask out non-land areas using the regionmask package
def mask_non_land(da):
    land_mask = regionmask.defined_regions.natural_earth_v5_1_2.land_10.mask(da[0].lon, da[0].lat)
    da = da.where(~land_mask.isnull())
    return da

def clip_to_even_chunks(da):
    n_lon_chunks = len(da.lon) // LON_CHUNK_SIZE
    n_lat_chunks = len(da.lat) // LAT_CHUNK_SIZE

    print(f'{len(da.lon) - n_lon_chunks*LON_CHUNK_SIZE} pixels will be dropped from the lon dimension to make it divisible by {LON_CHUNK_SIZE}')
    print(f'{len(da.lat) - n_lat_chunks*LAT_CHUNK_SIZE} pixels will be dropped from the lat dimension to make it divisible by {LAT_CHUNK_SIZE}')

    da = da.isel(lon=slice(0, n_lon_chunks * LON_CHUNK_SIZE), 
                lat=slice(0, n_lat_chunks * LAT_CHUNK_SIZE))
    return da

def rechunk_hansen(da):
 

    with dask.config.set({"array.rechunk.method": "p2p", "optimization.fuse.active": False}):
        rechunked = da.chunk({'time': 1, 
                                'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})

    return rechunked


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

    hansen = clip_to_even_chunks(hansen)
    target_lats, target_lons = get_target_lat_lons()

    # assert that the lat and lon coords of hansen match those of tmax
    assert np.allclose(hansen.lat.values, target_lats), "Latitudes of Hansen and TMAX do not match"
    assert np.allclose(hansen.lon.values, target_lons), "Longitudes of Hansen and TMAX do not match"
    print("Latitudes and longitudes of Hansen and TMAX match.")
    
    hansen.encoding = {}

    with LocalCluster(processes=False, n_workers=4, threads_per_worker=8, memory_limit='128GB') as cluster:
            with Client(cluster) as client:
                print(f"Dask Dashboard available at: {client.dashboard_link}")
                rechunked = rechunk_hansen(hansen)
                rechunked = xr.Dataset({'tree_cover': rechunked})
                rechunked.to_zarr(TARGET_STORE, mode='w', consolidated=True)


    ds = xr.open_zarr(TARGET_STORE, consolidated=True)
    print(ds)

