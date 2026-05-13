#%%
import xarray as xr
from constants import STORAGE_DIR
from dask.distributed import Client, LocalCluster
import os
import shutil
import dask
from ufuncs import add_metadata_to_zarr


dpath = STORAGE_DIR + 'regridded/era5_skt_p95_2003-2020_regridded.zarr'

target_store = STORAGE_DIR + 'rechunked/era5_skt_p95_2003-2020_rechunked.zarr'
target_chunks = (365, 250, 250)

def cleanup_stores():
    if os.path.exists(target_store):

        print(f"Deleting existing target store at {target_store}")

        shutil.rmtree(target_store)

#%%
if __name__ == '__main__':

    orig = xr.open_zarr(dpath, consolidated=False)
    orig.encoding = {}

    da = orig['heatwaves'].data

    with dask.config.set({"array.rechunk.method": "p2p", "optimization.fuse.active": False}):
        rechunked = da.rechunk(target_chunks)

    with LocalCluster(n_workers=16, threads_per_worker=1, memory_limit='8GB') as cluster:
        with Client(cluster) as client:
            print(f"Dask Dashboard available at: {client.dashboard_link}")
            rechunked.to_zarr(target_store, mode='w', consolidated=False, align_chunks=True)

    
    add_metadata_to_zarr(
        target_store,
        var_name="Ta_TMAX",
        dims=("time", "lat", "lon"),
        consolidate=True,
        orig_ds=orig,
    )


    ds = xr.open_zarr(target_store, consolidated=False)
    print(ds)
