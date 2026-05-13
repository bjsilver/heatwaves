#%%
import xarray as xr
from constants import STORAGE_DIR
from dask.distributed import Client, LocalCluster
import os
import shutil
import dask
from ufuncs import add_metadata_to_zarr

TARGET_CHUNKS = (365, 250, 250)

RECHUNK_JOBS = [
    {
        'input_path': STORAGE_DIR + 'regridded/zhang_zhou_regridded.zarr',
        'output_path': STORAGE_DIR + 'rechunked/zhang_zhou_regridded_rechunked.zarr',
        'var_name': 'Ta_TMAX',
        'consolidated': False,
    },
    {
        'input_path': STORAGE_DIR + 'percentiles/zz_tmax_95pc.zarr',
        'output_path': STORAGE_DIR + 'rechunked/zz_pct_rechunked.zarr',
        'var_name': 'Ta_TMAX_p95',
        'consolidated': True,
    },
]


def cleanup_store(store_path):
    if os.path.exists(store_path):
        print(f"Deleting existing store at {store_path}")
        shutil.rmtree(store_path)


def rechunk_and_save(input_path, output_path, var_name, consolidated=False):
    cleanup_store(output_path)
    
    print(f"Loading {var_name} from {input_path}")
    orig = xr.open_zarr(input_path, consolidated=consolidated)
    orig.encoding = {}
    da = orig[var_name].data

    print(f"Rechunking to {TARGET_CHUNKS}...")
    with dask.config.set({"array.rechunk.method": "p2p", "optimization.fuse.active": False}):
        rechunked = da.rechunk(TARGET_CHUNKS)

    print(f"Writing rechunked data to {output_path}...")
    with LocalCluster(n_workers=64, threads_per_worker=1, memory_limit='8GB') as cluster:
        with Client(cluster) as client:
            print(f"Dask Dashboard available at: {client.dashboard_link}")
            rechunked.to_zarr(output_path, mode='w', consolidated=False, align_chunks=True)

    print(f"Adding metadata to {output_path}...")
    add_metadata_to_zarr(
        output_path,
        var_name=var_name,
        dims=("time", "lat", "lon"),
        consolidate=True,
        orig_ds=orig,
    )

    ds = xr.open_zarr(output_path, consolidated=False)
    print(f"Output dataset:\n{ds}\n")
    return ds


if __name__ == '__main__':
    for job in RECHUNK_JOBS:
        rechunk_and_save(**job)
