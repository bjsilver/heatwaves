#%%
import xarray as xr
from constants import STORAGE_DIR
from dask.distributed import Client, LocalCluster
import os
import shutil
import dask
from ufuncs import add_metadata_to_zarr

TARGET_CHUNKS = (365, 250, 250)
FILLVALUE = -9999

RECHUNK_JOBS = [
    {
        'input_path': STORAGE_DIR + 'regridded/zhang_zhou_regridded.zarr',
        'output_path': STORAGE_DIR + 'rechunked/zhang_zhou_regridded_rechunked.zarr',
        'var_name': 'Ta_TMAX',
        'dtype': 'uint16',
        'dims': ('time', 'lat', 'lon'),
        'consolidated': False,
        'units':'tenths of degC'
    },
    # {
    #     'input_path': STORAGE_DIR + 'percentiles/zz_tmax_95pc.zarr',
    #     'output_path': STORAGE_DIR + 'rechunked/zz_pct_rechunked.zarr',
    #     'var_name': 'Ta_TMAX_p95',
    #     'dims': ('doy', 'lat', 'lon'),
    #     'consolidated': True,
    # },
    # {
    #     'input_path': STORAGE_DIR + 'percentiles/zz_tmax_95pc_smooth.zarr',
    #     'output_path': STORAGE_DIR + 'rechunked/zz_pct_smooth_rechunked.zarr',
    #     'var_name': 'Ta_TMAX_p95_smooth',
    #     'dims': ('doy', 'lat', 'lon'),
    #     'consolidated': True,
    # },
]


def cleanup_store(store_path):
    if os.path.exists(store_path):
        print(f"Deleting existing store at {store_path}")
        shutil.rmtree(store_path)


def rechunk_and_save(input_path, output_path, var_name, dtype, dims, units, consolidated=False):
    cleanup_store(output_path)
    
    print(f"Loading {var_name} from {input_path}")
    orig = xr.open_zarr(input_path, consolidated=consolidated)
    # if dtype is int replace NaN with FillValue
    if dtype in ['int16', 'int32', 'int64', 'uint16', 'uint32', 'uint64']:
        orig = orig.fillna(FILLVALUE)
        orig.attrs['_FillValue'] = FILLVALUE
    orig = orig.astype({var_name: dtype})

    orig.attrs['units'] = units


    orig.encoding = {}
    da = orig[var_name].data

    print(f"Rechunking to {TARGET_CHUNKS}...")
    with dask.config.set({"array.rechunk.method": "p2p", "optimization.fuse.active": False}):
        rechunked = da.rechunk(TARGET_CHUNKS)

    print(f"Writing rechunked data to {output_path}...")
    rechunked.to_zarr(output_path, mode='w', consolidated=False, align_chunks=True)

    print(f"Adding metadata to {output_path}...")
    add_metadata_to_zarr(
        output_path,
        var_name=var_name,
        dims=dims,
        consolidate=True,
        orig_ds=orig,
    )

    ds = xr.open_zarr(output_path, consolidated=False)
    print(f"Output dataset:\n{ds}\n")
    return ds

#%%
if __name__ == '__main__':
    with LocalCluster(n_workers=64, threads_per_worker=1, memory_limit='8GB') as cluster:
        with Client(cluster) as client:
            print(f"Dask Dashboard available at: {client.dashboard_link}")
            for job in RECHUNK_JOBS:
                rechunk_and_save(**job)


