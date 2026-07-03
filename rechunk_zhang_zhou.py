#%%
import xarray as xr
from constants import STORAGE_DIR
from dask.distributed import Client, LocalCluster
import os
import shutil
import dask
from ufuncs import add_metadata_to_zarr
from constants import LAT_CHUNK_SIZE, LON_CHUNK_SIZE

TARGET_CHUNKS = (365, LAT_CHUNK_SIZE, LON_CHUNK_SIZE)
FILLVALUE = -9999

RECHUNK_JOBS = [
    {
        'input_path': STORAGE_DIR + 'regridded/zhang_zhou_regridded.zarr',
        'output_path': STORAGE_DIR + 'rechunked/zhang_zhou_regridded_rechunked.zarr',
        'var_name': 'Ta_TMAX',
        'dtype': 'float32',
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
    ds = xr.open_zarr(input_path, consolidated=consolidated)
    # temp fix
    if '__xarray_dataarray_variable__' in ds:
        ds = ds.rename({'__xarray_dataarray_variable__': var_name})
    da = ds[var_name]
    da.encoding = {}

    # if dtype is int replace NaN with FillValue
    if 'int' in dtype or 'uint' in dtype:
        da = da.fillna(FILLVALUE)
        da.attrs['_FillValue'] = FILLVALUE
    da = da.astype(dtype)

    da.attrs['units'] = units

    da = da.astype(dtype)
    dask_array = da.data
  
    print(f"Rechunking to {TARGET_CHUNKS}...")
    with dask.config.set({"array.rechunk.method": "p2p", "optimization.fuse.active": False}):
        rechunked = dask_array.rechunk(TARGET_CHUNKS)

    print(f"Writing rechunked data to {output_path}...")
    rechunked.to_zarr(output_path, mode='w', consolidated=False, align_chunks=True)

    print(f"Adding metadata to {output_path}...")
    add_metadata_to_zarr(
        output_path,
        var_name=var_name,
        dims=dims,
        consolidate=True,
        orig_ds=ds,
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


