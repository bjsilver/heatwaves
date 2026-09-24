#%%
import xarray as xr
from constants import STORAGE_DIR, ZZ_SOURCE_DIR, LAT_CHUNK_SIZE, LON_CHUNK_SIZE, TMAX_RECHUNKED_PATH
import dask
from ufuncs import local_cluster, clip_to_even_chunks
import pickle

TIME_CHUNK_SIZE = 365  # chunk size for time dimension (1 year)
TARGET_CHUNKS = (LAT_CHUNK_SIZE, LON_CHUNK_SIZE)  # (time, lat, lon)
START_YEAR = 2003
END_YEAR = 2020
RESUME_RUN = True  # if True, will skip years that have already been processed and saved to zarr store

# replace DOY with cftime noleap calendar
def preprocess(ds):
    ds = ds.copy()
    # get filename
    filename = ds.encoding.get('source', 'unknown.nc')
    year = int(filename.split('/')[-1].split('_')[3].split('T')[0])
    datetimes = xr.date_range(f"{year}-01-01", f"{year}-12-31",
                               freq='D', calendar='noleap')
    ds = ds.rename({'DOY': 'time'})
    ds = ds.assign_coords(time=datetimes)
    
    return ds


def rechunk_tmax(year):
    tmax = xr.open_dataset(ZZ_SOURCE_DIR + f'SouthAmerica_Ta_SVCMsp_{year}TMAX_daily_tropics30N30S.nc')

    tmax = tmax.drop_vars(['spatial_ref'])

    tmax = preprocess(tmax)
    tmax = clip_to_even_chunks(tmax)

    # rechunk 
    tmax = tmax.chunk({'time': TIME_CHUNK_SIZE, 'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})


    if year == START_YEAR:
        # create new zarr store
        tmax.to_zarr(TMAX_RECHUNKED_PATH+'tmax_rechunked.zarr', mode='w',
                     zarr_format=2, consolidated=True)
    else:
        # append to existing zarr store
        tmax.to_zarr(TMAX_RECHUNKED_PATH+'tmax_rechunked.zarr', mode='a', append_dim='time',
                     zarr_format=2, consolidated=True)

def rechunk_all_tmax(start_year=START_YEAR, end_year=END_YEAR):

    if RESUME_RUN:
        # check which years have already been processed
        try:
            ds = xr.open_zarr(TMAX_RECHUNKED_PATH+'tmax_rechunked.zarr', consolidated=True)
            processed_years = set(ds.time.dt.year.values)
            print(f"Already processed years: {processed_years}")
        except Exception as e:
            print(f"Could not open existing zarr store: {e}")
            processed_years = set()
    else:
        processed_years = set()

    for year in range(start_year, end_year + 1):
        if year in processed_years:
            print(f"Year {year} already processed. Skipping...")
            continue
        print(f"Rechunking year {year}...")
        rechunk_tmax(year)
        print(f"Year {year} done.")


#%%
if __name__ == "__main__":
    with local_cluster(processes=False, n_workers=24, memory_limit='20GB',
                       threads_per_worker=1) as cluster:
        rechunk_all_tmax()

