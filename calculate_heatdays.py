#%%
# Criteria for heat day
# T > 28.5
# T > 95th Percentile

import xarray as xr
from glob import glob
from dask.diagnostics import ProgressBar
import multiprocessing as mp
from constants import LAT_STEP, LON_STEP
import time

PERCENTILE_PATH = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/percentile_chunks/'
DPATH = '/uolstore/Research/b/b0256/Users/earjba/ZhangandZhou/tropical_regions/'
HEATDAYS_CHUNKS_PATH = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/heatdays_chunks/'



def open_percentile_chunk(percentile_path):
    ds = xr.open_dataset(percentile_path)
    ds = ds.sortby('lat', ascending=False)
    da = ds['Ta_TMAX_95pc']
    return da


def open_tmax_chunk(year, pctl_chunk):
    
    tmax = xr.open_dataset(DPATH+f'SouthAmerica_Ta_SVCMsp_{year}TMAX_daily_tropics30N30S.nc')['Ta_TMAX']
    da = tmax.loc[{'lat':slice(pctl_chunk.lat.max(), pctl_chunk.lat.min()),
                    'lon':slice(pctl_chunk.lon.min(), pctl_chunk.lon.max())}]
    return da



def get_heatdays_year_chunk(year, pctl_chunk):
    
    tmax_chunk = open_tmax_chunk(year, pctl_chunk)
    tmax_chunk = tmax_chunk.load()
    
    heatdays = (tmax_chunk > 28.5) & (tmax_chunk > pctl_chunk)

    # replace DOY with datetime
    heatdays = heatdays.rename({'DOY':'time'})
    datetimes = xr.date_range(start=f'{year}-01-01', end=f'{year}-12-31',
                              freq='D', use_cftime=True, calendar='noleap')
    heatdays.coords['time'] = datetimes

    return heatdays

def process_chunk(percentile_path, YEAR_RANGE, pool):
    start_time = time.time()
    print('processing', f'{percentile_path}')

    pctl_chunk = open_percentile_chunk(percentile_path)
    pctl_chunk = pctl_chunk.load()

    args = [(y, pctl_chunk) for y in YEAR_RANGE]
    heatdays_chunks_year = pool.starmap(get_heatdays_year_chunk, args)

    heatdays_year = xr.concat(heatdays_chunks_year, dim='time')
    spath_fname = percentile_path.split('/')[-1].replace('percentiles', 'heatdays')
    heatdays_year.name = 'heatdays'
    heatdays_year.to_netcdf(HEATDAYS_CHUNKS_PATH+spath_fname)

    end_time = time.time()
    print(f'Finished {percentile_path} in {end_time-start_time:.1f} seconds')


#%%
if __name__ == '__main__':

    percentile_paths = glob(PERCENTILE_PATH+'percentiles_*.nc')

    YEAR_RANGE = range(2006,2018)

    

    with mp.Pool(len(YEAR_RANGE)) as pool:

        for percentile_path in percentile_paths:
            
            process_chunk(percentile_path, YEAR_RANGE, pool)