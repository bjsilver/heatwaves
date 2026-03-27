#%%
import xarray as xr
from dask.distributed import Client, LocalCluster


HEATDAYS_PATH = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/heatdays/'
HEATWAVE_DAY_THRESH = 3


#%%

def stack_time(heatdays):

    start_year = heatdays['year'].min().item()
    end_year = heatdays['year'].max().item()

    tindex = xr.date_range(start=f'{start_year}-01-01',
                end=f'{end_year}-12-31',
                freq='D', calendar='noleap',
                use_cftime=True)

    heatdays = heatdays.stack(time=("year", "doy"),
                              create_index=False)
    heatdays = heatdays.reset_coords(drop=True)
    heatdays.coords['time'] = tindex
    return heatdays


#%%

def remove_short_runs(heatdays, thresh=HEATWAVE_DAY_THRESH):
    # 1) True only at run-ends of length >= thresh
    is_run_end = heatdays
    for k in range(1, thresh):
        is_run_end = is_run_end & heatdays.shift(time=k, fill_value=False)

    # 2) Expand each run-end back over the previous thresh-1 days
    mask = is_run_end
    for k in range(1, thresh):
        mask = mask | is_run_end.shift(time=-k, fill_value=False)

    # 3) Keep only days belonging to long-enough runs
    heatwaves = heatdays & mask
    return heatwaves


def calculate_heatwaves(thresh=HEATWAVE_DAY_THRESH):
    heatdays = xr.open_zarr(HEATDAYS_PATH+'heatdays.zarr',
                            consolidated=True)['heatdays']
    heatdays = stack_time(heatdays)
    heatwaves = remove_short_runs(heatdays, thresh)
    return heatwaves

if __name__ == '__main__':
    with LocalCluster(n_workers=16,       
            threads_per_worker=2,  
            memory_limit='8GB',  
            dashboard_address=':8787') as cluster:
        with Client(cluster) as client:
            print('client started')
            heatwaves = calculate_heatwaves()
            heatwaves.name = 'heatwaves'
            heatwaves.to_zarr(HEATDAYS_PATH+'heatwaves.zarr',
                            consolidated=True)