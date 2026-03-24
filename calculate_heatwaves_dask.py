#!/usr/bin/env python3
#%%
import xarray as xr
import numpy as np
from dask.distributed import Client, LocalCluster

DPATH = '/uolstore/Research/b/b0256/Users/earjba/ZhangandZhou/tropical_regions/'
PERCENTILES_DIR = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/percentiles/'

PAD = 7


def preprocess(ds):
	ds = ds.drop_vars('spatial_ref')
	# replace DOY with cftime no leap calendar
	# get the year from the filename  'source': '/uolstore/Research/b/b0256/Users/earjba/ZhangandZhou/tropical_regions/SouthAmerica_Ta_SVCMsp_2007TMAX_daily_tropics30N30S.nc'}
	source = ds.encoding['source']
	year = int(source.split('TMAX')[0][-4:])
	ds.coords['DOY'] = xr.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D', calendar='noleap')
	ds = ds.rename({'DOY':'time'})
	return ds

def drop_coords(ds):
    # Drop DOY so mfdataset doesn't try to align/sort by it
    # We also drop spatial_ref as before
    return ds.drop_vars(['DOY', 'spatial_ref'], errors='ignore')

ds = xr.open_mfdataset(DPATH+'SouthAmerica_Ta_SVCMsp_*TMAX_daily_tropics30N30S.nc',
					   preprocess=drop_coords, concat_dim='time', combine='nested',
                       chunks={})
da  = ds['Ta_TMAX']

# create zarr
da_rechunked = da.chunk({'time': 365, 'lat': 200, 'lon': 200})
#%%

with LocalCluster(n_workers=12,       
    threads_per_worker=2,  
    memory_limit='20GB',  
    dashboard_address=':8787') as cluster:
        with Client(cluster) as client:
            print('cluster started')
            da_rechunked.to_zarr(PERCENTILES_DIR + 'temp_timeseries.zarr', mode='w')

#%%


def calculate_percentile(da, doy):

    doys = np.array([t.dayofyr for t in da.time.values])
    # check if DOY in circular range DOY-PAD to DOY+PAD
    mask = ((doys >= doy - PAD) & (doys <= doy + PAD)) | ((doys >= doy - PAD + 365) & (doys <= doy + PAD + 365))
    da_doy = da.isel(time=mask)

    doy_percentile = da_doy.quantile(0.95, dim='time')
    doy_percentile = doy_percentile.expand_dims({'DOY': [doy]})
    return doy_percentile


doy_percentile = calculate_percentile(da, 1)

# percentiles = []
# for doy in range(1, 366):
#     print(f'Calculating percentile for DOY {doy}')
#     doy_percentile = calculate_percentile(da, doy)
#     percentiles.append(doy_percentile)


#%%
# 1. Create the Cluster

with LocalCluster(n_workers=4,       
    threads_per_worker=4,  
    memory_limit='15GB',  
    dashboard_address=':8787') as cluster:
        with Client(cluster) as client:
            # All your processing happens here
            print(f"Dashboard: {client.dashboard_link}")


            doy_percentile.to_netcdf(PERCENTILES_DIR+'percentiles_doy_1.nc')

