#%%
import os
from constants import (
    STORAGE_DIR,
    LST_DIR,
    MGWR_DATA_DIR,
    DATE_SLICE,
    DOY_POOL_WINDOW,
    CLIMATOLOGY_SCORE_THRESHOLD,
    LST_CORRELATION_THRESHOLD,
    ELEVATION_THRESHOLD
)
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from ufuncs import local_cluster, get_amazon_basin_mask
from pyproj import CRS
import pickle as pkl
import statsmodels.formula.api as smf
import cartopy.crs as ccrs
from obs_data import get_harmonised_data



def interpolate_lst_to_stations(mdf):

    station = xr.DataArray(mdf['station_id'].values, dims='station')
    lat = xr.DataArray(mdf['lat'].values, dims='station', coords={'station': station})
    lon = xr.DataArray(mdf['lon'].values, dims='station', coords={'station': station})

    lst = xr.open_zarr(LST_DIR+'regridded_rechunked.zarr')['LST'] / 10

    lst_interped = lst.interp(lon=lon, lat=lat, method='linear')
    lst_interped = lst_interped.chunk({'station': -1, 'time': -1})
    lst_interped.to_zarr(LST_DIR+f'interped/lst_interped_obs.zarr', mode='w')

def intepolate_elevation_to_stations(mdf):

    station = xr.DataArray(mdf['station_id'].values, dims='station')
    lat = xr.DataArray(mdf['lat'].values, dims='station', coords={'station': station})
    lon = xr.DataArray(mdf['lon'].values, dims='station', coords={'station': station})

    elev = xr.open_dataset(os.path.join(STORAGE_DIR, '../' 'elevation/orig_resolution_regridded', 'elevation_regridded.nc'))['elevation']
    elev_interped = elev.interp(lon=lon, lat=lat, method='linear')
    elev_interped.to_zarr(STORAGE_DIR+'elevation/elev_interped_obs.zarr', mode='w')

def intepolate_data_to_stations(mdf, recompute=False):

    if recompute:

        with local_cluster(n_workers=16, threads_per_worker=16, memory_limit='50GB',
                        suppress_dask_warnings=False) as cluster:
            
            interpolate_lst_to_stations(mdf)
            intepolate_elevation_to_stations(mdf)
    else:
        print("Skipping interpolation of LST and elevation to stations. Set recompute=True to force recomputation.")




def get_interpolated_data(recompute=False):

    if not recompute and os.path.exists(MGWR_DATA_DIR+'mgwr_climatological_long.parquet'):
        print("Loading climatologically valid data from cache...")
        df = pd.read_parquet(MGWR_DATA_DIR+'mgwr_climatological_long.parquet')
        return df

    tmax_df, stations, az_proj = get_harmonised_data()


    # load the tmax data for the valid stations
    tmax_df = tmax_df.loc[slice(DATE_SLICE.start, DATE_SLICE.stop)]
    tmax_df = tmax_df.reindex(pd.date_range(start=DATE_SLICE.start, end=DATE_SLICE.stop, freq='D'))
    tindex = xr.date_range(start=DATE_SLICE.start, end=DATE_SLICE.stop, freq='D',
                                        calendar='noleap')
    tmax_df = tmax_df[~((tmax_df.index.month == 2) & (tmax_df.index.day == 29))]
    tmax_df.index = tindex
   

    tmax = tmax_df.stack()
    tmax.name = 'tmax'
    tmax = tmax.sort_index()

    lst = xr.open_zarr(LST_DIR+'interped/lst_interped_obs.zarr')['LST']
    elev = xr.open_zarr(STORAGE_DIR+'elevation/elev_interped_obs.zarr')['elevation']

    ds = xr.merge([lst, elev], compat='no_conflicts', join='override')
    ds = ds.rename({'station': 'station_id'})
    # select only stations in mdf+tmax
    ds = ds.loc[{'station_id':stations['station_id'].values}]
    # add northing and easting
    ds = ds.assign_coords({'easting': (('station_id'), stations['geometry'].x.values),
                            'northing': (('station_id'), stations['geometry'].y.values)})


    ds = ds.drop_vars(['band', 'spatial_ref'])
    ds = ds.stack(mindex=('time', 'station_id'))
    lst_df = ds.to_dataframe()
    lst_df = lst_df.sort_index()

    df = pd.concat([tmax, lst_df], axis=1)
    # set names for multiindex levels
    df.index.names = ['datetime', 'station_id']
    df = df.reset_index(level='station_id')

    df['year'] = [i.year for i in df.index]
    df['doy'] = [i.dayofyr for i in df.index]

    datetimes = df.reset_index()['datetime'].apply(lambda i: pd.Timestamp(i.strftime()))
    df.index = datetimes

    df.to_parquet(MGWR_DATA_DIR+'mgwr_climatological_long.parquet',
                  index=True, engine='pyarrow')





def get_climatologically_valid_data(climatology_score_threshold=CLIMATOLOGY_SCORE_THRESHOLD,
                                    lst_correlation_threshold=LST_CORRELATION_THRESHOLD,
                                    elevation_threshold=ELEVATION_THRESHOLD):

    tmax_df = get_interpolated_data(recompute=False)

    station_doy_count = tmax_df.groupby([tmax_df.station_id, tmax_df.doy])['tmax'].count()
    station_doy_mean = station_doy_count.groupby(level='station_id').mean()
    station_doy_std  = station_doy_count.groupby(level='station_id').std()
    score = station_doy_mean / np.sqrt(station_doy_std)
    score = score.sort_values(ascending=False)

    valid_stations = score[score >= climatology_score_threshold].index.get_level_values(0).unique()

    df = tmax_df[tmax_df['station_id'].isin(valid_stations)]

    if lst_correlation_threshold is not None:
        lst_corr = df.groupby('station_id')[['tmax', 'LST']].corr().iloc[0::2,-1]
        lst_corr = lst_corr.reset_index(level=1, drop=True)
        valid_stations = lst_corr[lst_corr >= lst_correlation_threshold].index.get_level_values(0).unique()
        df = df[df['station_id'].isin(valid_stations)]

    if elevation_threshold is not None:
        elev = df.groupby('station_id')['elevation'].mean()
        valid_stations = elev[elev < elevation_threshold].index.get_level_values(0).unique()
        df = df[df['station_id'].isin(valid_stations)]

    return df

#%%
if __name__ == '__main__':
    with local_cluster(n_workers=4, threads_per_worker=16, memory_limit='50GB',
                        suppress_dask_warnings=False) as cluster:


        stations_basin, mdf, crs = get_harmonised_data(recompute=False)
        intepolate_data_to_stations(mdf, recompute=True)
        get_interpolated_data(recompute=True)