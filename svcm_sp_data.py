#%%
import xarray as xr
import pandas as pd
import numpy as np
from constants import STORAGE_DIR, MIN_VALID_YEARS, MIN_VALID_DAYS_PER_YEAR
import os
import sqlite3
import geopandas as gpd
from dask.diagnostics import ProgressBar
from ufuncs import local_cluster
from pyproj import CRS, Transformer


DATE_SLICE = slice('2003-01-01', '2020-12-31')
METEOSTAT_DATA_PATH = STORAGE_DIR + 'meteostat_data/'
LST_DIR = STORAGE_DIR + 'lst/'
DATA_PATH = STORAGE_DIR + 'svcm_data/'


def load_meteostat_stations():
    db = "./data/stations.db"
    con = sqlite3.connect(db)
    stations = pd.read_sql_query(
        "SELECT * FROM stations",
        con
    )
    con.close()
    stations = gpd.GeoDataFrame(stations, geometry=gpd.points_from_xy(stations.longitude, stations.latitude), crs='EPSG:4326')
    return stations

def get_valid_meteostat_stations(min_valid_years=MIN_VALID_YEARS, min_valid_days_per_year=MIN_VALID_DAYS_PER_YEAR):

    meteostat_paths = [f for f in os.listdir(METEOSTAT_DATA_PATH) if f.endswith('.parquet')]

    year_counts = pd.DataFrame()
    for path in meteostat_paths:
        station_id = path.split('/')[-1].split('.')[0]
        df = pd.read_parquet(METEOSTAT_DATA_PATH+path)
        if 'temp' in df.columns:
            sr = df['temp'].resample('D').max().squeeze()
        elif 'tmax' in df.columns:
            sr = df.squeeze()
        else:
            continue
        if type(sr) is not pd.Series:
            continue
        sr_annual_count = sr.loc[DATE_SLICE].resample('YS').count()
        sr_annual_count =sr_annual_count.reindex(pd.date_range(start=DATE_SLICE.start, end=DATE_SLICE.stop, freq='YS'), fill_value=0)
        year_counts[station_id] = sr_annual_count

    valid_stations = year_counts.T.loc[((year_counts >= min_valid_days_per_year).sum() >= min_valid_years)].index

    meteostat_stations = load_meteostat_stations()
    valid_meteostat_stations = meteostat_stations[meteostat_stations['id'].isin(valid_stations)]
    return valid_meteostat_stations

def load_meteostat_data(valid_meteostat_stations):

    # load the stations in the valid paths
    meteostat_df = pd.DataFrame()
    for station_id in valid_meteostat_stations['id']:
        path = METEOSTAT_DATA_PATH + f'{station_id}.parquet'
        df = pd.read_parquet(path)
        if 'temp' in df.columns:
            sr = df['temp'].resample('D').max().squeeze()
        elif 'tmax' in df.columns:
            sr = df.squeeze()
        else:
            raise ValueError(f"Station {station_id} has unrecognised format")
        sr = sr.loc[DATE_SLICE]
        
        meteostat_df[station_id] = sr

    meteostat_df.columns.name = 'station_id'
    return meteostat_df


#%% get interpolated LST

def interpolate_lst_to_stations(mdf):

    station = xr.DataArray(mdf['id'].values, dims='station')
    lat = xr.DataArray(mdf['latitude'].values, dims='station', coords={'station': station})
    lon = xr.DataArray(mdf['longitude'].values, dims='station', coords={'station': station})


    for year in range(2003, 2021):
        lst = xr.open_zarr(LST_DIR+f'zz_lst_{year}.zarr')['LST'] / 10


        lst_interped = lst.interp(lon=lon, lat=lat, method='linear')
        lst_interped.to_zarr(LST_DIR+f'interped/lst_interped_obs_{year}.zarr', mode='w')

def intepolate_elevation_to_stations(mdf):

    station = xr.DataArray(mdf['id'].values, dims='station')
    lat = xr.DataArray(mdf['latitude'].values, dims='station', coords={'station': station})
    lon = xr.DataArray(mdf['longitude'].values, dims='station', coords={'station': station})

    elev = xr.open_dataset(os.path.join(STORAGE_DIR, '../' 'elevation/orig_resolution_regridded', 'elevation_regridded.nc'))['elevation']
    elev_interped = elev.interp(lon=lon, lat=lat, method='linear')
    elev_interped.to_zarr(STORAGE_DIR+'elevation/elev_interped_obs.zarr', mode='w')

def intepolate_data_to_stations(mdf):
    mdf = get_valid_meteostat_stations(min_valid_years=MIN_VALID_YEARS, min_valid_days_per_year=MIN_VALID_DAYS_PER_YEAR)

    with local_cluster(n_workers=16, threads_per_worker=16, memory_limit='50GB',
                    suppress_dask_warnings=False) as cluster:
        
        interpolate_lst_to_stations(mdf)
        intepolate_elevation_to_stations(mdf)        

if __name__ == '__main__':
    mdf = get_valid_meteostat_stations(min_valid_years=12, min_valid_days_per_year=50)
    intepolate_data_to_stations(mdf)

def load_svcm_data(recompute=False):

    if not recompute and os.path.exists(DATA_PATH+'svcm_long.parquet'):
        print("Loading long format dataframe from parquet...")
        df = pd.read_parquet(DATA_PATH+'svcm_long.parquet')
        return df

    print("Building long format dataframe...")
    tindex = xr.date_range(start=DATE_SLICE.start, end=DATE_SLICE.stop, freq='D',
                                        calendar='noleap')
    valid_meteostat_stations = get_valid_meteostat_stations(min_valid_years=12, min_valid_days_per_year=50)
    t2m_df = load_meteostat_data(valid_meteostat_stations)
    t2m_df = t2m_df.reindex(pd.date_range(start=DATE_SLICE.start, end=DATE_SLICE.stop, freq='D'))

    lon0, lat0 = valid_meteostat_stations.centroid.x, valid_meteostat_stations.centroid.y
    az_proj = CRS(proj='aeqd', lat_0=lat0, lon_0=lon0, datum='WGS84')
    # write projection to file
    with open(DATA_PATH+'azimuthal_projection.txt', 'w') as f:
        f.write(az_proj.to_proj4())
    mdf_az_proj = valid_meteostat_stations.to_crs(az_proj)

    # drop leap days
    t2m_df = t2m_df[~((t2m_df.index.month == 2) & (t2m_df.index.day == 29))]
    t2m_df.index = tindex

    sr = t2m_df.stack()
    sr.name = 't2m'
    sr = sr.sort_index()


    lst = xr.open_mfdataset(LST_DIR+'interped/lst_interped_obs_*.zarr', 
                            concat_dim='year', combine='nested')['LST']
    elev = xr.open_zarr(STORAGE_DIR+'elevation/elev_interped_obs.zarr')['elevation']

    ds = xr.merge([lst, elev], compat='no_conflicts', join='override')
    # add northing and easting
    ds = ds.assign_coords({'easting': (('station'), mdf_az_proj['geometry'].x.values),
                           'northing': (('station'), mdf_az_proj['geometry'].y.values)})


    ds = ds.rename({'station': 'station_id'})
    ds = ds.stack(time=('year', 'doy')).drop_vars(['band', 'spatial_ref'])
    ds = ds.drop_vars(['time', 'year', 'doy', 'lat', 'lon'])
    ds.coords['time'] = tindex
    ds = ds.stack(mindex=('time', 'station_id'))
    lst_df = ds.to_dataframe()
    lst_df = lst_df.sort_index()


    df = pd.concat([sr, lst_df], axis=1)
    df = df.dropna(how='any')
    df = df.reset_index()
    df = df.rename(columns={'level_0': 'time'})
    df['year'] = df['time'].apply(lambda i: i.year)
    df['doy'] = df['time'].apply(lambda i: i.dayofyr)
    df = df.drop(columns=['time'])


    df.to_parquet(DATA_PATH+'svcm_long.parquet', index=False)
    return df