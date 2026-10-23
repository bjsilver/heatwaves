#%%

import pandas as pd
import geopandas as gpd
from ufuncs import get_amazon_basin_mask
from constants import ISD_DATA_DIR, INMET_PATH
import concurrent.futures as cf
import os
from glob import glob

STATION_DOWNLOAD_WORKER_COUNT = 20


def load_isd_amazon_metadata(gdf=True):
    stations = pd.read_csv('data/isd-history.csv')

    stations = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations['LON'], stations['LAT']),
        crs='EPSG:4326'
    )

    AMAZON_BASIN_MASK = get_amazon_basin_mask(buffer_km=800)['vector']
    stations_in = stations[stations.within(AMAZON_BASIN_MASK.geometry.iloc[0])]

    stations_in['BEGIN'] = pd.to_datetime(stations_in['BEGIN'], format='%Y%m%d')
    stations_in['END'] = pd.to_datetime(stations_in['END'], format='%Y%m%d')
    stations_in = stations_in[(stations_in['BEGIN'] <= '2020-01-01') & (stations_in['END'] >= '2003-01-01')]
    stations_in['station_id'] = stations_in['USAF'] + stations_in['WBAN'].astype(str).str.zfill(5)
    stations_in = stations_in.rename({'LAT': 'latitude', 'LON': 'longitude'}, axis=1)
    return stations_in
#%%

def download_station_data(station_id, start_year=2003, end_year=2020):

    if os.path.exists(f'{ISD_DATA_DIR}{station_id}_{start_year}.csv'):
        print(f'Data for station {station_id} already exists. Skipping download.')
        return
        
    for year in range(start_year, end_year + 1):
        url = f'https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{year}/{station_id}.csv'
        try:
            df = pd.read_csv(url)
            df.to_csv(f'{ISD_DATA_DIR}{station_id}_{year}.csv', index=False)
            print(f'Downloaded data for station {station_id} for year {year}')
        except Exception as e:
            print(f'Error downloading data for station {station_id} for year {year}: {e}')

def download_all_stations_data(stations_df, start_year=2003, end_year=2020):
    station_ids = stations_df['station_id'].tolist()
    with cf.ThreadPoolExecutor(max_workers=STATION_DOWNLOAD_WORKER_COUNT) as executor:
        futures = {executor.submit(download_station_data, station_id, start_year, end_year): station_id for station_id in station_ids}
        for future in cf.as_completed(futures):
            station_id = futures[future]
            future.result()

# if __name__ == "__main__":
#     download_all_stations_data(stations_in)

#%% 
def load_station_data(station_id):
    paths = glob(f'{ISD_DATA_DIR}{station_id}_*.csv')
    dfs = []
    for path in paths:
        try:
            df = pd.read_csv(path)
            dfs.append(df)
        except Exception as e:
            print(f'Error loading data from {path}: {e}')
    if dfs:
        return pd.concat(dfs, ignore_index=True)

def load_tmax_data(station_id):
    df = load_station_data(station_id)
    if df is None:
        return None
    if len(df) < 2:
        return None
    if df is not None:
        df['DATE'] = pd.to_datetime(df['DATE'])
        df = df[['DATE', 'MAX']]
        sr = df.set_index('DATE').squeeze()
        sr = sr.sort_index()
        sr.name = 'tmax'
        # convert from farenheit to celsius
        sr = (sr - 32) * 5.0/9.0
        sr = sr.round(1)
        # remove outliers
        sr = sr[sr < 55]
        return sr

def load_all_tmax_data(stations_df):
    station_ids = stations_df['station_id'].tolist()
    tmax_data = {}
    for station_id in station_ids:
        df = load_tmax_data(station_id)
        if df is not None:
            tmax_data[station_id] = df
    df = pd.DataFrame(tmax_data).astype(float)
    return df

def get_isd_tmax(recompute=False):
    cache_path = f'{ISD_DATA_DIR}isd_tmax_data.pkl'
    if recompute or not os.path.exists(cache_path):
        stations_in = load_isd_amazon_metadata()
        tmax_df = load_all_tmax_data(stations_in)
        tmax_df.to_parquet(cache_path)
    else:
        tmax_df = pd.read_parquet(cache_path)
    return tmax_df
