#%%
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from constants import AREA_BBOX, STORAGE_DIR, RESULTS_BBOX, BORDERS_DIR
import geopandas as gpd
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import numpy as np  
import os
import xarray as xr
import warnings
from tqdm import tqdm
import contextily as ctx
import geopandas as gpd
import sqlite3
import meteostat as ms
# import kagglehub
from ufuncs import get_amazon_basin_mask
import contextily as ctx
from datetime import datetime

LEGAL_AMAZON = gpd.read_file(BORDERS_DIR+'Limites_Amazonia_Legal_2022.shp')

METEOSTAT_DATA_PATH = STORAGE_DIR + 'meteostat_data/'
INMET_PATH = STORAGE_DIR + 'INMET/'
ELEVATION_PATH = STORAGE_DIR + 'elevation/'

AMAZON_BASIN = get_amazon_basin_mask(buffer_km=800)

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


def get_INMET_station_data_metadata():
    path = STORAGE_DIR + 'INMET/'
    adf = pd.read_parquet(path+'automaticas_processed.parquet')
    condf = pd.read_parquet(path+'convencionais_processed.parquet')


    # extract metadata
    cols = ['station_code', 'station_name', 
            'latitude', 'longitude', 'start_date', 'end_date',
            'altitude', 'status']
    adf_mdf = adf.groupby(adf.station_code)[cols].first()
    adf_mdf['station_type'] = 'automatic'
    condf_mdf = condf.groupby(condf.station_code)[cols].first()
    condf_mdf['station_type'] = 'conventional'

    mdf = pd.concat([adf_mdf, condf_mdf], axis=0)
    mdf = gpd.GeoDataFrame(mdf, geometry=gpd.points_from_xy(mdf.longitude, mdf.latitude), crs='EPSG:4326')
    return mdf


def station_map_background():

    fig = plt.figure(figsize=(10, 10), dpi=400)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.stock_img()

    ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5, color='red')
    ax.add_feature(cfeature.COASTLINE, alpha=0.5)


    AMAZON_BASIN['vector'].boundary.plot(ax=ax, color='forestgreen', 
                                         linewidth=1, zorder=10, label='Amazon Basin boundary')
    # set extent to basin + 2 degrees
    ax.set_extent([AMAZON_BASIN['vector'].total_bounds[0]-2, AMAZON_BASIN['vector'].total_bounds[2]+2,
                   AMAZON_BASIN['vector'].total_bounds[1]-2, AMAZON_BASIN['vector'].total_bounds[3]+2], crs=ccrs.PlateCarree())
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    return fig, ax



def download_meteostat_data():

    stations = load_meteostat_stations()

    # get stations within the amazon basin
    stations_basin = stations[stations.geometry.within(AMAZON_BASIN['vector'].geometry.iloc[0])]

    ms.config.block_large_requests = False

    start = datetime(2003, 1, 1)
    end = datetime(2024, 12, 31, 23, 59)
    # use meteostat to get station data for the stations in the basin
    
    for station_id in tqdm(stations_basin['id']):

        if os.path.exists(METEOSTAT_DATA_PATH + f'{station_id}.parquet'):
            print(f'Skipping {station_id}, already exists.')
            continue
        station = ms.stations.meta(station_id) 
        inventory = ms.stations.inventory(station_id)
        if 'tmax' in inventory.parameters:
            
            ts = ms.daily(ms.Station(id=station_id), start, end,
                        parameters=['tmax'])
            df = ts.fetch()
        elif 'temp' in inventory.parameters:
            ts = ms.hourly(ms.Station(id=station_id), start, end,
                        parameters=['temp'])
            df = ts.fetch()
            print('found!')
        else:
            print(f'Skipping {station_id}, no temperature data.')
            continue
        if df is not None:
            print(f'Saving data for {station_id}.')
            df.to_parquet(METEOSTAT_DATA_PATH + f'{station_id}.parquet')

download_meteostat_data()

#%%
inmet_stations = get_INMET_station_data_metadata()
inmet_stations_automatic = inmet_stations.loc[inmet_stations['station_type']=='automatic']
inmet_stations_conventional = inmet_stations.loc[inmet_stations['station_type']=='conventional']
meteostat_stations = load_meteostat_stations()
meteostat_stations['station_type'] = 'meteostat'
#%%
fig, ax = station_map_background()
inmet_stations_automatic\
    .plot(ax=ax, color='blue',markersize=30, alpha=1, label='INMET Automatic')
inmet_stations_conventional\
    .plot(ax=ax, color='green',markersize=30, alpha=1, label='INMET Conventional',
          marker='^')
meteostat_stations.plot(ax=ax, color='yellow', markersize=10, alpha=1,
                        marker='x', label='Meteostat station')

ax.legend()

#%%

def get_stations_in_basin():
    inmet_stations = get_INMET_station_data_metadata()
    inmet_stations_automatic = inmet_stations.loc[inmet_stations['station_type']=='automatic']
    inmet_stations_conventional = inmet_stations.loc[inmet_stations['station_type']=='conventional']
    inmet_stations_automatic_basin = inmet_stations_automatic[inmet_stations_automatic.geometry.within(AMAZON_BASIN['vector'].geometry.iloc[0])]
    inmet_stations_conventional_basin = inmet_stations_conventional[inmet_stations_conventional.geometry.within(AMAZON_BASIN['vector'].geometry.iloc[0])]

    meteostat_stations = load_meteostat_stations()
    meteostat_stations['station_type'] = 'meteostat'
    meteostat_ids = [f.split('/')[-1].split('.')[0] for f in os.listdir(METEOSTAT_DATA_PATH) if f.endswith('.parquet')]
    meteostat_stations_basin = meteostat_stations[meteostat_stations['id'].isin(meteostat_ids)]

    stations = pd.concat([inmet_stations_automatic_basin, inmet_stations_conventional_basin, meteostat_stations_basin], axis=0)

    # identify the locations that are with 1km of each other
    stations = stations.to_crs(epsg=3857) # convert to meters
    stations['buffer'] = stations.geometry.buffer(100)

    # groupby overlapping buffers
    stations['group'] = None
    group_id = 0
    for idx, row in stations.iterrows():
        if stations.loc[idx, 'group'] is not None:
            continue
        # find overlapping buffers
        overlapping = stations[stations['buffer'].intersects(row['buffer'])]
        stations.loc[overlapping.index, 'group'] = group_id
        group_id += 1

    return stations


class StationsData:
    def __init__(self, inmet_path=INMET_PATH, meteostat_data_path=METEOSTAT_DATA_PATH):
        self.inmet_path = inmet_path
        self.meteostat_data_path = meteostat_data_path
        self.adf = pd.read_parquet(self.inmet_path + 'automaticas_processed.parquet')
        self.condf = pd.read_parquet(self.inmet_path + 'convencionais_processed.parquet')
        self.meteostat_paths = [
            self.meteostat_data_path + f
            for f in os.listdir(self.meteostat_data_path)
            if f.endswith('.parquet')
        ]

    def get_data(self, station_sr):
        if station_sr['station_type'] in ['automatic', 'conventional']:
            if station_sr['station_type'] == 'automatic':
                df = self.adf[self.adf['station_code'] == station_sr['station_code']]
                df = df[['datetime', 'temp_max_prev_hour_C']].copy()
                df = df.rename(columns={'temp_max_prev_hour_C': 'tmax'})
                sr = df.set_index('datetime').squeeze()
                sr = sr.resample('D').max()
            else:
                df = self.condf[self.condf['station_code'] == station_sr['station_code']]
                df = df[['datetime', 'dry_bulb_temp_C']].copy()
                df = df.rename(columns={'dry_bulb_temp_C': 'tmax'})
                sr = df.set_index('datetime').squeeze()
                sr = sr.resample('D').max()
        else:
            df = pd.read_parquet(self.meteostat_data_path + f"{station_sr['id']}.parquet")
            if 'temp' in df.columns:
                sr = df['temp'].resample('D').max().squeeze()
            elif 'tmax' in df.columns:
                sr = df.squeeze()
            else:
                sr = pd.Series(dtype=float)

        return sr

stations_data = StationsData()
stations = get_stations_in_basin()

#%%

date_slice = slice('2003-01-01', '2024-12-31')
date_range = pd.date_range(start=date_slice.start, end=date_slice.stop, freq='D')

automatic_conventional_diff = pd.Series()
automatic_meteostat_diff = pd.Series()
conventional_meteostat_diff = pd.Series()

for group in stations['group'].unique():
    group_stations = stations[stations['group'] == group]
    if len(group_stations) > 1:
        print(f"Group {group} has {len(group_stations)} stations:")
        group_dfs = {}
        for i, sr in group_stations.iterrows():
            group_dfs[sr['station_type']] = stations_data.get_data(sr)
        sdf = pd.DataFrame(index=group_dfs.keys())
        for station_type, sr in group_dfs.items():
            sdf.loc[station_type, 'n_obs'] = sr.loc[date_slice].notna().sum()

        if 'automatic' in group_dfs and 'conventional' in group_dfs:
            diff = group_dfs['automatic'].loc[date_slice] - group_dfs['conventional'].loc[date_slice]
            automatic_conventional_diff.loc[group] = diff.mean()
        elif 'automatic' in group_dfs and 'meteostat' in group_dfs:
            diff = group_dfs['automatic'].loc[date_slice] - group_dfs['meteostat'].loc[date_slice]
            automatic_meteostat_diff.loc[group] = diff.mean()
        elif 'conventional' in group_dfs and 'meteostat' in group_dfs:
            diff = group_dfs['conventional'].loc[date_slice] - group_dfs['meteostat'].loc[date_slice]
            conventional_meteostat_diff.loc[group] = diff.mean()

#%%
automatic_conventional_diff.plot.bar(ylabel='Mean Temperature Difference (C)', title='Automatic - Conventional mean temperature diff (100m buffer)',
                                     grid=True)
automatic_meteostat_diff.plot.bar(ylabel='Mean Temperature Difference (C)', title='Automatic - Meteostat mean temperature diff (100m buffer)',
                                  grid=True)
conventional_meteostat_diff.plot.bar(ylabel='Mean Temperature Difference (C)', title='Conventional - Meteostat mean temperature diff (100m buffer)',
                                     grid=True)

#%%
lengths_diff = pd.Series(index=automatic_meteostat_diff.index)
for group_number in automatic_meteostat_diff.index:
    group_stations = stations[stations['group'] == group_number]
    print(group_number)
    automatic_station = stations_data.get_data(group_stations[group_stations['station_type'] == 'automatic'].iloc[0])
    meteostat_station = stations_data.get_data(group_stations[group_stations['station_type'] == 'meteostat'].iloc[0])
    lengths_diff.loc[group_number] = len(automatic_station.loc[date_slice].dropna()) - len(meteostat_station.loc[date_slice].dropna())
#%%
lengths_diff.plot.bar(ylabel='Length Difference (days)', title='Automatic - Meteostat co-located station n obs diff',
                      grid=True)

#%%

meteostat_paths = [f for f in os.listdir(METEOSTAT_DATA_PATH) if f.endswith('.parquet')]
date_slice = slice('2003-01-01', '2020-12-31')
date_range = pd.date_range(start=date_slice.start, end=date_slice.stop, freq='D')

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
    sr_annual_count = sr.loc[date_slice].resample('YS').count()
    sr_annual_count =sr_annual_count.reindex(pd.date_range(start=date_slice.start, end=date_slice.stop, freq='YS'), fill_value=0)
    year_counts[station_id] = sr_annual_count

valid_stations = year_counts.T.loc[((year_counts >= 50).sum() >= 12)].index

meteostat_stations = load_meteostat_stations()
valid_meteostat_stations = meteostat_stations[meteostat_stations['id'].isin(valid_stations)]

#%%

fig, ax = station_map_background()

basin_no_buffer = get_amazon_basin_mask(buffer_km=0)
basin_no_buffer['vector'].boundary.plot(ax=ax, color='orange', 
                                         linewidth=1, zorder=10, label='Amazon Basin boundary (no buffer)')
valid_meteostat_stations.plot(ax=ax, color='red', markersize=30, alpha=1,
                        marker='x', label='Meteostat station')