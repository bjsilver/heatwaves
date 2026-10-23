#%%
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from constants import AREA_BBOX, STORAGE_DIR, AMAZON_BASIN_BUFFER_KM, FIGDIR
import geopandas as gpd
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import os
import contextily as ctx
import geopandas as gpd
from ufuncs import get_amazon_basin_mask
import contextily as ctx
from datetime import datetime
from inmet_data import load_inmet_data
from isd_download import get_isd_tmax, load_isd_amazon_metadata
import pickle as pkl


ELEVATION_PATH = STORAGE_DIR + 'elevation/'
AMAZON_BASIN = get_amazon_basin_mask(buffer_km=AMAZON_BASIN_BUFFER_KM)
AMAZON_BASIN_NO_BUFFER = get_amazon_basin_mask(buffer_km=0)


def define_local_azimuthal_projection(recalculate=False, gdf=None):
    if not recalculate and os.path.exists('./data/azimuthal_projection.pkl'):
        with open('./data/azimuthal_projection.pkl', 'rb') as f:
            return pkl.load(f)

    if gdf.empty:
        raise ValueError("Input GeoDataFrame is empty")

    lon0, lat0 = gdf.centroid.x.mean(), gdf.centroid.y.mean()
    az_proj =ccrs.AzimuthalEquidistant(central_longitude=lon0, central_latitude=lat0)
    # write projection to file
    with open('./data/azimuthal_projection.pkl', 'wb') as f:
        pkl.dump(az_proj, f)
    return az_proj

def station_map_background(crs):

    fig = plt.figure(figsize=(10, 10), dpi=400)
    ax = plt.axes(projection=crs)
    # satellite imagery
    

    ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5, color='red')
    ax.add_feature(cfeature.COASTLINE, alpha=0.5)


    AMAZON_BASIN['vector'].to_crs(crs).boundary.plot(ax=ax, color='forestgreen', 
                                         linewidth=1, zorder=10, label=f'Amazon Basin {AMAZON_BASIN_BUFFER_KM} km buffer')
    AMAZON_BASIN_NO_BUFFER['vector'].to_crs(crs).boundary.plot(ax=ax, color='black',
                                         linewidth=1, zorder=10, label='Amazon Basin boundary')
    # set extent to basin + 2 degrees
    ax.set_extent([AMAZON_BASIN['vector'].total_bounds[0]-2, AMAZON_BASIN['vector'].total_bounds[2]+2,
                   AMAZON_BASIN['vector'].total_bounds[1]-2, AMAZON_BASIN['vector'].total_bounds[3]+2], crs=ccrs.PlateCarree())
    ctx.add_basemap(ax, crs=crs, source=ctx.providers.Esri.WorldImagery, zoom=5,
        attribution='')
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='white', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    return fig, ax

def harmonise_metadata():
    inmet, inmet_meta = load_inmet_data()
    inmet.index = pd.to_datetime(inmet.index)
    inmet_meta = gpd.GeoDataFrame(inmet_meta, 
        geometry=gpd.points_from_xy(inmet_meta['lon'], inmet_meta['lat']), crs='EPSG:4326')
    inmet_meta = inmet_meta.reset_index()
    inmet_meta['station_id'] = inmet_meta['station_id'].astype(str)
    inmet_meta['data_source'] = 'INMET'
    isd = get_isd_tmax()
    isd_meta = load_isd_amazon_metadata()
    isd_meta['station_id'] = isd_meta['USAF'].astype(str) + isd_meta['WBAN'].astype(str).str.zfill(5)
    isd_meta = isd_meta.rename({'latitude': 'lat', 'longitude': 'lon'}, axis=1)
    isd_meta['data_source'] = 'ISD'
    return inmet_meta, isd_meta, inmet, isd


def plot_stations(meta, crs):

    fig, ax = station_map_background(crs=crs)
    meta.loc[meta['data_source'] == 'INMET']\
        .plot(ax=ax, color='cyan',markersize=30, alpha=1, label='INMET station')
    meta.loc[meta['data_source'] == 'ISD']\
        .plot(ax=ax, color='darkorange',markersize=30, alpha=1, label='ISD station',
            marker='^')
    ax.legend(loc='lower left')

    fig.savefig(FIGDIR+'mgwr_results/station_map.png', dpi=400, bbox_inches='tight')



def find_overlapping_stations(meta, inmet, isd, buffer_km=1):
    buffered_stations = meta.loc[meta.data_source == 'INMET'].geometry.buffer(buffer_km*1000)  # buffer of 1 km

    pairs = []
    for i, station in meta.loc[meta.data_source == 'ISD'].iterrows():
        # Check if the ISD station intersects with any of the buffered INMET stations
        if buffered_stations.intersects(station.geometry).any():
            print(f"ISD station {station['station_id']} is within 1 km of an INMET station.")
            buffered_stations.intersects(station.geometry).any()
            inmet_id = meta.loc[meta.data_source == 'INMET'].loc[buffered_stations.intersects(station.geometry), 'station_id'].item()
            pairs.append((inmet_id, station['station_id']))
    print(f"Found {len(pairs)} overlapping station pairs.")

    stations_to_drop = []
    for pair in pairs:
        inmet_id, isd_id = pair
        inmet_sr = inmet[inmet_id]


        if isd_id not in isd.columns:
            print(f"ISD station {isd_id} not found in ISD data.")
            continue
        isd_sr = isd[isd_id]

        if inmet_sr.isna().sum() < isd_sr.isna().sum():
            stations_to_drop.append(isd_id)
        else:
            stations_to_drop.append(inmet_id)

    return stations_to_drop


def harmonise_data(inmet, isd, inmet_meta, isd_meta):
    
    meta = pd.concat([inmet_meta[['station_id', 'lat', 'lon', 'data_source']], 
                    isd_meta[['station_id', 'lat', 'lon', 'data_source']]], axis=0)
    meta = gpd.GeoDataFrame(meta, geometry=gpd.points_from_xy(meta['lon'], meta['lat']), crs='EPSG:4326')
    # clip to buffer
    meta = meta[meta.geometry.within(AMAZON_BASIN['vector'].geometry.iloc[0])]
    crs = define_local_azimuthal_projection(recalculate=True, gdf=meta)

    meta = meta.to_crs(crs)
    stations_to_drop = find_overlapping_stations(meta, inmet, isd, buffer_km=1)

    df = pd.concat([inmet, isd], axis=1)
    keep_stations = meta['station_id'].unique()
    keep_stations = [s for s in keep_stations if s in df.columns]
    keep_stations = [s for s in keep_stations if s not in stations_to_drop]
    df = df[keep_stations]
    meta = meta[meta['station_id'].isin(keep_stations)]
    return df, meta, crs


def get_harmonised_data(recompute=False):

    cache_path = STORAGE_DIR+'obs_data/harmonised_data.pkl'

    if not recompute and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pkl.load(f)

    inmet_meta, isd_meta, inmet, isd = harmonise_metadata()
    df, meta, crs = harmonise_data(inmet, isd, inmet_meta, isd_meta)
    # make lat and lon column floats
    meta['lat'] = meta['lat'].astype(float)
    meta['lon'] = meta['lon'].astype(float)
    # save to cache
    with open(cache_path, 'wb') as f:
        pkl.dump((df, meta, crs), f)
    return df, meta, crs

#%%
if __name__ == '__main__':
    inmet_meta, isd_meta, inmet, isd = harmonise_metadata()
    df, meta, crs = harmonise_data(inmet, isd, inmet_meta, isd_meta)
    plot_stations(meta, crs=crs)