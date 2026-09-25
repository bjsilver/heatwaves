#%%
import os
from constants import STORAGE_DIR, LST_DIR, DOY_POOL_WINDOW, MGWR_DATA_DIR, METEOSTAT_DATA_PATH, DATE_SLICE, MGWR_RESULTS_DIR, LAPSE_RATE
import numpy as np
import pandas as pd
import sqlite3
import geopandas as gpd
import xarray as xr
from ufuncs import local_cluster, get_amazon_basin_mask
from pyproj import CRS
import pickle as pkl
import statsmodels.formula.api as smf


MIN_VALID_YEARS = 12
MIN_VALID_DAYS_PER_YEAR = 120

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

def get_climatologically_valid_meteostat_stations(min_valid_years=MIN_VALID_YEARS, min_valid_days_per_year=MIN_VALID_DAYS_PER_YEAR):

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

def load_meteostat_data(mdf):

    # load the stations in the valid paths
    meteostat_df = pd.DataFrame()
    for station_id in mdf['id']:
        path = METEOSTAT_DATA_PATH + f'{station_id}.parquet'
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        if len(df) < 25:
            continue
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

def intepolate_data_to_stations(mdf, recompute=False):

    if recompute:

        with local_cluster(n_workers=16, threads_per_worker=16, memory_limit='50GB',
                        suppress_dask_warnings=False) as cluster:
            
            interpolate_lst_to_stations(mdf)
            intepolate_elevation_to_stations(mdf)
    else:
        print("Skipping interpolation of LST and elevation to stations. Set recompute=True to force recomputation.")

def define_local_azimuthal_projection(mdf):
    lon0, lat0 = mdf.centroid.x, mdf.centroid.y
    az_proj = CRS(proj='aeqd', lat_0=lat0, lon_0=lon0, datum='WGS84')
    # write projection to file
    with open('./data/azimuthal_projection.txt', 'w') as f:
        f.write(az_proj.to_wkt())
    return az_proj


def calc_lapse_rate(df):
    df = df.dropna(subset=['tmax', 'LST', 'elevation'])

    model = smf.ols(
        'tmax ~ elevation + cc(doy, df=10)',
        data=df
    ).fit()

    lapse_rate = model.params['elevation']
    return lapse_rate

def get_climatologically_valid_data(recompute=False):

    if not recompute and os.path.exists(MGWR_DATA_DIR+'mgwr_climatological_long.parquet'):
        print("Loading climatologically valid data from cache...")
        df = pd.read_parquet(MGWR_DATA_DIR+'mgwr_climatological_long.parquet')
        return df
    print("Computing climatologically valid data...")

    mdf = get_climatologically_valid_meteostat_stations(min_valid_years=MIN_VALID_YEARS, min_valid_days_per_year=MIN_VALID_DAYS_PER_YEAR)
    az_proj = define_local_azimuthal_projection(mdf)
    mdf_az_proj = mdf.to_crs(az_proj)

    # load the tmax data for the valid stations
    tmax_df = load_meteostat_data(mdf)
    tmax_df = tmax_df.loc[slice(DATE_SLICE.start, DATE_SLICE.stop)]
    tmax_df = tmax_df.reindex(pd.date_range(start=DATE_SLICE.start, end=DATE_SLICE.stop, freq='D'))
    # drop leap days
    tindex = xr.date_range(start=DATE_SLICE.start, end=DATE_SLICE.stop, freq='D',
                                            calendar='noleap')
    tmax_df = tmax_df[~((tmax_df.index.month == 2) & (tmax_df.index.day == 29))]
    tmax_df.index = tindex

    tmax = tmax_df.stack()
    tmax.name = 'tmax'
    tmax = tmax.sort_index()

    lst = xr.open_mfdataset(LST_DIR+'interped/lst_interped_obs_*.zarr', 
                                concat_dim='year', combine='nested')['LST']
    elev = xr.open_zarr(STORAGE_DIR+'elevation/elev_interped_obs.zarr')['elevation']

    ds = xr.merge([lst, elev], compat='no_conflicts', join='override')
    ds = ds.rename({'station': 'station_id'})
    # select only stations in mdf+tmax
    ds = ds.loc[{'station_id':mdf['id'].values}]
    # add northing and easting
    ds = ds.assign_coords({'easting': (('station_id'), mdf_az_proj['geometry'].x.values),
                            'northing': (('station_id'), mdf_az_proj['geometry'].y.values)})




    ds = ds.stack(time=('year', 'doy')).drop_vars(['band', 'spatial_ref'])
    ds = ds.drop_vars(['time', 'year', 'doy', 'lat', 'lon'])
    ds.coords['time'] = tindex
    ds = ds.stack(mindex=('time', 'station_id'))
    lst_df = ds.to_dataframe()
    lst_df = lst_df.sort_index()

    df = pd.concat([tmax, lst_df], axis=1)
    df = df.reset_index(level='station_id')

    df['year'] = [i.year for i in df.index]
    df['doy'] = [i.dayofyr for i in df.index]


    df['tmax'] = df['tmax'] - LAPSE_RATE * df['elevation']

    df.to_parquet(MGWR_DATA_DIR+'mgwr_climatological_long.parquet',
                  index=False, engine='pyarrow')

#%%




#%%
def doy_gaussian_mean(df, target_doy, window=10, sigma=None, n_days_per_year=365):
    if sigma is None:
        sigma = window


    offsets = np.arange(-window, window + 1)
    days = ((target_doy - 1 + offsets) % n_days_per_year) + 1
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)

    weight_lookup = pd.Series(weights, index=days)
    offset_lookup = pd.Series(offsets, index=days)

    doy_df = df[df['doy'].isin(days)].copy()
    doy_df['_weight'] = doy_df['doy'].map(weight_lookup)
    doy_df['_offset'] = doy_df['doy'].map(offset_lookup)

    doy_df['_target_year'] = doy_df['year']

    mask = (doy_df['_offset'] < 0) & (doy_df['doy'] > target_doy)
    doy_df.loc[mask, '_target_year'] = (
        doy_df.loc[mask, 'year'].to_numpy() + 1
    )

    mask = (doy_df['_offset'] > 0) & (doy_df['doy'] < target_doy)
    doy_df.loc[mask, '_target_year'] = (
        doy_df.loc[mask, 'year'].to_numpy() - 1
    )

    doy_df['_tmax_weighted'] = doy_df['tmax'] * doy_df['_weight']
    doy_df['_LST_weighted'] = doy_df['LST'] * doy_df['_weight']

    weighted_sum = (
        doy_df.groupby(['station_id', '_target_year'])
        [['_tmax_weighted', '_LST_weighted']]
        .sum()
    )

    weight_sum = (
        doy_df.assign(
            _tmax_weight=doy_df['_weight'].where(doy_df['tmax'].notna()),
            _LST_weight=doy_df['_weight'].where(doy_df['LST'].notna())
        )
        .groupby(['station_id', '_target_year'])
        [['_tmax_weight', '_LST_weight']]
        .sum()
    )

    annual = pd.DataFrame(index=weighted_sum.index)
    annual['tmax'] = weighted_sum['_tmax_weighted'] / weight_sum['_tmax_weight']
    annual['LST'] = weighted_sum['_LST_weighted'] / weight_sum['_LST_weight']

    result = (
        annual.groupby('station_id')[['tmax', 'LST']]
        .mean()
        .reset_index()
    )

    station_info = (
        df.reset_index()[['station_id', 'elevation', 'easting', 'northing']]
        .drop_duplicates('station_id')
    )

    return result.merge(station_info, on='station_id', how='left')


#%%
def doy_gaussian_climatology(window=10, sigma=None, recompute=False):

    if not recompute and os.path.exists(MGWR_DATA_DIR+'lst_climatology.zarr'):
        return xr.open_zarr(MGWR_DATA_DIR+'lst_climatology.zarr')['LST']

    da = xr.open_mfdataset(LST_DIR+'interped/lst_interped_obs_*.zarr', 
                            concat_dim='year', combine='nested')['LST']
    if sigma is None:
        sigma = window

    offsets = np.arange(-window, window + 1)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)

    results = []

    for target_doy in range(1, 366):
        doys = ((target_doy - 1 + offsets) % 365)

        values = da.isel(
            doy=xr.DataArray(doys, dims='window')
        )

        weights_xr = xr.DataArray(weights, dims='window')

        annual = (
            (values * weights_xr)
            .sum('window', skipna=True)
            / weights_xr.where(values.notnull()).sum('window')
        )

        result = annual.mean('year')
        results.append(result)

    lst_clim = xr.concat(
        results,
        dim=xr.DataArray(np.arange(1, 366), dims='doy', name='doy')
    )
    lst_clim.name = 'LST'
    lst_clim.to_zarr(MGWR_DATA_DIR+'lst_climatology.zarr', mode='w')



AMAZON_BASIN = get_amazon_basin_mask(buffer_km=800)
stations = load_meteostat_stations()
stations_basin = stations[stations.geometry.within(AMAZON_BASIN['vector'].geometry.iloc[0])]
meteo = load_meteostat_data(stations_basin)

lst = xr.open_mfdataset(LST_DIR+'interped/lst_interped_obs_*.zarr', 
                        concat_dim='year', combine='nested')['LST']
lst = lst.assign_coords({'year':range(2003, 2021), 'doy':range(1, 366)})
lst_clim = doy_gaussian_climatology(window=10, sigma=None)
elev = xr.open_zarr(STORAGE_DIR+'elevation/elev_interped_obs.zarr')['elevation']
mdf_az_proj = stations_basin.to_crs(define_local_azimuthal_projection(stations_basin))

#%%


date = '2015-06-05'
doy = pd.Timestamp(date).dayofyear
year = pd.Timestamp(date).year
# get the tmax 
tmax = meteo.loc[date].dropna()
tmax.name = 'tmax'

df = pd.DataFrame(tmax)
df['northing'] = mdf_az_proj.set_index('id').loc[tmax.index]['geometry'].y
df['easting'] = mdf_az_proj.set_index('id').loc[tmax.index]['geometry'].x
# get the climatological LST
df['lst_clim'] = lst_clim.loc[{'station':df.index.values, 'doy':doy}]
df['lst'] = lst.loc[{'station':df.index.values, 'doy':doy, 'year':year}]
df['lst_anomaly'] = df['lst'] - df['lst_clim']
df['elev'] = elev.loc[{'station':df.index.values}]


# get the model-predicted tmax
with open(f'{MGWR_RESULTS_DIR}/layer1/mgwr_model_doy_{doy}.pkl', 'rb') as f:
    layer1 = pkl.load(f)

selector = layer1['selector']

#%%
#%% interpolate data to station_locations
if __name__ == '__main__':

    AMAZON_BASIN = get_amazon_basin_mask(buffer_km=800)
    stations = load_meteostat_stations()
    stations_basin = stations[stations.geometry.within(AMAZON_BASIN['vector'].geometry.iloc[0])]
    intepolate_data_to_stations(stations_basin, recompute=True)
    get_climatologically_valid_data(recompute=True)