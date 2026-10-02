#%%
import os
from constants import (
    STORAGE_DIR,
    LST_DIR,
    MGWR_DATA_DIR,
    DATE_SLICE,
    DOY_POOL_WINDOW,
    CLIMATOLOGY_SCORE_THRESHOLD
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




def get_climatologically_valid_data(climatology_score_threshold=CLIMATOLOGY_SCORE_THRESHOLD, min_valid_years=10):

    tmax_df = get_interpolated_data(recompute=False)

    station_doy_count = tmax_df.groupby([tmax_df.station_id, tmax_df.doy])['tmax'].count()
    station_doy_mean = station_doy_count.groupby(level='station_id').mean()
    station_doy_std  = station_doy_count.groupby(level='station_id').std()
    score = station_doy_mean / np.sqrt(station_doy_std)
    score = score.sort_values(ascending=False)

    valid_stations = score[score >= climatology_score_threshold].index.get_level_values(0).unique()

    df = tmax_df[tmax_df['station_id'].isin(valid_stations)]
    return df

def interpolate_lst_to_stations(mdf):

    station = xr.DataArray(mdf['station_id'].values, dims='station')
    lat = xr.DataArray(mdf['lat'].values, dims='station', coords={'station': station})
    lon = xr.DataArray(mdf['lon'].values, dims='station', coords={'station': station})


    for year in range(2003, 2021):
        lst = xr.open_zarr(LST_DIR+f'zz_lst_{year}.zarr')['LST'] / 10


        lst_interped = lst.interp(lon=lon, lat=lat, method='linear')
        lst_interped.to_zarr(LST_DIR+f'interped/lst_interped_obs_{year}.zarr', mode='w')

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



def calc_lapse_rate(df):
    df = df.dropna(subset=['tmax', 'LST', 'elevation'])

    model = smf.ols(
        'tmax ~ elevation + cc(doy, df=10)',
        data=df
    ).fit()

    lapse_rate = model.params['elevation']
    return lapse_rate

def get_interpolated_data(recompute=False):

    if not recompute and os.path.exists(MGWR_DATA_DIR+'mgwr_climatological_long.parquet'):
        print("Loading climatologically valid data from cache...")
        df = pd.read_parquet(MGWR_DATA_DIR+'mgwr_climatological_long.parquet')
        return df
    print("Computing climatologically valid data...")

    tmax_df, stations, az_proj = get_harmonised_data()


    # load the tmax data for the valid stations
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
    ds = ds.loc[{'station_id':stations['station_id'].values}]
    # add northing and easting
    ds = ds.assign_coords({'easting': (('station_id'), stations['geometry'].x.values),
                            'northing': (('station_id'), stations['geometry'].y.values)})


    ds = ds.stack(time=('year', 'doy')).drop_vars(['band', 'spatial_ref'])
    ds = ds.drop_vars(['time', 'year', 'doy', 'lat', 'lon'])
    ds.coords['time'] = tindex
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

def doy_gaussian_mean(df, target_doy, window=DOY_POOL_WINDOW, sigma=None, n_days_per_year=365):
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


def doy_gaussian_climatology(window=DOY_POOL_WINDOW, sigma=None, recompute=False):

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


# with local_cluster(n_workers=4, threads_per_worker=16, memory_limit='50GB',
#                     suppress_dask_warnings=False) as cluster:
#     doy_gaussian_climatology(window=DOY_POOL_WINDOW, sigma=None, recompute=True)

#%%
#%% interpolate data to station_locations
if __name__ == '__main__':

    stations_basin, mdf, crs = get_harmonised_data(recompute=False)
    intepolate_data_to_stations(mdf, recompute=False)
    get_interpolated_data(recompute=False)