#%%
#https://link.springer.com/article/10.1007/s10109-025-00481-4
import pandas as pd
import numpy as np
import geopandas as gpd
import xarray as xr
import pickle as pkl
from mgwr_data import (get_amazon_basin_mask,
                        doy_gaussian_climatology,
                        get_interpolated_data)
from obs_data import define_local_azimuthal_projection, get_harmonised_data
from mgwr_evaluation import retrieve_pickled_results
from mgwr_estimate import predict_tmax
from isd_download import load_isd_amazon_metadata, get_isd_tmax
from constants import MGWR_RESULTS_DIR, LST_DIR, STORAGE_DIR
from scipy.spatial import cKDTree
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import os
from dask.diagnostics import ProgressBar



def load_data_with_anomalies(recompute=False):
    cache_dir = STORAGE_DIR + 'mgwr_results/cached_dfs/'
    if not recompute and os.path.exists(cache_dir + 'mgwr_results_with_anomalies.pkl'):
        with open(cache_dir + 'mgwr_results_with_anomalies.pkl', 'rb') as f:
            df = pkl.load(f)
        return df

    results = retrieve_pickled_results()
    
    _, meta, _ = get_harmonised_data()
    meta = meta.set_index('station_id')

    tindex = xr.date_range(start='2003-01-01', end='2020-12-31', freq='D', calendar='noleap',
                        use_cftime=True)
    lst = xr.open_mfdataset(LST_DIR+'interped/lst_interped_obs_*.zarr', 
                            concat_dim='year', combine='nested')['LST']
    lst = lst.assign_coords({'year':range(2003, 2021), 'doy':range(1, 366)})
    lst_clim = doy_gaussian_climatology(window=10, sigma=None)
    # create year dimension with repeated values for each day of year
    lst_clim = lst_clim.expand_dims({'year': range(2003, 2021)}, axis=0)
    lst_clim = lst_clim.drop(['lat', 'lon'])
    # stack year and doy into a single time dimension with tindex as coords
    lst_clim = lst_clim.stack(time=('year', 'doy')).drop_vars(['year', 'doy'])
    lst_clim= lst_clim.assign_coords({'time': tindex})
    with ProgressBar():
        lst_clim = lst_clim.compute()
    # stack into single multiindex
    # lst_clim = lst_clim.stack(mindex=('time', 'station'))

    elev = xr.open_zarr(STORAGE_DIR+'elevation/elev_interped_obs.zarr')['elevation']
    # az_crs = define_local_azimuthal_projection(recalculate=False)
    # mdf_az_proj = stations_basin.to_crs(az_crs)
    # mdf_az_proj['easting'] = mdf_az_proj.geometry.x
    # mdf_az_proj['northing'] = mdf_az_proj.geometry.y

    df = get_interpolated_data(recompute=False)
    df = df.rename({'LST':'lst', 'elevation':'elev'}, axis=1)
    df = df.reset_index().set_index(['datetime', 'station_id'])
    df = df.dropna()
    lst_clim_df = lst_clim.to_pandas().T
    lst_clim_df.index = pd.DatetimeIndex([pd.Timestamp(i.strftime()) for i in  lst_clim_df.index])
    lst_clim_sr = lst_clim_df.stack()
    lst_clim_sr.name = 'lst_clim'
    lst_clim_sr.index.names = ['datetime', 'station_id']
    df = df.join(lst_clim_sr, how='left')

    df = df.reset_index(level='station_id')
    df['northing'] = meta.loc[df['station_id']]['geometry'].y.values
    df['easting'] = meta.loc[df['station_id']]['geometry'].x.values
    df['lst_anomaly'] = df['lst'] - df['lst_clim']
    df = df.dropna(subset=['tmax', 'lst_clim', 'elev', 'easting', 'northing', 'lst', 'lst_anomaly'])
    for doy in range(1, 366):
        with open(f'{MGWR_RESULTS_DIR}/layer1/mgwr_model_doy_{doy}.pkl', 'rb') as f:
            layer1 = pkl.load(f)
            results = layer1['results']
            ldf = layer1['df']
            station_xy = ldf[['easting', 'northing']].values
            target_xy = df[df['doy'] == doy][['easting', 'northing']].values
            target_lst = df[df['doy'] == doy]['lst_clim'].values
            target_elev = df[df['doy'] == doy]['elev'].values

            tmax_clim_pred, beta0_target, beta_lst_target, beta_elev_target = predict_tmax(
                results, station_xy, target_xy, target_lst, target_elev, k=3
            )

            df.loc[df['doy'] == doy, 'tmax_clim'] = tmax_clim_pred

    df['tmax_anomaly'] = df['tmax'] - df['tmax_clim']

    # df.index = df.reset_index()['index'].apply(lambda i: pd.Timestamp(i.strftime()))

    # cache the df 
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_dir + 'mgwr_results_with_anomalies.pkl', 'wb') as f:
        pkl.dump(df, f)

    return df

df = load_data_with_anomalies(recompute=False)

#%%
# calculate the correlation coefficient between tmax_anomaly and lst_anomaly for each station
def calculate_correlation(df):
    corr_df = df.groupby('station_id').apply(lambda x: x['tmax_anomaly'].corr(x['lst_anomaly']))
    corr_df.name = 'corr'
    return corr_df

def filter_stations(df):


    count = df.groupby('station_id')['tmax'].count()


    df = df.reset_index().set_index(['datetime', 'station_id'])
    # drop where count < 10
    print(f'Dropping {len(count[count < 10])} stations with less than 10 observations')
    df = df[df.index.get_level_values('station_id').isin(count[count >= 10].index)]
    corr_df = calculate_correlation(df)

    # drop where correlation is < .1
    print(f'Dropping {len(corr_df[corr_df < 0.1])} stations with correlation < 0.1' )
    df = df[df.index.get_level_values('station_id').isin(corr_df[corr_df >= 0.1].index)]
    return df

#%%
def train_layer2():

    df = load_data_with_anomalies(recompute=False)
    df = filter_stations(df)
    az_crs = define_local_azimuthal_projection(recalculate=False)

    AMAZON_BASIN = get_amazon_basin_mask()['vector'].to_crs(az_crs)

    dates = df.index.unique()

    for date in dates:
        doy_df = df[df.index == date]



        gdf = gpd.GeoDataFrame(doy_df, geometry=gpd.points_from_xy(doy_df.easting, doy_df.northing), crs=az_crs)
        gdf.index.name = 'datetime'
        gdf = gdf.reset_index().set_index(['datetime', 'station_id'])
        stations_in_basin = gdf[gdf.geometry.within(AMAZON_BASIN.geometry.iloc[0])]
        stations_out_basin = gdf[~gdf.geometry.within(AMAZON_BASIN.geometry.iloc[0])]

        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        predictions = pd.Series(index=stations_in_basin.index, dtype=float)
        for fold, (train_idx, test_idx) in enumerate(kf.split(stations_in_basin)):

            test_stations = stations_in_basin.iloc[test_idx]

            basin_train = stations_in_basin.iloc[train_idx]
            basin_test = stations_in_basin.iloc[test_idx]
            train_stations = pd.concat([basin_train, stations_out_basin])

            y_train = train_stations['tmax_anomaly'].astype(float).values.reshape(-1, 1)
            X_train = train_stations[['lst_anomaly']].values
            coords_train = train_stations[['easting', 'northing']].values

            sel_bw = Sel_BW(coords_train, y_train, X_train, multi=False, constant=True, n_jobs=1)

            bw = sel_bw.search(bw_min=2)
            model = GWR(coords_train, y_train, X_train, bw, constant=True, n_jobs=1)
            results = model.fit()

            coords_test = test_stations[['easting', 'northing']].values
            X_test = test_stations[['lst_anomaly']].values
            y_test = test_stations['tmax_anomaly'].astype(float).values.reshape(-1, 1)

            pred_results = model.predict(coords_test, X_test)
            y_pred = pred_results.predictions.flatten()
            predictions.loc[basin_test.index] = y_pred


        y_obs = stations_in_basin['tmax_anomaly'].astype(float).values

        y_pred = predictions.values

        rmse = np.sqrt(
            mean_squared_error(
                y_obs,
                y_pred
            )
        )

        r2 = r2_score(
            y_obs,
            y_pred
        )

        mae = np.mean(np.abs(y_obs - y_pred))

        # print(
        #     f'Day {doy}: '
        #     f'RMSE = {rmse:.2f}, '
        #     f'R² = {r2:.2f}',
        #     f'MAE = {mae:.2f}'
        # )


    
    