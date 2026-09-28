#%%
#https://link.springer.com/article/10.1007/s10109-025-00481-4
import pandas as pd
import numpy as np
import geopandas as gpd
import xarray as xr
import pickle as pkl
from mgwr_data import (get_amazon_basin_mask, load_meteostat_stations, 
                       load_meteostat_data, doy_gaussian_climatology,
                       define_local_azimuthal_projection)
from constants import MGWR_RESULTS_DIR, LST_DIR, STORAGE_DIR
from scipy.spatial import cKDTree
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score


def predict_tmax(results, station_xy, target_xy, target_lst,
                 target_elev, k=3,):
    """
    station_xy : ndarray, shape (n_stations, 2)
        Projected x/y coordinates of the MGWR training stations.

    target_xy : ndarray, shape (n_targets, 2)
        Projected x/y coordinates of prediction locations.

    target_lst : ndarray, shape (n_targets,)
        LST at the prediction locations.

    k : int, default=3
        Number of nearest stations used for coefficient extrapolation.

    Returns
    -------
    tmax : ndarray
        Predicted TMAX.

    beta0_target : ndarray
        Extrapolated intercept.

    beta_lst_target : ndarray
        Extrapolated LST coefficient.
    """

    station_xy = np.asarray(station_xy)
    target_xy = np.asarray(target_xy)
    target_lst = np.asarray(target_lst)
    target_elev = np.asarray(target_elev)  

    # MGWR coefficients at the observed stations
    beta0 = results.params[:, 0]
    beta_lst = results.params[:, 1]
    beta_elev = results.params[:, 2]

    # Find k nearest stations
    tree = cKDTree(station_xy)

    distances, indices = tree.query(
        target_xy,
        k=k
    )

    # Adaptive bandwidth = distance to kth nearest station
    bandwidth = distances[:, -1]

    # Gaussian weights
    weights = np.exp(-distances**2 / (2 * bandwidth[:, None]**2))

    # Normalise
    weights /= weights.sum(axis=1, keepdims=True)

    # Extrapolate coefficients
    beta0_target = np.sum(weights * beta0[indices], axis=1)
    beta_elev_target = np.sum(weights * beta_elev[indices], axis=1)
    beta_lst_target = np.sum( weights * beta_lst[indices],  axis=1)

    # Evaluate regression at target location
    tmax = ( beta0_target + beta_lst_target * target_lst + beta_elev_target * target_elev)

    return tmax, beta0_target, beta_lst_target, beta_elev_target

AMAZON_BASIN = get_amazon_basin_mask(buffer_km=800)
stations = load_meteostat_stations()
stations_basin = stations[stations.geometry.within(AMAZON_BASIN['vector'].geometry.iloc[0])]
meteo = load_meteostat_data(stations_basin)

lst = xr.open_mfdataset(LST_DIR+'interped/lst_interped_obs_*.zarr', 
                        concat_dim='year', combine='nested')['LST']
lst = lst.assign_coords({'year':range(2003, 2021), 'doy':range(1, 366)})
lst_clim = doy_gaussian_climatology(window=10, sigma=None)
elev = xr.open_zarr(STORAGE_DIR+'elevation/elev_interped_obs.zarr')['elevation']
az_crs = define_local_azimuthal_projection(recalculate=False)
mdf_az_proj = stations_basin.to_crs(az_crs)
mdf_az_proj['easting'] = mdf_az_proj.geometry.x
mdf_az_proj['northing'] = mdf_az_proj.geometry.y

#%%

def get_anomaly_df(date, meteo, lst, lst_clim, elev, mdf_az_proj):
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
    df = df.dropna(subset=['tmax', 'lst_clim', 'elev', 'easting', 'northing',
                           'lst', 'lst_anomaly'])


    # get the model-predicted tmax
    with open(f'{MGWR_RESULTS_DIR}/layer1/mgwr_model_doy_{doy}.pkl', 'rb') as f:
        layer1 = pkl.load(f)

        results = layer1['results']
        ldf = layer1['df']
        station_xy = ldf[['easting', 'northing']].values
        target_xy = df[['easting', 'northing']].values
        target_lst = df['lst_clim'].values
        target_elev = df['elev'].values

    tmax_clim_pred, beta0_target, beta_lst_target, beta_elev_target = predict_tmax(
        results, station_xy, target_xy, target_lst, target_elev, k=3
    )

    df['tmax_clim'] = tmax_clim_pred
    df['tmax_anomaly'] = df['tmax'] - df['tmax_clim']
    return df

AMAZON_BASIN = get_amazon_basin_mask()['vector'].to_crs(az_crs)

for date in meteo.index[-100:]:
        

    df = get_anomaly_df(date, meteo, lst, lst_clim, elev, mdf_az_proj)
    if len(df) < 50:
        print(f'Not enough data for {date}, skipping...')
        continue

    

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.easting, df.northing), crs=az_crs)
    stations_in_basin = gdf[gdf.geometry.within(AMAZON_BASIN.geometry.iloc[0])]
    stations_out_basin = gdf[~gdf.geometry.within(AMAZON_BASIN.geometry.iloc[0])]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

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

    print(
        f'{date}: '
        f'RMSE = {rmse:.2f}, '
        f'R² = {r2:.2f}'
    )

    