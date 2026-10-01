#%%
import os
from constants import DOY_POOL_WINDOW, MGWR_RESULTS_DIR
import numpy as np
from mgwr.gwr import MGWR
from mgwr.sel_bw import Sel_BW
import pandas as pd
import pickle as pkl
from joblib import Parallel, delayed
from mgwr_data import doy_gaussian_mean, get_climatologically_valid_data, get_interpolated_data, get_harmonised_data
from sklearn.model_selection import KFold, train_test_split
from obs_data import define_local_azimuthal_projection
from ufuncs import get_amazon_basin_mask
import geopandas as gpd
from scipy.spatial import cKDTree

CRS = define_local_azimuthal_projection(recalculate=False)
AMAZON_BASIN = get_amazon_basin_mask(buffer_km=0)['vector'].to_crs(CRS)


def predict_tmax(results, station_xy, target_xy, target_lst,
                 target_elev, k=5,):
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

def fit_model(coords_train, y_train, X_train):
    selector = Sel_BW(coords_train, y_train, X_train, multi=True, constant=True,
                            n_jobs=1)
    selector.search(bw_min=2)

    model = MGWR(coords_train, y_train, X_train, selector, n_jobs=1,
                constant=True)
    results = model.fit()
    return selector, model, results

def kfold_cross_validation(doy_set, n_splits=10):
    stations_in_basin = doy_set[doy_set.geometry.within(AMAZON_BASIN.geometry.iloc[0])]
    stations_out_basin = doy_set[~doy_set.geometry.within(AMAZON_BASIN.geometry.iloc[0])]

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    predictions = pd.DataFrame(index=stations_in_basin.index, columns=['obs', 'pred'], dtype=float)
    for fold, (train_idx, test_idx) in enumerate(kf.split(stations_in_basin)):

        test_stations = stations_in_basin.iloc[test_idx]
        basin_train = stations_in_basin.iloc[train_idx]
        train_stations = pd.concat([basin_train, stations_out_basin])

        y_train = train_stations['tmax'].astype(float).values.reshape(-1, 1)
        X_train = train_stations[['LST', 'elevation']].values
        coords_train = train_stations[['easting', 'northing']].values
    
        _, _, results = fit_model(coords_train, y_train, X_train)

        coords_test = test_stations[['easting', 'northing']].values
        X_test = test_stations[['LST', 'elevation']].values
        y_test = test_stations['tmax'].astype(float).values.reshape(-1, 1)

        predictions.loc[test_stations.index, 'obs'] = y_test.flatten()
        tmax, _, _, _ = predict_tmax(results, coords_train, coords_test, X_test[:, 0], X_test[:, 1], k=5)
        predictions.loc[test_stations.index, 'pred'] = tmax

    return predictions

def fit_layer1_model(df, target_doy, all_df):
    doy_set = doy_gaussian_mean(
        df,
        target_doy,
        window=DOY_POOL_WINDOW,
        n_days_per_year=365
    )

    doy_set = doy_set.dropna(subset=['tmax', 'LST', 'elevation', 'easting', 'northing'])
    doy_set = gpd.GeoDataFrame(doy_set, geometry=gpd.points_from_xy(doy_set['easting'], doy_set['northing']), crs=CRS)
    
    predictions = kfold_cross_validation(doy_set, n_splits=10)

    # do a final fit on all stations in and outside basin, and save the model and results
    y_train = doy_set['tmax'].astype(float).values.reshape(-1, 1)
    X_train = doy_set[['LST', 'elevation']].values
    coords_train = doy_set[['easting', 'northing']].values

    selector, model, results = fit_model(coords_train, y_train, X_train)


    def predict_tmax_doy_all(all_df, target_doy, results):
        # use the final model to predict tmax at the target coordinates
        # on stations in that basin that aren't climatologically valid
        doy_all = all_df[all_df['doy'] == target_doy]
        doy_all = doy_all.loc[~doy_all.station_id.isin(doy_set.station_id)]
        doy_all = gpd.GeoDataFrame(doy_all, geometry=gpd.points_from_xy(doy_all['easting'], doy_all['northing']), crs=CRS)
        doy_all = doy_all[doy_all.geometry.within(AMAZON_BASIN.geometry.iloc[0])]

        coords_all = doy_all[['easting', 'northing']].values
        lst_all = doy_all['LST'].values
        elev_all = doy_all['elevation'].values

        tmax_all, _, _, _ = predict_tmax(results, coords_train, coords_all, lst_all, elev_all, k=5)
        cdf = pd.DataFrame({
            'station_id': doy_all['station_id'],
            'tmax_pred': tmax_all,
            'tmax_obs': doy_all['tmax'],},
            index=doy_all.index
        )

        return cdf

    # predict tmax at all stations in the basin that weren't used in model training
    # i.e. these models weren't used because the weren't in the climatologically valid set, but we can still predict tmax at those stations using the model fitted on the valid stations
    cdf = predict_tmax_doy_all(all_df, target_doy, results)

    # pickle the selector, model and results
    with open(f'{MGWR_RESULTS_DIR}layer1/mgwr_model_doy_{target_doy}.pkl', 'wb') as f:
        pkl.dump({
            'k-fold_predictions': predictions,
            'selector': selector,
            'model': model,
            'results': results,
            'df': doy_set,
            'unseen_data_predictions': cdf
        }, f)

    print(f'MGWR model for DOY {target_doy} fitted and saved to {MGWR_RESULTS_DIR}layer1/mgwr_model_doy_{target_doy}.pkl')
    return target_doy

#%%

def fit_all_layer1_models():
    df = get_climatologically_valid_data()
    _, meta, _ = get_harmonised_data(recompute=False)
    all_df = get_interpolated_data(recompute=False)
    predict_coords_xy = meta.geometry.apply(lambda p: (p.x, p.y)).to_list()
    target_doys = range(1, 366)
    Parallel(n_jobs=64)(delayed(fit_layer1_model)(df, doy, all_df) for doy in target_doys)



#%%
if __name__ == '__main__':
    fit_all_layer1_models()


