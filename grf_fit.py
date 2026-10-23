#%%
import os
from constants import  GRF_RESULTS_DIR, CLIMATOLOGY_SCORE_THRESHOLD, LST_CORRELATION_THRESHOLD, ELEVATION_THRESHOLD
from pygrf_modified import PyGRFBuilder
import pandas as pd
import pickle as pkl
from sklearn.utils.parallel import Parallel
from grf_data import get_climatologically_valid_data
from sklearn.model_selection import LeaveOneOut, KFold
from obs_data import define_local_azimuthal_projection
from ufuncs import get_amazon_basin_mask
import geopandas as gpd
import numpy as np

az_crs = define_local_azimuthal_projection(recalculate=False)
AMAZON_BASIN = get_amazon_basin_mask(buffer_km=0)['vector'].to_crs(az_crs)
N_JOBS_MODELS = 70
FEATURES = ['LST', 'elevation', 'doy_sin', 'doy_cos', 'day']


def fit_model(train_stations, model_type="rf", parallel=None):

    y_train = train_stations['tmax'].astype(float).values.reshape(-1, 1)
    X_train = train_stations[['LST', 'elevation', 'doy_sin', 'doy_cos', 'day']].values
    coords_train = train_stations[['easting', 'northing']].values



    if model_type == "rf":
        model_hyperparameters = {
            "n_estimators": 40,
            "min_samples_leaf": 13,
            "max_depth": 28,
            "max_features": 0.3,
            "bootstrap": True,
        }
        band_width = 40
    elif model_type == "xgb":
        model_hyperparameters = {
            "n_estimators": 487,
            "max_depth": 8,
            "learning_rate": 0.129,
            "subsample": 0.69,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
        }
        band_width = 27
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


    model = PyGRFBuilder(
            adaptive="adaptive",
            band_width=band_width,
            base_model=model_type,
            model_hyperparameters=model_hyperparameters,
            train_weighted=True,
            predict_weighted=True,
            bootstrap=True,
            random_state=42,
            n_jobs=4,
            n_jobs_models=N_JOBS_MODELS,
        )

    station_ids = train_stations.index.get_level_values('station_id').values

    model.fit(X_train, y_train, coords_train, location_ids=station_ids, parallel=parallel)
    
    return model

def pickle_fold_results(fold, predictions, model, model_type, sdir):
    spath = os.path.join(GRF_RESULTS_DIR, f'{model_type}/{sdir}/fold_{fold}_results.pkl')
    # ensure path exists
    os.makedirs(os.path.dirname(spath), exist_ok=True)

    results = {}
    results['predictions'] = predictions
    feature_importances = {}
    # local_models[i] is the model centred on model.station_ids[i]
    for station_id, local_model in zip(model.station_ids, model.local_models):
        feature_importances[station_id] = local_model.feature_importances_
    results['feature_importances'] = feature_importances
    results['features'] = FEATURES
    with open(spath, 'wb') as f:
        pkl.dump(results, f)

def test_feature_partial_dependencies(train_stations, model, parallel=None):

    station_ids = model.station_ids
    local_observation_indices = model.local_observation_indices

    # draw n samples from a normal distribution for each feature, with mean and std from the training data
    n_samples = 200
    n_intervals = 30

    # for each local model, draw n_samples from the observations that were
    # actually used to fit that model. Predictions are still evaluated at the
    # focal station coordinates.
    # dimensions will be (n_stations, n_samples, n_features+2)
    X_samples = np.zeros((len(station_ids), n_samples, len(FEATURES)+2))
    feat_llim = np.zeros((len(station_ids), len(FEATURES)))
    feat_ulim = np.zeros((len(station_ids), len(FEATURES)))
    for i, local_obs_idx in enumerate(local_observation_indices):
        local_data = train_stations.iloc[local_obs_idx]
        sample_rows = np.random.choice(len(local_data), size=n_samples, replace=True)
        X_samples[i, :, :-2] = local_data.iloc[sample_rows][FEATURES].values
        X_samples[i, :, -2:] = model.station_coords[i]
        feat_llim[i, :] = local_data[FEATURES].quantile(0.01).to_numpy()
        feat_ulim[i, :] = local_data[FEATURES].quantile(0.99).to_numpy()

    # for each feature, create a grid of n_intervals between the 1st and 99th percentile of the feature for each station_id
    # dimensions will be (n_stations, n_samples, n_intervals, n_features+2)
    feature_grids = {}
    feature_intervals = {}
    for j, feature in enumerate(FEATURES):
        X_test = np.repeat(
            X_samples[:, :, np.newaxis, :],
            n_intervals,
            axis=2
        )
        
        grids = np.stack([
            np.linspace(
                -1 if feature in ['doy_sin', 'doy_cos'] else feat_llim[i, j],
                1 if feature in ['doy_sin', 'doy_cos'] else feat_ulim[i, j],
                n_intervals
            )
            for i in range(len(station_ids))
        ])
        # Replace only the feature being evaluated
        X_test[:, :, :, j] = grids[:, np.newaxis, :]

        feature_grids[feature] = X_test
        feature_intervals[feature] = grids

    feature_responses = {}
    for i, feature in enumerate(FEATURES):
        X_test = feature_grids[feature]
        # reshape to combine first three dimensions into one, so that we can predict all samples at once
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        coords_test = X_test_reshaped[:, -2:]
        predictions = model.predict(X_test_reshaped[:, :-2], coords_test, parallel=parallel)

        # reshape predictions back into (nstations, n_samples, n_intervals)
        predictions = predictions.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])
        # take the mean over the samples axis
        predictions_mean = predictions.mean(axis=1)
        feature_responses[feature] = pd.DataFrame({
            'station_id': np.repeat(station_ids, n_intervals),
            'prediction': predictions_mean.reshape(-1)
        }, index=feature_intervals[feature].reshape(-1))
        feature_responses[feature].index.name = feature

    # Add a physically consistent day-of-year response by varying doy_sin and
    # doy_cos jointly along the unit circle.
    doy_intervals = np.linspace(1, 365, n_intervals)
    doy_angle = 2 * np.pi * doy_intervals / 365
    doy_sin_grid = np.sin(doy_angle)
    doy_cos_grid = np.cos(doy_angle)

    X_test_doy = np.repeat(
        X_samples[:, :, np.newaxis, :],
        n_intervals,
        axis=2
    )
    doy_sin_idx = FEATURES.index('doy_sin')
    doy_cos_idx = FEATURES.index('doy_cos')
    X_test_doy[:, :, :, doy_sin_idx] = doy_sin_grid[np.newaxis, np.newaxis, :]
    X_test_doy[:, :, :, doy_cos_idx] = doy_cos_grid[np.newaxis, np.newaxis, :]

    X_test_doy_reshaped = X_test_doy.reshape(-1, X_test_doy.shape[-1])
    coords_test_doy = X_test_doy_reshaped[:, -2:]
    predictions_doy = model.predict(X_test_doy_reshaped[:, :-2], coords_test_doy, parallel=parallel)
    predictions_doy = predictions_doy.reshape(X_test_doy.shape[0], X_test_doy.shape[1], X_test_doy.shape[2])
    predictions_doy_mean = predictions_doy.mean(axis=1)

    feature_responses['doy'] = pd.DataFrame({
        'station_id': np.repeat(station_ids, n_intervals),
        'prediction': predictions_doy_mean.reshape(-1)
    }, index=np.tile(doy_intervals, len(station_ids)))
    feature_responses['doy'].index.name = 'doy'

    return feature_responses


def kfold_cross_validation(rfd, sdir, model_type="rf", parallel=None):
    coord_df = rfd.groupby('station_id')['geometry'].first()
    stations_in_basin = coord_df[coord_df.geometry.within(AMAZON_BASIN.geometry.iloc[0])].index
    stations_in_basin = rfd.loc[stations_in_basin]
    stations_out_basin = coord_df[~coord_df.geometry.within(AMAZON_BASIN.geometry.iloc[0])].index
    stations_out_basin = rfd.loc[stations_out_basin]
    kf = LeaveOneOut()
    basin_ids = stations_in_basin.index.get_level_values('station_id')
    unique_basin_ids = basin_ids.unique()
    n_splits = len(unique_basin_ids)
    print(f"Leave-one-out over {n_splits} in-basin stations; {rfd.index.get_level_values('station_id').nunique() - n_splits} out-of-basin stations always in training.", flush=True)
    for fold, (train_idx, test_idx) in enumerate(kf.split(unique_basin_ids)):
        is_test = basin_ids.isin(unique_basin_ids[test_idx])
        test_stations = stations_in_basin[is_test]
        basin_train = stations_in_basin[~is_test]
        train_stations = pd.concat([basin_train, stations_out_basin])
        print(f"Fold {fold + 1}/{n_splits}: test station {unique_basin_ids[test_idx][0]} ({len(test_stations):,} obs), {len(train_stations):,} training obs...", flush=True)
        predictions = pd.DataFrame(index=test_stations.index, columns=['obs', 'pred'])  


        model = fit_model(train_stations, model_type=model_type, parallel=parallel)

        coords_test = test_stations[['easting', 'northing']].values
        X_test = test_stations[FEATURES].values
        y_test = test_stations['tmax'].astype(float).values.reshape(-1, 1)

        predictions.loc[test_stations.index, 'obs'] = y_test.flatten()
        model_predictions = model.predict(X_test, coords_test, parallel=parallel)
        predictions.loc[test_stations.index, 'pred'] = model_predictions

        rmse = np.sqrt(np.mean((model_predictions - y_test.flatten())**2))
        r2 = 1 - np.sum((model_predictions - y_test.flatten())**2) / np.sum((y_test.flatten() - np.mean(y_test.flatten()))**2)
        print(f"Fold {fold + 1}/{n_splits}: RMSE = {rmse:.4f}, R2 = {r2:.4f}", flush=True)

        pickle_fold_results(fold, predictions, model, model_type=model_type, sdir=sdir)

def add_aux_time_features(rfd):
    datetimes = rfd.index.get_level_values('datetime')
    rfd['doy_sin'] = np.sin(2 * np.pi * datetimes.dayofyear / 365)
    rfd['doy_cos'] = np.cos(2 * np.pi * datetimes.dayofyear / 365)
    rfd['day'] = (datetimes - datetimes.min()).days
    return rfd

def fit_evaluate_grf(df, sdir, model_type="rf"):

    rfd = df.copy()
    
    rfd = rfd.dropna(subset=['tmax', 'LST', 'elevation', 'easting', 'northing'])
    
    rfd = gpd.GeoDataFrame(rfd, 
                          geometry=gpd.points_from_xy(rfd['easting'], 
                                                      rfd['northing']), 
                                                      crs=az_crs)
    rfd = rfd.reset_index().set_index(['station_id', 'datetime'])
    rfd = add_aux_time_features(rfd)

    
    # One persistent thread pool is reused for every fold and the final fit.
    with Parallel(n_jobs=N_JOBS_MODELS, backend="threading") as parallel:
        kfold_cross_validation(rfd, sdir, model_type=model_type, parallel=parallel)
        print("Cross-validation complete; fitting final model on all stations...")

        # train final model on all data
        final_model = fit_model(rfd, model_type=model_type, parallel=parallel)
        feature_responses = test_feature_partial_dependencies(rfd, final_model, parallel)

    final_model_results = {'model': final_model, 'feature_responses': feature_responses}
    spath = os.path.join(GRF_RESULTS_DIR, f'{model_type}/{sdir}/grf_final_model.pkl')
    # ensure path exists
    os.makedirs(os.path.dirname(spath), exist_ok=True)
    with open(spath, 'wb') as f:
        pkl.dump(final_model_results, f)

    
#%%
if __name__ == '__main__':
    # df = get_climatologically_valid_data(
    #     climatology_score_threshold=CLIMATOLOGY_SCORE_THRESHOLD,
    #     lst_correlation_threshold=-np.inf,
    #     elevation_threshold=np.inf
    # )
    # fit_evaluate_grf(df, sdir='all_elev', model_type="rf")
    df = get_climatologically_valid_data(
        climatology_score_threshold=CLIMATOLOGY_SCORE_THRESHOLD,
        lst_correlation_threshold=-np.inf,
        elevation_threshold=np.inf
    )
    fit_evaluate_grf(df, sdir='all_elev', model_type="rf")
