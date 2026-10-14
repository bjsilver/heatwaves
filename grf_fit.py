#%%
import os

from sklearn.utils import parallel
from constants import DOY_POOL_WINDOW, GRF_RESULTS_DIR, CLIMATOLOGY_SCORE_THRESHOLD, LST_CORRELATION_THRESHOLD
from pygrf_modified import PyGRFBuilder
import pandas as pd
import pickle as pkl
from joblib import delayed
from sklearn.utils.parallel import Parallel
from mgwr_data import doy_gaussian_mean, get_climatologically_valid_data, get_interpolated_data, get_harmonised_data
from sklearn.model_selection import LeaveOneOut, KFold
from obs_data import define_local_azimuthal_projection
from ufuncs import get_amazon_basin_mask
import geopandas as gpd
import numpy as np

CRS = define_local_azimuthal_projection(recalculate=False)
AMAZON_BASIN = get_amazon_basin_mask(buffer_km=0)['vector'].to_crs(CRS)
N_JOBS_MODELS = 70
FEATURES = ['LST', 'elevation', 'doy_sin', 'doy_cos', 'day']


def fit_model(train_stations, parallel=None):

    y_train = train_stations['tmax'].astype(float).values.reshape(-1, 1)
    X_train = train_stations[['LST', 'elevation', 'doy_sin', 'doy_cos', 'day']].values
    coords_train = train_stations[['easting', 'northing']].values


    model = PyGRFBuilder(
            n_estimators=30,
            adaptive="adaptive",
            band_width=25,
            min_samples_leaf=5,
            max_depth=20,
            max_features=.6,
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

def pickle_fold_results(fold, predictions, model, sdir):
    spath = os.path.join(GRF_RESULTS_DIR, f'{sdir}/fold_{fold}_results.pkl')
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

#%% test feature partial dependencies
def test_feature_partial_dependencies(train_stations, model, parallel=None):

    feat_ulim = train_stations.groupby('station_id')[FEATURES].quantile(0.99)
    feat_llim = train_stations.groupby('station_id')[FEATURES].quantile(0.01)
    station_ids = train_stations.index.get_level_values('station_id').unique()

    # draw n samples from a normal distribution for each feature, with mean and std from the training data
    n_samples = 100
    n_intervals = 40

    # for each station_id, draw n_samples (random rows of train_stations) from the train_station data
    # also include the coordinates (easting, northing) of the station_id in the samples
    # dimensions will be (n_stations, n_samples, n_features+2)
    X_samples = np.zeros((len(station_ids), n_samples, len(FEATURES)+2))
    for i, station_id in enumerate(station_ids):
        station_data = train_stations.loc[station_id]
        sample_indices = np.random.choice(station_data.index, size=n_samples, replace=True)
        X_samples[i, :, :-2] = station_data.loc[sample_indices, FEATURES].values
        X_samples[i, :, -2:] = station_data[['easting', 'northing']].values[0]

    # then for each feature, create a grid of n_intervals between the 1st and 99th percentile of the feature for each station_id
    # dimensions will be (n_stations, n_intervals, n_features+2)
    X_grid = np.zeros((len(station_ids), n_intervals, len(FEATURES)+2))
    for i, station_id in enumerate(station_ids):
        station_data = train_stations.loc[station_id]
        for j, feature in enumerate(FEATURES):
            feat_min = station_data[feature].quantile(0.01)
            feat_max = station_data[feature].quantile(0.99)
            X_grid[i, :, j] = np.linspace(feat_min, feat_max, n_intervals)
        # fill in the coordinates (easting, northing) of the station_id in the last two columns
        X_grid[i, :, -2:] = station_data[['easting', 'northing']].values[0]

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
                -1 if feature in ['doy_sin', 'doy_cos'] else feat_llim.loc[sid, feature],
                1 if feature in ['doy_sin', 'doy_cos'] else feat_ulim.loc[sid, feature],
                n_intervals
            )
            for sid in station_ids
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

    return feature_responses

#%%

def kfold_cross_validation(rfd, sdir, parallel=None):
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


        model = fit_model(train_stations, parallel)

        coords_test = test_stations[['easting', 'northing']].values
        X_test = test_stations[FEATURES].values
        y_test = test_stations['tmax'].astype(float).values.reshape(-1, 1)

        predictions.loc[test_stations.index, 'obs'] = y_test.flatten()
        model_predictions = model.predict(X_test, coords_test, parallel=parallel)
        predictions.loc[test_stations.index, 'pred'] = model_predictions

        rmse = np.sqrt(np.mean((model_predictions - y_test.flatten())**2))
        r2 = 1 - np.sum((model_predictions - y_test.flatten())**2) / np.sum((y_test.flatten() - np.mean(y_test.flatten()))**2)
        print(f"Fold {fold + 1}/{n_splits}: RMSE = {rmse:.4f}, R2 = {r2:.4f}", flush=True)

        pickle_fold_results(fold, predictions, model, sdir)


def fit_evaluate_grf(df, sdir):

    rfd = df.copy()
    
    rfd = rfd.dropna(subset=['tmax', 'LST', 'elevation', 'easting', 'northing'])
    datetimes = rfd.index
    rfd = gpd.GeoDataFrame(rfd, 
                          geometry=gpd.points_from_xy(rfd['easting'], 
                                                      rfd['northing']), 
                                                      crs=CRS)
    rfd = rfd.reset_index().set_index(['station_id', 'datetime'])

    rfd['doy_sin'] = np.sin(2 * np.pi * datetimes.dayofyear / 365)
    rfd['doy_cos'] = np.cos(2 * np.pi * datetimes.dayofyear / 365)
    rfd['day'] = (datetimes - datetimes.min()).days
    
    # One persistent thread pool is reused for every fold and the final fit.
    with Parallel(n_jobs=N_JOBS_MODELS, backend="threading") as parallel:
        # kfold_cross_validation(rfd, sdir, parallel)
        print("Cross-validation complete; fitting final model on all stations...")

        # train final model on all data
        final_model = fit_model(rfd, parallel)
        feature_responses = test_feature_partial_dependencies(rfd, final_model, parallel)

    final_model_results = {'model': final_model, 'feature_responses': feature_responses}
    spath = os.path.join(GRF_RESULTS_DIR, f'{sdir}/grf_final_model.pkl')
    # ensure path exists
    os.makedirs(os.path.dirname(spath), exist_ok=True)
    with open(spath, 'wb') as f:
        pkl.dump(final_model_results, f)

    
#%%
if __name__ == '__main__':
    df = get_climatologically_valid_data()
    fit_evaluate_grf(df, sdir='climvalid_obs')
