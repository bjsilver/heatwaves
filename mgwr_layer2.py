#%%
from pygrf_modified import PyGRFBuilder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import geopandas as gpd
from mgwr_data import  get_amazon_basin_mask, doy_gaussian_climatology, get_interpolated_data
from mgwr_evaluation_layer1 import interpolate_zz_to_stations, merge_zz_into_df, retrieve_pickled_results
from mgwr_layer1 import predict_tmax
from obs_data import define_local_azimuthal_projection, get_harmonised_data
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from constants import STORAGE_DIR, LST_DIR, DOY_POOL_WINDOW, MGWR_RESULTS_DIR
import pickle as pkl
import os
import xarray as xr
from dask.diagnostics import ProgressBar

az_crs = define_local_azimuthal_projection(recalculate=False)
AMAZON_MASK = get_amazon_basin_mask()['vector'].to_crs(az_crs)


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
    lst_clim = doy_gaussian_climatology(window=DOY_POOL_WINDOW, sigma=None)
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


def get_grf_input_data():

    df = load_data_with_anomalies(recompute=False)
    df = filter_stations(df)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.easting, df.northing), crs="EPSG:27700")
    # gdf.plot(column='tmax_anomaly', cmap='coolwarm', legend=True, figsize=(10, 8))


    gdf = gdf.reset_index(level='station_id')

    zz = interpolate_zz_to_stations(gdf, recompute=False)
    gdf = gdf.reset_index().set_index(['datetime', 'station_id'])
    gdf = merge_zz_into_df(gdf, zz)
    gdf = gdf.reset_index('station_id')

    return gdf

def print_eval_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    std = np.std(y_pred)

    print(f"RMSE = {rmse:.2f}")
    print(f"R² = {r2:.2f}")
    print(f"Prediction SD = {std:.2f}")




# global model without weighting

def get_grf_data():
    gdf = get_grf_input_data()
    gdf['day'] = [d.days for d in (gdf.index - gdf.index.min())]
    gdf["doy_sin"] = np.sin(2 * np.pi * gdf["doy"] / 365)
    gdf["doy_cos"] = np.cos(2 * np.pi * gdf["doy"] / 365)

    xys = gdf.groupby(gdf.station_id).first()[['easting', 'northing']]
    xys = gpd.GeoDataFrame(xys, geometry=gpd.points_from_xy(xys.easting, xys.northing), crs=az_crs)
    basin_stations = xys[xys.within(AMAZON_MASK.geometry.iloc[0])].index
    outside_stations = xys[~xys.within(AMAZON_MASK.geometry.iloc[0])].index
    gdf_basin = gdf[gdf.station_id.isin(basin_stations)]
    gdf_outside = gdf[gdf.station_id.isin(outside_stations)]

    basin_station_ids = gdf_basin["station_id"].unique()
    return gdf_basin, gdf_outside, basin_station_ids

# test single random forest at single station
def test_single_station_rf(gdf_basin, basin_station_ids):
    sdf = gdf_basin[gdf_basin.station_id == basin_station_ids[100]]
    plt.scatter(sdf['lst_anomaly'], sdf['tmax_anomaly'], label='Basin', alpha=0.7)
    print(f'correlation coefficient: {sdf["lst_anomaly"].corr(sdf["tmax_anomaly"])}')
    # calculate day of timeseries



    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    predictions =np.full((len(sdf), 2), np.nan)
    for fold, (train_idx, test_idx) in enumerate(kf.split(sdf)):
        train_df = sdf.iloc[train_idx]
        test_df = sdf.iloc[test_idx]

        X_train = train_df[["lst_anomaly", 'elev', 'day', 'doy_sin', 'doy_cos']].to_numpy()
        y_train = train_df["tmax_anomaly"].astype(float).to_numpy()

        X_test = test_df[["lst_anomaly", 'elev', 'day', 'doy_sin', 'doy_cos']].to_numpy()
        y_test = test_df["tmax_anomaly"].astype(float).to_numpy()

        rf = RandomForestRegressor(n_estimators=50, random_state=42,
                                bootstrap=True,
                                min_samples_leaf=5, n_jobs=10,
                                max_depth=10)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        predictions[test_idx, 0] = y_pred
        predictions[test_idx, 1] = y_test

    # calc and print error metrics
    y_true = predictions[:, 1]
    y_pred = predictions[:, 0]
    print_eval_metrics(y_true, y_pred)
    # print feature importances
    importances = rf.feature_importances_
    for i, col in enumerate(["lst_anomaly", 'elev', 'day', 'doy_sin', 'doy_cos']):
        print(f"Feature importance of {col}: {importances[i]:.4f}")



def train_grf_model(gdf_basin, gdf_outside, basin_station_ids,
                    n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    predictions = pd.DataFrame(
        index=pd.MultiIndex.from_arrays(
            [gdf_basin['station_id'], gdf_basin.index],
            names=['station_id', 'datetime'],
        ),
        columns=['obs_resid', 'pred_resid', 'pred_zz', 'tmax_clim', 'tmax'],
        dtype=float,
    )
    results = {}
    results['folds'] = {}
    for fold, (train_idx, test_idx) in enumerate(kf.split(basin_station_ids)):
        print(f"Training fold {fold + 1}/{n_splits}...")
        train_station_ids = basin_station_ids[train_idx]
        test_station_ids = basin_station_ids[test_idx]

        basin_train = gdf_basin[
            gdf_basin.station_id.isin(train_station_ids)
        ]
        basin_test_mask = gdf_basin.station_id.isin(test_station_ids)
        basin_test = gdf_basin.loc[basin_test_mask]

        # All outside-basin observations remain in training
        train_df = pd.concat([basin_train, gdf_outside])

        X_train = train_df[["lst_anomaly", 'elev', 'day', 'doy_sin', 'doy_cos']].to_numpy()
        y_train = train_df["tmax_anomaly"].astype(float).to_numpy()
        xy_train = train_df[["easting", "northing"]].to_numpy()

        X_test = basin_test[["lst_anomaly", 'elev', 'day', 'doy_sin', 'doy_cos']].to_numpy()
        xy_test = basin_test[["easting", "northing"]].to_numpy()
        y_test = basin_test["tmax_anomaly"].astype(float).to_numpy()

        model = PyGRFBuilder(
            n_estimators=60,
            adaptive="adaptive",
            band_width=20,
            min_samples_leaf=5,
            max_depth=12,
            train_weighted=True,
            predict_weighted=True,
            bootstrap=True,
            random_state=42,
            n_jobs=15,
        )

        model.fit(X_train, y_train, xy_train)

        y_pred = np.asarray(model.predict(X_test, xy_test)).reshape(-1)

        # Assign predictions to the held-out rows
        test_positions = basin_test_mask.to_numpy()
        predictions.loc[test_positions, 'obs_resid'] = y_test
        predictions.loc[test_positions, 'pred_resid'] = y_pred
        predictions.loc[test_positions, 'pred_zz'] = basin_test['tmax_pred_zz'].to_numpy()
        predictions.loc[test_positions, 'tmax_clim'] = basin_test['tmax_clim'].to_numpy()
        predictions.loc[test_positions, 'tmax'] = basin_test['tmax'].to_numpy()

        fold_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        fold_r2 = r2_score(y_test, y_pred)

        print(
            f"Fold {fold}: {len(test_station_ids)} held-out stations, "
            f"RMSE = {fold_rmse:.2f}, R² = {fold_r2:.2f}"
        )

        fis = []
        for model in model.local_models:
            fis.append(model.feature_importances_)
        fi = pd.DataFrame(fis, columns=["lst_anomaly", 'elev', 'day', 'doy_sin', 'doy_cos'])

        mean_fi = fi.mean()
        std_fi = fi.std()
        print("Mean feature importances:")
        for i, col in enumerate(["lst_anomaly", 'elev', 'day', 'doy_sin', 'doy_cos']):
            print(f"  {col}: {mean_fi[col]:.4f} ± {std_fi[col]:.4f}")

        results['folds'][fold] = {
            "rmse": fold_rmse,
            "r2": fold_r2,
            "feature_importances": fi,
            'model': model,
            'train_idx': train_idx,
            'test_idx': test_idx,
        }

    results['predictions'] = predictions

    # pickle the models and data
    dump_path = STORAGE_DIR + 'mgwr_results/layer2/grf_models.pkl'
    with open(dump_path, 'wb') as f:
        pkl.dump(results, f)

def run_layer2_grf():
    gdf_basin, gdf_outside, basin_station_ids = get_grf_data()
    train_grf_model(gdf_basin, gdf_outside, basin_station_ids,
    n_splits=20)
        
if __name__ == "__main__":
    run_layer2_grf()