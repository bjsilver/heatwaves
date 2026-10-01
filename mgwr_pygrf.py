#%%
from pygrf_modified import PyGRFBuilder
from mgwr_layer2 import load_data_with_anomalies, filter_stations
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import geopandas as gpd
from mgwr_data import  get_amazon_basin_mask
from mgwr_evaluation import interpolate_zz_to_stations, merge_zz_into_df
from obs_data import define_local_azimuthal_projection
from sklearn.ensemble import RandomForestRegressor

az_crs = define_local_azimuthal_projection(recalculate=False)
AMAZON_MASK = get_amazon_basin_mask()['vector'].to_crs(az_crs)


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




#%% global model without weighting

# date = '2020-06-01'
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
# plt.scatter(gdf_basin['tmax_anomaly'], gdf_basin['lst_anomaly'], label='Basin', alpha=0.7)

#%% test single random forest at single station
sdf = gdf_basin[gdf_basin.station_id == basin_station_ids[100]]
import matplotlib.pyplot as plt
plt.scatter(sdf['lst_anomaly'], sdf['tmax_anomaly'], label='Basin', alpha=0.7)
print(f'correlation coefficient: {sdf["lst_anomaly"].corr(sdf["tmax_anomaly"])}')
# calculate day of timeseries


#%%

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

#%%


kf = KFold(n_splits=10, shuffle=True, random_state=42)

predictions = pd.DataFrame(
    index=pd.MultiIndex.from_arrays(
        [gdf_basin['station_id'], gdf_basin.index],
        names=['station_id', 'datetime'],
    ),
    columns=['obs_resid', 'pred_resid', 'pred_zz', 'tmax_clim', 'tmax'],
    dtype=float,
)
for fold, (train_idx, test_idx) in enumerate(kf.split(basin_station_ids)):

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


    # comparison with ZZ data
    break

    
#%%

p = predictions.dropna()
p['tmax_pred'] = p['pred_resid'] + p['tmax_clim']

# plot tmax pred vs tmax, pred_zz vs tmax and tmax clim vs tmax
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(p['tmax'], p['tmax_clim'], alpha=0.1, s=10, color='green', label='Climatology')
ax.scatter(p['tmax'], p['pred_zz'], alpha=0.1, s=10, color='blue', label='ZZ Predicted')
ax.scatter(p['tmax'], p['tmax_pred'], alpha=0.1, s=10, color='orange', label='GRF Predicted')

# calc error metrics for both models
rmse_grf = np.sqrt(mean_squared_error(p['tmax'], p['tmax_pred']))
r2_grf = r2_score(p['tmax'], p['tmax_pred'])
rmse_zz = np.sqrt(mean_squared_error(p['tmax'], p['pred_zz']))
r2_zz = r2_score(p['tmax'], p['pred_zz'])
rmse_clim = np.sqrt(mean_squared_error(p['tmax'], p['tmax_clim']))
r2_clim = r2_score(p['tmax'], p['tmax_clim'])

print(f"GRF - RMSE: {rmse_grf:.2f}, R²: {r2_grf:.2f}")
print(f"ZZ - RMSE: {rmse_zz:.2f}, R²: {r2_zz:.2f}")
print(f"Climatology - RMSE: {rmse_clim:.2f}, R²: {r2_clim:.2f}")

#%%