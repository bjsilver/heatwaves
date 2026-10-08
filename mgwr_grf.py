#%%
from pygrf_modified import PyGRFBuilder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import geopandas as gpd
from mgwr_data import  get_amazon_basin_mask, get_climatologically_valid_data
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

gdf = get_climatologically_valid_data().dropna()
gdf['day'] = [d.days for d in (gdf.index - gdf.index.min())]
gdf["doy_sin"] = np.sin(2 * np.pi * gdf["doy"] / 365)
gdf["doy_cos"] = np.cos(2 * np.pi * gdf["doy"] / 365)

#%%
xys = gdf.groupby(gdf.station_id).first()[['easting', 'northing']]
xys = gpd.GeoDataFrame(xys, geometry=gpd.points_from_xy(xys.easting, xys.northing), crs=az_crs)
basin_stations = xys[xys.within(AMAZON_MASK.geometry.iloc[0])].index
outside_stations = xys[~xys.within(AMAZON_MASK.geometry.iloc[0])].index
gdf_basin = gdf[gdf.station_id.isin(basin_stations)]
gdf_outside = gdf[gdf.station_id.isin(outside_stations)]

basin_station_ids = gdf_basin["station_id"].unique()

#%%
n_splits = 10


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

    X_train = train_df[["LST", 'elevation', 'day', 'doy_sin', 'doy_cos']].to_numpy()
    y_train = train_df["tmax"].astype(float).to_numpy()
    xy_train = train_df[["easting", "northing"]].to_numpy()

    X_test = basin_test[["LST", 'elevation', 'day', 'doy_sin', 'doy_cos']].to_numpy()
    xy_test = basin_test[["easting", "northing"]].to_numpy()
    y_test = basin_test["tmax"].astype(float).to_numpy()

    model = PyGRFBuilder(
        n_estimators=60,
        adaptive="adaptive",
        band_width=30,
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
    fi = pd.DataFrame(fis, columns=["LST", 'elevation', 'day', 'doy_sin', 'doy_cos'])

    mean_fi = fi.mean()
    std_fi = fi.std()
    print("Mean feature importances:")
    for i, col in enumerate(["LST", 'elevation', 'day', 'doy_sin', 'doy_cos']):
        print(f"  {col}: {mean_fi[col]:.4f} ± {std_fi[col]:.4f}")

    results['folds'][fold] = {
        "rmse": fold_rmse,
        "r2": fold_r2,
        "feature_importances": fi,
        'model': model,
        'train_idx': train_idx,
        'test_idx': test_idx,
    }

    break

results['predictions'] = predictions