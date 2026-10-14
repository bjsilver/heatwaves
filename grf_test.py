"""
Parameter / basin-buffer sweep for the station-level GRF.

Evaluates every combination of GRF hyperparameters x outside-basin buffer
distances using leave-one-out over a fixed random subset of in-basin stations
(same folds for every configuration, so configurations are directly comparable).
Training data can be thinned in time with --time-stride to keep it fast.

Results are appended to a CSV after every configuration, and configurations
already in the CSV are skipped, so the sweep can be interrupted and resumed.

Examples
--------
python grf_test.py                                  # default grid
python grf_test.py --n-folds 15 --time-stride 5     # cheaper
python grf_test.py --max-configs 30                 # random subset of the grid
"""
import argparse
import itertools
import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.utils.parallel import Parallel

from constants import GRF_RESULTS_DIR
from grf_fit import AMAZON_BASIN, CRS, N_JOBS_MODELS
from mgwr_data import get_climatologically_valid_data
from pygrf_modified import PyGRFBuilder

FEATURES = ['LST', 'elevation', 'doy_sin', 'doy_cos', 'day']

# ----------------------------------------------------------------------
# Search space. Edit freely: every key except `buffer_km` is passed to
# PyGRFBuilder. buffer_km=None keeps ALL out-of-basin stations; a number keeps
# only those within that many km of the basin perimeter.
# ----------------------------------------------------------------------
PARAM_GRID = {
    'band_width': [10, 25, 50],
    'n_estimators': [30, 60],
    'min_samples_leaf': [5, 20],
    'max_depth': [10, 20],
    'max_features': [1.0, 0.6],
    'train_weighted': [True],
    'buffer_km': [0, 50, 100,  350],
}


def prepare_data(df, time_stride):
    rfd = df.copy()
    rfd = rfd.dropna(subset=['tmax', 'LST', 'elevation', 'easting', 'northing'])
    datetimes = rfd.index
    rfd = gpd.GeoDataFrame(
        rfd,
        geometry=gpd.points_from_xy(rfd['easting'], rfd['northing']),
        crs=CRS,
    )
    rfd = rfd.reset_index().set_index(['station_id', 'datetime'])
    rfd['doy_sin'] = np.sin(2 * np.pi * datetimes.dayofyear / 365)
    rfd['doy_cos'] = np.cos(2 * np.pi * datetimes.dayofyear / 365)
    # computed before thinning so 'day' keeps its meaning
    rfd['day'] = (datetimes - datetimes.min()).days

    if time_stride > 1:
        dates = rfd.index.get_level_values('datetime')
        keep = ((dates - dates.min()).days % time_stride) == 0
        rfd = rfd[keep]
    return rfd


def split_stations(rfd, n_folds, seed):
    basin_geom = AMAZON_BASIN.geometry.iloc[0]
    coord_df = rfd.groupby('station_id')['geometry'].first()
    inside = coord_df.geometry.within(basin_geom)

    in_ids = coord_df.index[inside]
    out_ids = coord_df.index[~inside]
    # distance (km) from each outside station to the basin; CRS is in metres
    out_dist_km = coord_df[~inside].apply(basin_geom.distance) / 1000.0

    rng = np.random.default_rng(seed)
    n_folds = min(n_folds, len(in_ids))
    test_ids = rng.choice(np.asarray(in_ids), size=n_folds, replace=False)
    return in_ids, out_ids, out_dist_km, test_ids


def evaluate(rfd, in_ids, out_dist_km, test_ids, model_params, buffer_km, parallel):
    station_level = rfd.index.get_level_values('station_id')

    if buffer_km is None:
        out_keep = out_dist_km.index
    else:
        out_keep = out_dist_km.index[out_dist_km <= buffer_km]
    out_df = rfd[station_level.isin(out_keep)]
    in_df = rfd[station_level.isin(in_ids)]
    in_level = in_df.index.get_level_values('station_id')

    obs_all, pred_all, fold_rmse = [], [], []
    for sid in test_ids:
        is_test = in_level == sid
        test = in_df[is_test]
        train = pd.concat([in_df[~is_test], out_df])

        model = PyGRFBuilder(
            kernel='adaptive',
            bootstrap=True,
            random_state=42,
            n_jobs=4,
            n_jobs_models=N_JOBS_MODELS,
            **model_params,
        )
        model.fit(
            train[FEATURES].values,
            train['tmax'].astype(float).values.reshape(-1, 1),
            train[['easting', 'northing']].values,
            location_ids=train.index.get_level_values('station_id').values,
            parallel=parallel,
        )
        pred = np.asarray(
            model.predict(test[FEATURES].values,
                          test[['easting', 'northing']].values,
                          parallel=parallel)
        ).flatten()
        obs = test['tmax'].astype(float).values

        obs_all.append(obs)
        pred_all.append(pred)
        fold_rmse.append(np.sqrt(np.mean((pred - obs) ** 2)))

    obs = np.concatenate(obs_all)
    pred = np.concatenate(pred_all)
    err = pred - obs
    return {
        'n_out_stations': len(out_keep),
        'rmse': float(np.sqrt(np.mean(err ** 2))),
        'mae': float(np.mean(np.abs(err))),
        'bias': float(np.mean(err)),
        'r2': float(1 - np.sum(err ** 2) / np.sum((obs - obs.mean()) ** 2)),
        'mean_fold_rmse': float(np.mean(fold_rmse)),
        'worst_fold_rmse': float(np.max(fold_rmse)),
        'n_test_obs': len(obs),
    }


def config_key(cfg):
    return json.dumps(cfg, sort_keys=True, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-folds', type=int, default=20,
                    help='number of in-basin stations held out (LOO subset)')
    ap.add_argument('--time-stride', type=int, default=3,
                    help='keep every k-th day to speed things up (1 = all)')
    ap.add_argument('--max-configs', type=int, default=None,
                    help='randomly sample this many configs from the grid')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='grf_test/results.csv',
                    help='CSV path relative to GRF_RESULTS_DIR')
    args = ap.parse_args()

    out_path = os.path.join(GRF_RESULTS_DIR, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df = get_climatologically_valid_data()
    rfd = prepare_data(df, args.time_stride)
    in_ids, out_ids, out_dist_km, test_ids = split_stations(rfd, args.n_folds, args.seed)
    print(f"{len(in_ids)} in-basin stations ({len(test_ids)} held out in turn), "
          f"{len(out_ids)} out-of-basin stations, {len(rfd):,} obs.", flush=True)

    keys = list(PARAM_GRID)
    configs = [dict(zip(keys, v)) for v in itertools.product(*PARAM_GRID.values())]
    if args.max_configs and args.max_configs < len(configs):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(configs), size=args.max_configs, replace=False)
        configs = [configs[i] for i in sorted(idx)]

    results = pd.DataFrame()
    if os.path.exists(out_path):
        results = pd.read_csv(out_path)
    done = set(results['config_key']) if len(results) else set()
    print(f"{len(configs)} configs, {len(done)} already done.", flush=True)

    with Parallel(n_jobs=N_JOBS_MODELS, backend='threading') as parallel:
        for i, cfg in enumerate(configs):
            key = config_key(cfg)
            if key in done:
                continue
            model_params = {k: v for k, v in cfg.items() if k != 'buffer_km'}
            print(f"[{i + 1}/{len(configs)}] {cfg}", flush=True)
            metrics = evaluate(rfd, in_ids, out_dist_km, test_ids,
                               model_params, cfg['buffer_km'], parallel)
            print(f"    RMSE={metrics['rmse']:.4f}  R2={metrics['r2']:.4f}  "
                  f"bias={metrics['bias']:.4f}  n_out={metrics['n_out_stations']}",
                  flush=True)
            row = pd.DataFrame([{**cfg, **metrics, 'config_key': key}])
            results = pd.concat([results, row], ignore_index=True)
            # rewrite the full DataFrame each time so the CSV is always complete
            results.to_csv(out_path, index=False)

    res = results.drop(columns='config_key').sort_values('rmse').reset_index(drop=True)
    sorted_path = out_path.replace('.csv', '_sorted.csv')
    res.to_csv(sorted_path, index=False)
    print(f"\nSaved {len(res)} results to {out_path}\nand ranked copy to {sorted_path}")
    print("\nTop 10 configurations by RMSE:")
    print(res.head(10).to_string(index=False))

    # marginal effect of each parameter: mean RMSE per value, averaged over the rest
    print("\nMean RMSE by parameter value:")
    for k in keys:
        if res[k].nunique() > 1:
            print(res.assign(**{k: res[k].astype(str)}).groupby(k)['rmse'].mean().round(4).to_string())
            print()


if __name__ == '__main__':
    main()
