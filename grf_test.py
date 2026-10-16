#%%
import json
import os
import time

import geopandas as gpd
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from sklearn.utils.parallel import Parallel

from constants import GRF_RESULTS_DIR
from grf_data import get_climatologically_valid_data
from grf_fit import AMAZON_BASIN, CRS, N_JOBS_MODELS
from pygrf_modified import PyGRFBuilder
import sys

FEATURES = ["LST", "elevation", "doy_sin", "doy_cos", "day"]

# Minimal fixed settings
N_FOLDS = 10
TIME_STRIDE = 1
SEED = 0
INIT_POINTS = 25
N_ITER = 100
MODEL_TYPE = sys.argv[1] if len(sys.argv) > 1 else "rf"  # "rf" or "xgb"
OUT_CSV = os.path.join(GRF_RESULTS_DIR, f"grf_test/results_{MODEL_TYPE}.csv")


RF_PBOUNDS = {
    "band_width": (10, 80),
    "n_estimators": (20, 60),
    "min_samples_leaf": (2, 40),
    "max_depth": (5, 30),
    "max_features": (0.3, 1.0),
    "min_samples_split": (2, 40),
}

XGB_PBOUNDS = {
    "band_width": (10, 80),
    "n_estimators": (100, 600),
    "max_depth": (3, 12),
    "learning_rate": (0.01, 0.3),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
}


def prepare_data(df, time_stride):
    rfd = df.copy()
    rfd = rfd.dropna(subset=["tmax", "LST", "elevation", "easting", "northing"])
    datetimes = rfd.index
    rfd = gpd.GeoDataFrame(
        rfd,
        geometry=gpd.points_from_xy(rfd["easting"], rfd["northing"]),
        crs=CRS,
    )
    rfd = rfd.reset_index().set_index(["station_id", "datetime"])
    rfd["doy_sin"] = np.sin(2 * np.pi * datetimes.dayofyear / 365)
    rfd["doy_cos"] = np.cos(2 * np.pi * datetimes.dayofyear / 365)
    rfd["day"] = (datetimes - datetimes.min()).days

    if time_stride > 1:
        dates = rfd.index.get_level_values("datetime")
        keep = ((dates - dates.min()).days % time_stride) == 0
        rfd = rfd[keep]
    return rfd


def split_in_basin(rfd, n_folds, seed):
    basin_geom = AMAZON_BASIN.geometry.iloc[0]
    station_geom = rfd.groupby("station_id")["geometry"].first()
    in_ids = station_geom.index[station_geom.geometry.within(basin_geom)]

    rng = np.random.default_rng(seed)
    n_folds = min(n_folds, len(in_ids))
    test_ids = rng.choice(np.asarray(in_ids), size=n_folds, replace=False)
    return in_ids, test_ids


def evaluate_metrics(rfd, in_ids, test_ids, model_params, parallel):
    station_level = rfd.index.get_level_values("station_id")
    in_df = rfd[station_level.isin(in_ids)]
    in_level = in_df.index.get_level_values("station_id")

    obs_all, pred_all, fold_rmse = [], [], []
    n_folds = len(test_ids)
    for fold_i, sid in enumerate(test_ids, start=1):
        fold_start = time.perf_counter()
        print(f"  fold {fold_i}/{n_folds} station={sid} starting", flush=True)
        is_test = in_level == sid
        test = in_df[is_test]
        train = in_df[~is_test]

        model = PyGRFBuilder(
            band_width=model_params["band_width"],
            base_model=MODEL_TYPE,
            model_hyperparameters=model_params["model_hyperparameters"],
            kernel="adaptive",
            train_weighted=True,
            predict_weighted=True,
            random_state=42,
            n_jobs=4,
            n_jobs_models=N_JOBS_MODELS,
        )
        model.fit(
            train[FEATURES].values,
            train["tmax"].astype(float).values.reshape(-1, 1),
            train[["easting", "northing"]].values,
            location_ids=train.index.get_level_values("station_id").values,
            parallel=parallel,
        )
        pred = np.asarray(
            model.predict(
                test[FEATURES].values,
                test[["easting", "northing"]].values,
                parallel=parallel,
            )
        ).flatten()
        obs = test["tmax"].astype(float).values

        obs_all.append(obs)
        pred_all.append(pred)
        this_fold_rmse = np.sqrt(np.mean((pred - obs) ** 2))
        fold_rmse.append(this_fold_rmse)
        fold_secs = time.perf_counter() - fold_start
        running_mean = float(np.mean(fold_rmse))
        print(
            f"  fold {fold_i}/{n_folds} done rmse={this_fold_rmse:.4f} "
            f"running_mean={running_mean:.4f} time={fold_secs:.1f}s",
            flush=True,
        )

    obs = np.concatenate(obs_all)
    pred = np.concatenate(pred_all)
    err = pred - obs
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "r2": float(1 - np.sum(err ** 2) / np.sum((obs - obs.mean()) ** 2)),
        "mean_fold_rmse": float(np.mean(fold_rmse)),
        "worst_fold_rmse": float(np.max(fold_rmse)),
        "n_test_obs": int(len(obs)),
    }


def normalize_params(raw):
    if MODEL_TYPE == "rf":
        return {
            "band_width": int(round(raw["band_width"])),
            "model_hyperparameters": {
                "n_estimators": int(round(raw["n_estimators"])),
                "min_samples_leaf": int(round(raw["min_samples_leaf"])),
                "max_depth": int(round(raw["max_depth"])),
                "max_features": float(raw["max_features"]),
                "bootstrap": True,
            },
        }
    if MODEL_TYPE == "xgb":
        return {
            "band_width": int(round(raw["band_width"])),
            "model_hyperparameters": {
                "n_estimators": int(round(raw["n_estimators"])),
                "max_depth": int(round(raw["max_depth"])),
                "learning_rate": float(raw["learning_rate"]),
                "subsample": float(raw["subsample"]),
                "colsample_bytree": float(raw["colsample_bytree"]),
                "tree_method": "hist",
            },
        }
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")


def config_key(params):
    hp = params["model_hyperparameters"]
    stable = {
        "model_type": MODEL_TYPE,
        "band_width": int(params["band_width"]),
    }
    for k, v in hp.items():
        if isinstance(v, float):
            stable[k] = round(v, 6)
        else:
            stable[k] = v
    return json.dumps(stable, sort_keys=True)


def flat_result_row(model_params, metrics):
    row = {
        "model_type": MODEL_TYPE,
        "band_width": model_params["band_width"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "bias": metrics["bias"],
        "r2": metrics["r2"],
        "mean_fold_rmse": metrics["mean_fold_rmse"],
        "worst_fold_rmse": metrics["worst_fold_rmse"],
        "n_test_obs": metrics["n_test_obs"],
        "config_key": config_key(model_params),
    }
    row.update(model_params["model_hyperparameters"])
    return row


def optimizer_params_from_row(row):
    if MODEL_TYPE == "rf":
        return {
            "band_width": float(row["band_width"]),
            "n_estimators": float(row["n_estimators"]),
            "min_samples_leaf": float(row["min_samples_leaf"]),
            "max_depth": float(row["max_depth"]),
            "max_features": float(row["max_features"]),
            "min_samples_split": float(row["min_samples_split"]),
        }
    if MODEL_TYPE == "xgb":
        return {
            "band_width": float(row["band_width"]),
            "n_estimators": float(row["n_estimators"]),
            "max_depth": float(row["max_depth"]),
            "learning_rate": float(row["learning_rate"]),
            "subsample": float(row["subsample"]),
            "colsample_bytree": float(row["colsample_bytree"]),
        }
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    df = get_climatologically_valid_data()
    rfd = prepare_data(df, TIME_STRIDE)
    in_ids, test_ids = split_in_basin(rfd, N_FOLDS, SEED)

    print(
        f"In-basin stations: {len(in_ids)}, held-out stations: {len(test_ids)}, obs: {len(rfd):,}",
        flush=True,
    )

    results = pd.DataFrame()
    if os.path.exists(OUT_CSV):
        results = pd.read_csv(OUT_CSV)

    if len(results) and "model_type" in results.columns:
        results = results[results["model_type"] == MODEL_TYPE].copy()

    done = set(results["config_key"]) if len(results) else set()
    rmse_by_key = {row["config_key"]: float(row["rmse"]) for _, row in results.iterrows()} if len(results) else {}

    with Parallel(n_jobs=N_JOBS_MODELS, backend="threading") as parallel:

        eval_counter = len(done)
        total_eval_budget = INIT_POINTS + N_ITER

        def objective(**suggested):
            nonlocal results, eval_counter
            model_params = normalize_params(suggested)
            key = config_key(model_params)

            if key in done:
                print("Skipping already-computed config.", flush=True)
                return -rmse_by_key[key]

            eval_counter += 1
            eval_start = time.perf_counter()
            print(
                f"\n=== Evaluation {eval_counter} (budget {total_eval_budget}) ===",
                flush=True,
            )
            print(f"params={model_params}", flush=True)

            metrics = evaluate_metrics(rfd, in_ids, test_ids, model_params, parallel)
            row = flat_result_row(model_params, metrics)
            results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
            results.to_csv(OUT_CSV, index=False)

            done.add(key)
            rmse_by_key[key] = metrics["rmse"]
            eval_secs = time.perf_counter() - eval_start
            print(
                "evaluation complete "
                f"rmse={metrics['rmse']:.4f} "
                f"mae={metrics['mae']:.4f} "
                f"bias={metrics['bias']:.4f} "
                f"r2={metrics['r2']:.4f} "
                f"mean_fold_rmse={metrics['mean_fold_rmse']:.4f} "
                f"worst_fold_rmse={metrics['worst_fold_rmse']:.4f} "
                f"n_test_obs={metrics['n_test_obs']} "
                f"time={eval_secs:.1f}s",
                flush=True,
            )
            return -metrics["rmse"]

        optimizer = BayesianOptimization(
            f=objective,
            pbounds=RF_PBOUNDS if MODEL_TYPE == "rf" else XGB_PBOUNDS,
            random_state=SEED,
            verbose=0,
            allow_duplicate_points=True,
        )

        if len(results):
            for _, row in results.iterrows():
                optimizer.register(
                    params=optimizer_params_from_row(row),
                    target=-float(row["rmse"]),
                )

        print(
            f"Loaded {len(done)} completed runs from CSV. "
            f"New evaluation budget: {total_eval_budget}",
            flush=True,
        )
        optimizer.maximize(init_points=INIT_POINTS, n_iter=N_ITER)

    best = optimizer.max
    best_params = normalize_params(best["params"])
    print("\nBest parameters:")
    print(best_params)
    print(f"Best RMSE: {-best['target']:.4f}")
    print(f"Results CSV: {OUT_CSV}")

#%%
if __name__ == "__main__":
    main()

#%% plot results csv

results = pd.read_csv(OUT_CSV)