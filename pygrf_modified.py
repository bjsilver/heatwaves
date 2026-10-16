import numpy as np
import pandas as pd
from sklearn.utils.parallel import Parallel, delayed
from scipy.spatial import distance
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from contextlib import contextmanager
import joblib

try:
    from tqdm.auto import tqdm
except Exception:  # tqdm is optional; fall back to no progress bar
    tqdm = None


@contextmanager
def joblib_progress_bar(total, desc):
    if tqdm is None:
        yield
        return

    pbar = tqdm(total=total, desc=desc, unit="model")

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            pbar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        pbar.close()


class PyGRFBuilder:

    def __init__(
        self,
        band_width,
        base_model="rf",
        model_hyperparameters=None,
        kernel="adaptive",
        train_weighted=True,
        predict_weighted=True,
        random_state=None,
        n_jobs=None,
        n_jobs_models=1,
        **kwargs,
    ):
        """
        Station-level geographically weighted random forest.

        Each unique station/location gets its own local RF.

        For station i:
            - Find neighbouring stations using the spatial bandwidth.
            - All temporal observations from those stations are included.
            - Observations are weighted according to the distance between
              their station and station i.
            - One RF is fitted for station i.

        For prediction at an unseen location:
            - Every nearby station-level RF makes a prediction.
            - These predictions are spatially weighted according to the
              distance between the prediction location and the station.
            - The weighted predictions are combined.

        Parameters
        ----------
        band_width : int or float
            If kernel='adaptive', number of neighbouring STATIONS to use.

            If kernel='fixed', spatial bandwidth in coordinate units.

        n_estimators : int
            Number of trees in each RF.

        max_features : float, int, or str
            Passed to RandomForestRegressor.

        kernel : {"adaptive", "fixed"}
            Adaptive = use nearest `band_width` stations.
            Fixed = use all stations within `band_width`.

        train_weighted : bool
            Whether local RF training observations receive spatial weights.

        predict_weighted : bool
            Whether predictions from local RFs are spatially weighted.

        bootstrap : bool
            Passed to RandomForestRegressor.

        random_state : int
            Random seed.

        n_jobs : int
            Number of RF threads.

        min_samples_leaf : int
            Passed to RandomForestRegressor.
        """

        self.band_width = band_width
        self.kernel = kernel
        self.train_weighted = train_weighted
        self.predict_weighted = predict_weighted
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.n_jobs_models = n_jobs_models
        self.base_model = base_model
        self.model_hyperparameters = model_hyperparameters or {}

        self.local_models = []
        self.station_coords = None
        self.station_ids = None
        self.station_to_index = None
        self.distance_matrix = None
        self.train_station_index = None
        self.X_columns = None
        self.local_observation_indices = []

    # ------------------------------------------------------------------
    # Spatial kernel
    # ------------------------------------------------------------------

    def _calculate_weights(self, distances, bandwidth):
        """
        Bi-square kernel used by the original PyGRF.

        w = (1 - (d / h)^2)^2

        Distances >= h receive zero weight.
        """

        distances = np.asarray(distances)

        weights = np.zeros_like(distances, dtype=float)

        inside = distances < bandwidth

        weights[inside] = (
            1.0 - (distances[inside] / bandwidth) ** 2
        ) ** 2

        return weights

    # ------------------------------------------------------------------
    # Determine station neighbourhoods
    # ------------------------------------------------------------------

    def _get_station_neighbourhood(self, station_index):
        """
        Return neighbouring station indices and spatial weights for
        a particular station.
        """

        distances = self.distance_matrix[station_index]

        if self.kernel == "adaptive":

            k = min(int(self.band_width), len(distances))

            # Indices of k nearest stations
            neighbour_indices = np.argpartition(
                distances,
                k - 1
            )[:k]

            neighbour_distances = distances[neighbour_indices]

            # Adaptive bandwidth = distance to kth nearest station
            bandwidth = (
                np.partition(distances, k - 1)[k - 1]
                * 1.0000001
            )

        elif self.kernel == "fixed":

            bandwidth = float(self.band_width)

            neighbour_indices = np.where(
                distances < bandwidth
            )[0]

            neighbour_distances = distances[neighbour_indices]

        else:
            raise ValueError(
                "kernel must be either 'adaptive' or 'fixed'"
            )

        weights = self._calculate_weights(
            neighbour_distances,
            bandwidth
        )

        return neighbour_indices, weights

    # ------------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------------

    def fit(
        self,
        X,
        y,
        coords,
        location_ids=None,
        parallel=None,
    ):
        """
        Fit one local RF per unique spatial location.

        Parameters
        ----------
        X : pandas.DataFrame
            Predictor observations.

            Example:
                lst_anomaly
                elev

        y : pandas.Series or array-like
            Target observations.

            Example:
                tmax_anomaly

        coords : pandas.DataFrame or array-like, shape (n_samples, 2)
            Coordinates for every temporal observation.

            Repeated coordinates are expected because each station has
            many observations.

        location_ids : array-like, optional
            Station/location identifier for every observation.

            Strongly recommended.

            If omitted, unique coordinate pairs are used to identify
            stations.

        Returns
        -------
        self
        """

        # --------------------------------------------------------------
        # Input formatting
        # --------------------------------------------------------------

        if isinstance(X, pd.DataFrame):
            self.X_columns = X.columns.tolist()
            X_array = X.to_numpy(dtype=float)
        else:
            X_array = np.asarray(X, dtype=float)
            self.X_columns = None

        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_array = np.asarray(y).reshape(-1)
        else:
            y_array = np.asarray(y).reshape(-1)

        coords_array = np.asarray(coords, dtype=float)

        if coords_array.ndim != 2 or coords_array.shape[1] != 2:
            raise ValueError(
                "coords must have shape (n_samples, 2)"
            )

        n_samples = len(X_array)

        if len(y_array) != n_samples:
            raise ValueError("X and y have different numbers of observations")

        if len(coords_array) != n_samples:
            raise ValueError("X and coords have different numbers of observations")

        # --------------------------------------------------------------
        # Identify unique stations
        # --------------------------------------------------------------

        if location_ids is None:

            # Use coordinate pairs as station IDs
            station_keys, station_index = np.unique(
                coords_array,
                axis=0,
                return_inverse=True
            )

            self.station_ids = station_keys

        else:

            location_ids = np.asarray(location_ids)

            if len(location_ids) != n_samples:
                raise ValueError(
                    "location_ids must have one value per observation"
                )

            # Preserve order of first occurrence
            unique_ids, station_index = np.unique(
                location_ids,
                return_inverse=True
            )

            self.station_ids = unique_ids

            # Get coordinates for each station.
            #
            # We check that a station has one spatial location.
            station_coords = []

            for i in range(len(unique_ids)):

                rows = coords_array[station_index == i]

                if not np.allclose(rows, rows[0]):
                    raise ValueError(
                        f"location_id {unique_ids[i]} has multiple "
                        "different coordinates"
                    )

                station_coords.append(rows[0])

            station_keys = np.asarray(station_coords)

        self.station_coords = station_keys
        self.train_station_index = station_index

        n_stations = len(self.station_coords)

        print(
            f"Found {n_stations} unique locations "
            f"from {n_samples:,} observations."
        )

        # --------------------------------------------------------------
        # CRITICAL:
        #
        # Distance matrix is now station x station rather than
        # observation x observation.
        # --------------------------------------------------------------

        self.distance_matrix = distance.cdist(
            self.station_coords,
            self.station_coords,
            metric="euclidean"
        )

        # --------------------------------------------------------------
        # Store which observations belong to each station
        # --------------------------------------------------------------

        station_observation_indices = [
            np.where(station_index == i)[0]
            for i in range(n_stations)
        ]

        # --------------------------------------------------------------
        # Fit one local RF at every station
        # --------------------------------------------------------------

        self.local_models = []
        self.local_observation_indices = []

        def fit_local(i):

            # ----------------------------------------------------------
            # Find neighbouring STATIONS
            # ----------------------------------------------------------

            neighbour_stations, station_weights = (
                self._get_station_neighbourhood(i)
            )

            # ----------------------------------------------------------
            # Convert station-level neighbourhood into temporal
            # observations.
            #
            # This is the crucial bit.
            #
            # If station j has 5,000 observations and gets spatial
            # weight 0.7, ALL 5,000 observations receive weight 0.7.
            # ----------------------------------------------------------

            local_observations = []

            local_weights = []

            for station_j, station_weight in zip(
                neighbour_stations,
                station_weights
            ):

                obs_idx = station_observation_indices[station_j]

                local_observations.append(obs_idx)

                local_weights.append(
                    np.full(
                        len(obs_idx),
                        station_weight
                    )
                )

            local_observations = np.concatenate(
                local_observations
            )

            local_weights = np.concatenate(
                local_weights
            )

            X_local = X_array[local_observations]
            y_local = y_array[local_observations]

            # ----------------------------------------------------------
            # Fit RF
            # ----------------------------------------------------------

            if self.base_model == "rf":

                model = RandomForestRegressor(
                    random_state=(
                        None
                        if self.random_state is None
                        else self.random_state + i
                    ),
                    n_jobs=self.n_jobs,
                    **self.model_hyperparameters
                )

            elif self.base_model == "xgb":

                model = XGBRegressor(
                    random_state=(
                        None
                        if self.random_state is None
                        else self.random_state + i
                    ),
                    n_jobs=self.n_jobs,
                    **self.model_hyperparameters
                )

            if self.train_weighted:

                model.fit(
                    X_local,
                    y_local,
                    sample_weight=local_weights
                )

            else:

                model.fit(
                    X_local,
                    y_local
                )

            return model, local_observations


        # Threads share X/y without copying; RF fitting releases the GIL.
        # print(
        #     f"Fitting {n_stations} local RFs "
        #     f"({self.n_jobs_models} models in parallel x {self.n_jobs} RF jobs)..."
        # )

        parallel_runner = (
            parallel
            or Parallel(n_jobs=self.n_jobs_models, backend="threading")
        )

        with joblib_progress_bar(
            total=n_stations,
            desc=f"Fitting local {self.base_model.upper()} models",
        ):
            results = parallel_runner(delayed(fit_local)(i) for i in range(n_stations))

        self.local_models = [r[0] for r in results]
        self.local_observation_indices = [r[1] for r in results]

        return self

    # ------------------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------------------

    def predict(
        self,
        X,
        coords,
        parallel=None,
    ):
        """
        Predict at arbitrary locations.

        Importantly, coords do NOT have to correspond to existing
        stations.

        Parameters
        ----------
        X : pandas.DataFrame or array-like
            Predictor values at prediction locations.

        coords : array-like, shape (n_predictions, 2)
            Coordinates of prediction locations.

        Returns
        -------
        predictions : ndarray
            Spatially weighted local-RF predictions.
        """

        if isinstance(X, pd.DataFrame):
            X_array = X.to_numpy(dtype=float)
        else:
            X_array = np.asarray(X, dtype=float)

        coords_array = np.asarray(coords, dtype=float)

        if coords_array.ndim != 2 or coords_array.shape[1] != 2:
            raise ValueError(
                "coords must have shape (n_predictions, 2)"
            )

        n_predictions = len(X_array)
        n_stations = len(self.local_models)

        prediction_distances = distance.cdist(
            coords_array,
            self.station_coords,
            metric="euclidean"
        )

        # --------------------------------------------------------------
        # Spatial neighbourhood of every prediction point. Only the
        # neighbouring station RFs are evaluated for each point.
        # --------------------------------------------------------------

        rows, stations, weights = [], [], []

        for i in range(n_predictions):

            distances_i = prediction_distances[i]

            if self.kernel == "adaptive":

                k = min(int(self.band_width), n_stations)

                station_indices = np.argpartition(distances_i, k - 1)[:k]

                bandwidth = (
                    np.partition(distances_i, k - 1)[k - 1] * 1.0000001
                )

            else:

                bandwidth = float(self.band_width)

                station_indices = np.where(distances_i < bandwidth)[0]

            w = self._calculate_weights(
                distances_i[station_indices],
                bandwidth
            )

            if not self.predict_weighted:
                w = np.ones_like(w)

            keep = w > 0

            rows.append(np.full(keep.sum(), i))
            stations.append(station_indices[keep])
            weights.append(w[keep])

        rows = np.concatenate(rows)
        stations = np.concatenate(stations)
        weights = np.concatenate(weights)

        # --------------------------------------------------------------
        # Group by station so each local RF predicts only the rows that
        # need it, in parallel across RFs.
        # --------------------------------------------------------------

        order = np.argsort(stations, kind="stable")
        rows, stations, weights = rows[order], stations[order], weights[order]

        unique_stations, starts = np.unique(stations, return_index=True)
        row_groups = np.split(rows, starts[1:])
        weight_groups = np.split(weights, starts[1:])

        local_predictions = (
            parallel
            or Parallel(n_jobs=self.n_jobs_models, backend="threading")
        )(
            delayed(self.local_models[s].predict)(X_array[r])
            for s, r in zip(unique_stations, row_groups)
        )

        numerator = np.zeros(n_predictions)
        denominator = np.zeros(n_predictions)

        for p, r, w in zip(local_predictions, row_groups, weight_groups):
            numerator[r] += p * w
            denominator[r] += w

        predictions = np.full(n_predictions, np.nan)
        valid = denominator > 0
        predictions[valid] = numerator[valid] / denominator[valid]

        return predictions
