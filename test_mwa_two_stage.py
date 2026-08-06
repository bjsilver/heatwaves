import inspect

import numpy as np
import pandas as pd
import xarray as xr

import mwa


def _reference_pair_map(ind_year, driver_year, window_radius=1):
    """Reference annual neighbour selection equivalent to legacy behavior."""
    n_year, n_lat, n_lon = ind_year.shape
    neighbour_index = np.full((n_year, n_lat, n_lon), -1, dtype=np.int32)
    valid = np.zeros((n_year, n_lat, n_lon), dtype=bool)

    for y in range(n_year):
        for i in range(n_lat):
            for j in range(n_lon):
                c_driver = driver_year[y, i, j]
                best_diff = -np.inf
                best_ij = (-1, -1)
                has_lower = False

                for di in range(-window_radius, window_radius + 1):
                    for dj in range(-window_radius, window_radius + 1):
                        if di == 0 and dj == 0:
                            continue
                        ni = i + di
                        nj = j + dj
                        if ni < 0 or nj < 0 or ni >= n_lat or nj >= n_lon:
                            continue

                        n_driver = driver_year[y, ni, nj]
                        diff = c_driver - n_driver
                        if n_driver < c_driver:
                            has_lower = True
                            if not np.isnan(diff) and diff > best_diff:
                                best_diff = diff
                                best_ij = (ni, nj)

                cell_valid = (c_driver >= 0) and has_lower
                valid[y, i, j] = cell_valid
                if cell_valid:
                    neighbour_index[y, i, j] = best_ij[0] * n_lon + best_ij[1]

    return neighbour_index, valid


def _reference_old_full(ind_year, driver_year, dep_time, dep_years, window_radius=1):
    """Reference old combined workflow on daily dependent variable."""
    n_time, n_lat, n_lon = dep_time.shape

    pair_idx, pair_valid = _reference_pair_map(ind_year, driver_year, window_radius=window_radius)

    ind_center = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)
    ind_neighbor = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)
    dep_center = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)
    dep_neighbor = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)

    year0 = int(dep_years.min())
    for t in range(n_time):
        y_pos = int(dep_years[t] - year0)
        for i in range(n_lat):
            for j in range(n_lon):
                if not pair_valid[y_pos, i, j]:
                    continue
                c_dep = dep_time[t, i, j]
                if np.isnan(c_dep):
                    continue
                nidx = pair_idx[y_pos, i, j]
                ni = int(nidx // n_lon)
                nj = int(nidx % n_lon)
                ind_center[t, i, j] = ind_year[y_pos, i, j]
                ind_neighbor[t, i, j] = ind_year[y_pos, ni, nj]
                dep_center[t, i, j] = c_dep
                dep_neighbor[t, i, j] = dep_time[t, ni, nj]

    return ind_center, ind_neighbor, dep_center, dep_neighbor


def _make_synthetic_dataset():
    years = np.array([2003, 2004, 2005], dtype=np.int32)
    lat = np.arange(5, dtype=np.float32)
    lon = np.arange(5, dtype=np.float32)
    time = pd.date_range("2003-01-01", periods=9, freq="120D")

    # Distinct patterns by year to make year-routing verifiable.
    driver = np.stack([
        np.add.outer(np.arange(5, dtype=np.float32), np.arange(5, dtype=np.float32)) + 10.0,
        np.add.outer(np.arange(5, dtype=np.float32), np.arange(5, dtype=np.float32)) + 20.0,
        np.add.outer(np.arange(5, dtype=np.float32), np.arange(5, dtype=np.float32)) + 30.0,
    ])
    # Independent variable differs from driver to ensure decoupled retrieval is tested.
    ind = driver * 2.0 + 1.0

    # Daily dependent variable with a year-specific offset and one NaN center.
    dep = np.zeros((len(time), 5, 5), dtype=np.float32)
    for t, ts in enumerate(time):
        dep[t] = np.add.outer(np.arange(5, dtype=np.float32), np.arange(5, dtype=np.float32)) + (ts.year - 2000) * 100.0 + t
    dep[2, 1, 1] = np.nan

    ind_da = xr.DataArray(ind, dims=("year", "lat", "lon"), coords={"year": years, "lat": lat, "lon": lon}, name="ind")
    driver_da = xr.DataArray(driver, dims=("year", "lat", "lon"), coords={"year": years, "lat": lat, "lon": lon}, name="driver")
    dep_da = xr.DataArray(dep, dims=("time", "lat", "lon"), coords={"time": time, "lat": lat, "lon": lon}, name="dep")

    # Keep chunking small but explicit for dask-lazy path.
    ind_da = ind_da.chunk({"year": 1, "lat": 3, "lon": 3})
    driver_da = driver_da.chunk({"year": 1, "lat": 3, "lon": 3})
    dep_da = dep_da.chunk({"time": 3, "lat": 3, "lon": 3})
    return ind_da, driver_da, dep_da


def test_calculate_mwa_pairs_matches_reference_annual_selection():
    ind_da, driver_da, _ = _make_synthetic_dataset()

    pairs = mwa.calculate_mwa_pairs(ind_da, driver_da, window_radius=1)
    pairs_loaded = pairs.compute()

    expected_idx, expected_valid = _reference_pair_map(ind_da.values, driver_da.values, window_radius=1)

    np.testing.assert_array_equal(pairs_loaded["neighbour_index"].values, expected_idx)
    np.testing.assert_array_equal(pairs_loaded["valid"].values, expected_valid)


def test_apply_mwa_pairs_daily_matches_reference_old_workflow():
    ind_da, driver_da, dep_da = _make_synthetic_dataset()

    pairs = mwa.calculate_mwa_pairs(ind_da, driver_da, window_radius=1)

    new_dep = mwa.apply_mwa_pairs(pairs, dep_da).compute()

    years = dep_da.time.dt.year.values.astype(np.int32)
    _, _, dep_center_ref, dep_neighbor_ref = _reference_old_full(
        ind_da.values,
        driver_da.values,
        dep_da.values,
        years,
        window_radius=1,
    )

    np.testing.assert_allclose(new_dep["dep_center"].values, dep_center_ref, equal_nan=True)
    np.testing.assert_allclose(new_dep["dep_neighbor"].values, dep_neighbor_ref, equal_nan=True)


def test_moving_window_analysis_outputs_match_reference_old_workflow():
    ind_da, driver_da, dep_da = _make_synthetic_dataset()

    ds = xr.Dataset({"ind": ind_da, "driver": driver_da, "dep": dep_da})
    new_all = mwa.moving_window_analysis(
        ds,
        window_radius=1,
        ind_var_name="ind",
        dep_var_name="dep",
        driver_ind_var_name="driver",
        time_scale="daily",
    ).compute()

    years = dep_da.time.dt.year.values.astype(np.int32)
    ind_center_ref, ind_neighbor_ref, dep_center_ref, dep_neighbor_ref = _reference_old_full(
        ind_da.values,
        driver_da.values,
        dep_da.values,
        years,
        window_radius=1,
    )

    np.testing.assert_allclose(new_all["ind_center"].values, ind_center_ref, equal_nan=True)
    np.testing.assert_allclose(new_all["ind_neighbor"].values, ind_neighbor_ref, equal_nan=True)
    np.testing.assert_allclose(new_all["dep_center"].values, dep_center_ref, equal_nan=True)
    np.testing.assert_allclose(new_all["dep_neighbor"].values, dep_neighbor_ref, equal_nan=True)


def test_apply_mwa_pairs_no_pair_expansion_helper_used():
    source = inspect.getsource(mwa.apply_mwa_pairs)
    assert "_expand_pairs_to_dep_dim" not in source
    assert "reshape" in source
