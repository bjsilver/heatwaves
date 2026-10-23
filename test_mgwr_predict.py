import pickle
from types import SimpleNamespace

import numpy as np
import pytest

import mgwr_predict


@pytest.fixture(scope="module")
def fitted_model():
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
        ]
    )
    predictors = np.column_stack([coords[:, 0] + 1.0, coords[:, 1] + 1.0])
    response = (3.0 + 2.0 * predictors[:, 0] - predictors[:, 1]).reshape(-1, 1)
    selector = SimpleNamespace(
        bw=(np.array([9, 9, 9]), np.array([[9, 9, 9]])),
        bw_init=9,
        params=np.zeros((len(coords), 3)),
    )
    model = mgwr_predict.MGWR(coords, response, predictors, selector)
    selector.params = model.exact_fit().params
    return model


def test_predict_matches_calibrated_locations(fitted_model):
    fitted_results = fitted_model.fit()

    prediction = fitted_model.predict(
        fitted_model.coords,
        fitted_model.X[:, 1:],
    )

    np.testing.assert_allclose(prediction.params, fitted_results.params, rtol=1e-5)
    np.testing.assert_allclose(prediction.predy, fitted_results.predy, rtol=1e-5)


def test_predict_returns_consistent_shapes_without_mutating_model(fitted_model):
    points = np.array([[0.5, 0.5], [1.5, 0.5]])
    predictors = np.array([[1.5, 1.5], [2.5, 1.5]])
    original_points = fitted_model.points
    original_predictors = fitted_model.P

    prediction = fitted_model.predict(points, predictors)

    assert prediction.params.shape == (2, fitted_model.k)
    assert prediction.predy.shape == (2, 1)
    np.testing.assert_allclose(
        prediction.predy,
        np.sum(
            np.column_stack([np.ones(2), predictors]) * prediction.params,
            axis=1,
            keepdims=True,
        ),
    )
    assert fitted_model.points is original_points
    assert fitted_model.P is original_predictors


@pytest.mark.parametrize(
    ("points", "predictors", "message"),
    [
        (np.array([0.0, 0.0]), np.array([[0.0, 0.0]]), "points must be"),
        (np.array([[0.0, 0.0]]), np.array([[0.0]]), "P must have"),
        (np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([[0.0, 0.0]]), "same number"),
    ],
)
def test_predict_validates_input_shapes(fitted_model, points, predictors, message):
    with pytest.raises(ValueError, match=message):
        fitted_model.predict(points, predictors)


def test_unpickled_standard_model_gains_predict_method(fitted_model):
    restored_model = pickle.loads(pickle.dumps(fitted_model))

    prediction = restored_model.predict(
        restored_model.coords[:1],
        restored_model.X[:1, 1:],
    )

    assert prediction.predy.shape == (1, 1)