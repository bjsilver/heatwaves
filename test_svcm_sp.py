import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix

import svcm_sp


def _make_triangulation(n_points=40, seed=0):
    rng = np.random.default_rng(seed)
    points = rng.uniform(0, 10, size=(n_points, 2))
    tri = Delaunay(points)
    simplex, bary = svcm_sp.barycentric_coordinates(tri, points)
    B_local = svcm_sp.bernstein_basis_degree3(bary)
    n_triangles = len(tri.simplices)
    B_full = svcm_sp.build_full_basis_matrix(simplex, B_local, n_triangles)
    H = svcm_sp.build_continuity_matrix(tri)
    K = svcm_sp.build_energy_penalty_matrix(tri)
    return tri, points, B_full, H, K


def _fit_surface(y, B_full, H, K, lam=1e-6):
    # only the intercept surface is exercised here, so the LST/elev
    # blocks of X are zero and their coefficients are irrelevant
    zeros = csr_matrix(B_full.shape)
    X = svcm_sp.hstack([B_full, zeros, zeros]).tocsr()
    fit = svcm_sp.fit_svcm_sp(X, y, H, K, lambda0=lam)
    return fit['gamma0']


def test_continuity_matrix_is_satisfied_by_constant_surface():
    tri, points, B_full, H, K = _make_triangulation()
    n_coef = H.shape[1]
    gamma = np.full(n_coef, 3.5)
    np.testing.assert_allclose(H @ gamma, 0, atol=1e-10)


def test_energy_penalty_matrix_is_symmetric_psd():
    tri, points, B_full, H, K = _make_triangulation(n_points=15, seed=1)
    K_dense = K.toarray()
    np.testing.assert_allclose(K_dense, K_dense.T, atol=1e-8)
    eigvals = np.linalg.eigvalsh(K_dense)
    assert eigvals.min() > -1e-6


def test_recovers_constant_surface():
    tri, points, B_full, H, K = _make_triangulation()
    y = np.full(points.shape[0], 12.3)

    gamma0 = _fit_surface(y, B_full, H, K)
    pred = B_full @ gamma0

    np.testing.assert_allclose(pred, y, atol=1e-4)


def test_recovers_global_linear_surface():
    tri, points, B_full, H, K = _make_triangulation()
    x, yc = points[:, 0], points[:, 1]
    y = 2.0 + 3.0 * x - 1.5 * yc

    gamma0 = _fit_surface(y, B_full, H, K)
    pred = B_full @ gamma0

    np.testing.assert_allclose(pred, y, atol=1e-3)


def test_recovers_global_quadratic_surface():
    tri, points, B_full, H, K = _make_triangulation()
    x, yc = points[:, 0], points[:, 1]
    y = x**2 + 2 * x * yc + 3 * yc**2 + 4 * x - 5 * yc + 7

    gamma0 = _fit_surface(y, B_full, H, K, lam=1e-8)
    pred = B_full @ gamma0

    np.testing.assert_allclose(pred, y, atol=1e-2)


def test_sign_preservation_holds_under_adversarial_data():
    tri, points, B_full, H, K = _make_triangulation()
    n_stations = points.shape[0]
    rng = np.random.default_rng(2)

    lst = rng.uniform(20, 40, size=n_stations)
    elev = rng.uniform(0, 2000, size=n_stations)

    # Adversarial: t2m anti-correlated with LST and correlated with
    # elevation, i.e. pushing the unconstrained OLS solution towards
    # the physically wrong sign.
    y = 30 - 0.05 * lst + 0.01 * elev

    X = svcm_sp.hstack([
        B_full,
        B_full.multiply(lst[:, None]),
        B_full.multiply(elev[:, None]),
    ]).tocsr()

    fit = svcm_sp.fit_svcm_sp(X, y, H, K, lambda0=1e-4)

    assert fit['gammaL'].min() >= -1e-6
    assert fit['gammaE'].max() <= 1e-6


def test_recovers_synthetic_ta_from_known_coefficient_surfaces():
    tri, points, B_full, H, K = _make_triangulation(n_points=60, seed=3)
    n_stations = points.shape[0]
    rng = np.random.default_rng(4)

    x, yc = points[:, 0], points[:, 1]
    beta0_true = 20 + 0.1 * x
    betaLST_true = 0.5 + 0.01 * yc
    betaelev_true = -0.005 - 0.0001 * x

    lst = rng.uniform(20, 40, size=n_stations)
    elev = rng.uniform(0, 2000, size=n_stations)
    noise = rng.normal(scale=0.05, size=n_stations)

    y = beta0_true + betaLST_true * lst + betaelev_true * elev + noise

    X = svcm_sp.hstack([
        B_full,
        B_full.multiply(lst[:, None]),
        B_full.multiply(elev[:, None]),
    ]).tocsr()

    fit = svcm_sp.fit_svcm_sp(X, y, H, K, lambda0=1e-5)
    gamma = np.concatenate([fit['gamma0'], fit['gammaL'], fit['gammaE']])
    pred = X @ gamma

    np.testing.assert_allclose(pred, y, atol=0.3)
    assert fit['gammaL'].min() >= -1e-6
    assert fit['gammaE'].max() <= 1e-6


def test_get_pooled_doy_set_wraps_around_year_boundary():
    days = svcm_sp.get_pooled_doy_set(target_doy=1, window=3, n_days_per_year=365)
    np.testing.assert_array_equal(days, np.array([1, 2, 3, 4, 363, 364, 365]))
