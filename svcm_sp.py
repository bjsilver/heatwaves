#%%
from svcm_sp_data import load_svcm_data, get_valid_meteostat_stations
from scipy.spatial import Delaunay
from constants import (
    STORAGE_DIR, LST_DIR, MIN_VALID_YEARS, MIN_VALID_DAYS_PER_YEAR,
    DOY_POOL_WINDOW, LAMBDA_GRID,
)
from pyproj import CRS, Transformer
from math import factorial
import numpy as np
import cvxpy as cp
from collections import defaultdict
from scipy.sparse import coo_matrix, hstack, block_diag
from scipy.linalg import null_space


SPLINE_DEGREE = 3
BERNSTEIN_EXPONENTS = tuple(
    (i, j, SPLINE_DEGREE - i - j)
    for i in range(SPLINE_DEGREE, -1, -1)
    for j in range(SPLINE_DEGREE - i, -1, -1)
)
EXPONENT_INDEX = {exponent: index for index, exponent in enumerate(BERNSTEIN_EXPONENTS)}



DATA_PATH = STORAGE_DIR + 'svcm_data/'

if __name__ == '__main__':
    df = load_svcm_data()
    mdf = get_valid_meteostat_stations(min_valid_years=MIN_VALID_YEARS, min_valid_days_per_year=MIN_VALID_DAYS_PER_YEAR)


def get_delaunay_triangulation(mdf):
    with open(DATA_PATH+'azimuthal_projection.txt', 'r') as f:
        az_proj = CRS.from_proj4(f.read())
    mdf = mdf.to_crs(az_proj)
    mdf['easting'] = mdf['geometry'].x
    mdf['northing'] = mdf['geometry'].y

    points = mdf[['easting', 'northing']].values
    tri = Delaunay(points)

    return tri, points


def barycentric_coordinates(tri, points):
    simplex = tri.find_simplex(points)
    if np.any(simplex < 0):
        raise ValueError("Some points lie outside the triangulation")

    X = points - tri.transform[simplex, 2]
    bary12 = np.einsum(
            'nij,nj->ni',
            tri.transform[simplex, :2],
            X
        )
    bary3 = 1 - bary12.sum(axis=1)
    bary = np.column_stack([
        bary12,
        bary3
    ])
    return simplex, bary

def bernstein_basis_degree2(bary):
    
    l1 = bary[:, 0]
    l2 = bary[:, 1]
    l3 = bary[:, 2]
    
    return np.column_stack([
        l1**2,
        l2**2,
        l3**2,
        2*l1*l2,
        2*l1*l3,
        2*l2*l3
    ])


def bernstein_basis_degree3(bary):
    basis = []
    for i, j, k in BERNSTEIN_EXPONENTS:
        coefficient = factorial(SPLINE_DEGREE) / (
            factorial(i) * factorial(j) * factorial(k)
        )
        basis.append(coefficient * bary[:, 0] ** i * bary[:, 1] ** j * bary[:, 2] ** k)
    return np.column_stack(basis)


def build_full_basis_matrix(simplex, B_local, n_triangles):
    """
    Expand per-point local Bernstein values (n_points, 6) into the
    sparse global basis (n_points, 6 * n_triangles) used in the
    design matrix, placing each point's six values at the columns
    belonging to its containing triangle.
    """
    n_points, n_basis = B_local.shape
    rows = np.repeat(np.arange(n_points), n_basis)
    cols = (n_basis * simplex[:, None] + np.arange(n_basis)[None, :]).ravel()
    data = B_local.ravel()

    return coo_matrix(
        (data, (rows, cols)),
        shape=(n_points, n_basis * n_triangles)
    ).tocsr()


def build_continuity_matrix(tri):
    """
    Build H for the paper's degree-3, first-order-continuous
    bivariate spline space.
    """

    simplices = tri.simplices
    n_triangles = len(simplices)
    n_basis = len(BERNSTEIN_EXPONENTS)

    # Map global edge -> [(triangle_index, local_vertex_a,
    #                      local_vertex_b), ...]
    edge_map = defaultdict(list)

    for t, vertices in enumerate(simplices):

        for a, b in [(0, 1), (0, 2), (1, 2)]:

            edge = tuple(sorted([
                vertices[a],
                vertices[b]
            ]))

            edge_map[edge].append((t, a, b))

    rows = []
    cols = []
    data = []

    row = 0

    for edge, triangles in edge_map.items():

        # Only shared/internal edges create continuity constraints
        if len(triangles) != 2:
            continue

        t1, a1, b1 = triangles[0]
        t2, a2, b2 = triangles[1]

        vertices1 = simplices[t1]
        vertices2 = simplices[t2]

        global_v0, global_v1 = edge
        local1_v0 = int(np.flatnonzero(vertices1 == global_v0)[0])
        local1_v1 = int(np.flatnonzero(vertices1 == global_v1)[0])
        local2_v0 = int(np.flatnonzero(vertices2 == global_v0)[0])
        local2_v1 = int(np.flatnonzero(vertices2 == global_v1)[0])
        local1_opp = ({0, 1, 2} - {local1_v0, local1_v1}).pop()
        local2_opp = ({0, 1, 2} - {local2_v0, local2_v1}).pop()

        for i in range(SPLINE_DEGREE + 1):
            exponent1 = [0, 0, 0]
            exponent2 = [0, 0, 0]
            exponent1[local1_v0] = i
            exponent1[local1_v1] = SPLINE_DEGREE - i
            exponent2[local2_v0] = i
            exponent2[local2_v1] = SPLINE_DEGREE - i
            exponent1 = tuple(exponent1)
            exponent2 = tuple(exponent2)
            col1 = n_basis * t1 + EXPONENT_INDEX[exponent1]
            col2 = n_basis * t2 + EXPONENT_INDEX[exponent2]
            rows.extend([row, row])
            cols.extend([col1, col2])
            data.extend([1.0, -1.0])
            row += 1

        edge_vector = tri.points[vertices1[b1]] - tri.points[vertices1[a1]]
        normal = np.array([-edge_vector[1], edge_vector[0]])
        normal /= np.linalg.norm(normal)

        gradients1 = tri.transform[t1, :2, :2]
        gradients2 = tri.transform[t2, :2, :2]
        barycentric_gradients1 = np.vstack([gradients1, -gradients1.sum(axis=0)])
        barycentric_gradients2 = np.vstack([gradients2, -gradients2.sum(axis=0)])

        edge_degree = SPLINE_DEGREE - 1
        for i in range(edge_degree + 1):
            beta1 = [0, 0, 0]
            beta2 = [0, 0, 0]
            beta1[local1_v0] = i
            beta1[local1_v1] = edge_degree - i
            beta2[local2_v0] = i
            beta2[local2_v1] = edge_degree - i

            for beta, gradients, triangle_index in [
                (beta1, barycentric_gradients1, t1),
                (beta2, barycentric_gradients2, t2),
            ]:
                sign = 1.0 if triangle_index == t1 else -1.0
                for local_index in range(3):
                    exponent = list(beta)
                    exponent[local_index] += 1
                    exponent = tuple(exponent)
                    coefficient_index = EXPONENT_INDEX[exponent]
                    value = SPLINE_DEGREE * normal @ gradients[local_index]
                    rows.append(row)
                    cols.append(n_basis * triangle_index + coefficient_index)
                    data.append(sign * value)
            row += 1

    H = coo_matrix(
        (data, (rows, cols)),
        shape=(row, n_basis * n_triangles)
    ).tocsr()

    return H


def build_energy_penalty_matrix(tri):
    """
    Reconstructed roughness penalty for cubic Bernstein-Bezier
    polynomials over the fixed triangulation, following the
    thin-plate-type energy functional used in bivariate penalized
    spline smoothing over triangulations (Lai & Schumaker; the
    "BPST" framework co-developed by L. Wang, a co-author of the
    SVCM-SP methodology). This is a best-effort reconstruction, not
    a transcription of Kim et al. (2021), whose exact penalty could
    not be verified (paywalled).

    For cubic pieces the Hessian is linear, so the three-point
    degree-2 exact quadrature rule integrates the energy exactly.
    """
    n_triangles = len(tri.simplices)
    n_basis = len(BERNSTEIN_EXPONENTS)
    W = np.diag([1.0, 2.0, 1.0])
    quadrature = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])
    quadrature_weights = np.full(3, 1 / 3)

    def bernstein_values(degree, bary):
        values = []
        for i in range(degree, -1, -1):
            for j in range(degree - i, -1, -1):
                k = degree - i - j
                values.append(
                    factorial(degree) / (factorial(i) * factorial(j) * factorial(k))
                    * bary[0] ** i * bary[1] ** j * bary[2] ** k
                )
        return np.asarray(values)

    def shifted(alpha, index, amount):
        if alpha is None:
            return None
        beta = list(alpha)
        beta[index] -= amount
        if min(beta) < 0:
            return None
        return tuple(beta)

    blocks = []

    for t in range(n_triangles):

        J = tri.transform[t, :2, :2]
        j11, j12 = J[0, 0], J[0, 1]
        j21, j22 = J[1, 0], J[1, 1]

        # maps (Hg_uu, Hg_uv, Hg_vv) to Cartesian (F_xx, F_xy, F_yy)
        # via Hess_f = J^T Hess_g J for the affine (u, v) <- (x, y) map
        T_J = np.array([
            [j11**2,     2 * j11 * j21,       j21**2],
            [j11 * j12,  j11 * j22 + j12 * j21, j21 * j22],
            [j12**2,     2 * j12 * j22,       j22**2],
        ])

        area = 0.5 / abs(np.linalg.det(J))

        M = np.zeros((n_basis, n_basis))
        for (u, v), weight in zip(quadrature, quadrature_weights):
            bary = np.array([u, v, 1 - u - v])
            hessian = np.zeros((3, n_basis))
            degree2_exponents = tuple(
                (i, j, SPLINE_DEGREE - 2 - i - j)
                for i in range(SPLINE_DEGREE - 2, -1, -1)
                for j in range(SPLINE_DEGREE - 2 - i, -1, -1)
            )
            degree2_index = {exponent: index for index, exponent in enumerate(degree2_exponents)}
            basis2 = bernstein_values(SPLINE_DEGREE - 2, bary)

            for column, alpha in enumerate(BERNSTEIN_EXPONENTS):
                terms_uu = [
                    (1, shifted(alpha, 0, 2)),
                    (-2, shifted(shifted(alpha, 0, 1), 2, 1)),
                    (1, shifted(alpha, 2, 2)),
                ]
                terms_uv = [
                    (1, shifted(shifted(alpha, 0, 1), 1, 1)),
                    (-1, shifted(shifted(alpha, 0, 1), 2, 1)),
                    (-1, shifted(shifted(alpha, 1, 1), 2, 1)),
                    (1, shifted(alpha, 2, 2)),
                ]
                terms_vv = [
                    (1, shifted(alpha, 1, 2)),
                    (-2, shifted(shifted(alpha, 1, 1), 2, 1)),
                    (1, shifted(alpha, 2, 2)),
                ]

                for hessian_row, terms in enumerate((terms_uu, terms_uv, terms_vv)):
                    hessian[hessian_row, column] = SPLINE_DEGREE * (SPLINE_DEGREE - 1) * sum(
                        coefficient * basis2[degree2_index[exponent]]
                        for coefficient, exponent in terms
                        if exponent is not None
                    )

            C = T_J @ hessian
            M += area * weight * (C.T @ W @ C)

        blocks.append(coo_matrix(M))

    K = block_diag(blocks).tocsr()
    return K


def prepare_solver_cache(H, K):
    """Precompute spatial matrices shared by every fit on one mesh."""
    Q2 = null_space(H.toarray())
    K_reduced = Q2.T @ K.toarray() @ Q2
    K_reduced = (K_reduced + K_reduced.T) / 2
    return {'Q2': Q2, 'K_reduced': K_reduced}


def _solve_svcm_qp(X, y, H, K, lambda0, lambdaL, lambdaE,
                   solver_cache=None, solver=cp.OSQP):
    """
    Reconstructed SVCM-SP constrained penalized least-squares fit.

    minimize ||y - X gamma||^2
        + lambda0 * gamma0' K gamma0
        + lambdaL * gammaL' K gammaL
        + lambdaE * gammaE' K gammaE
    s.t. H @ gamma0 = H @ gammaL = H @ gammaE = 0  (C0 continuity)
         gammaL >= 0, gammaE <= 0                  (sign preservation)

    gamma = [gamma0, gammaL, gammaE], each of length H.shape[1].
    """
    if solver_cache is None:
        solver_cache = prepare_solver_cache(H, K)

    Q2 = solver_cache['Q2']
    K_reduced = solver_cache['K_reduced']
    n_reduced = Q2.shape[1]

    theta0 = cp.Variable(n_reduced)
    thetaL = cp.Variable(n_reduced)
    thetaE = cp.Variable(n_reduced)
    gamma0 = Q2 @ theta0
    gammaL = Q2 @ thetaL
    gammaE = Q2 @ thetaE
    gamma = cp.hstack([gamma0, gammaL, gammaE])

    residual = X @ gamma - y
    objective = cp.sum_squares(residual)
    objective += lambda0 * cp.quad_form(theta0, K_reduced, assume_PSD=True)
    objective += lambdaL * cp.quad_form(thetaL, K_reduced, assume_PSD=True)
    objective += lambdaE * cp.quad_form(thetaE, K_reduced, assume_PSD=True)

    constraints = [
        Q2 @ thetaL >= 0,
        Q2 @ thetaE <= 0,
    ]

    problem = cp.Problem(cp.Minimize(objective), constraints)

    try:
        problem.solve(
            solver=solver,
            eps_abs=1e-8,
            eps_rel=1e-8,
            alpha=1.6,
            adaptive_rho=True,
            scaled_termination=True,
            max_iter=200000,
            polishing=True,
        )
    except cp.error.SolverError:
        problem.solve(solver=cp.SCS)

    if theta0.value is None:
        raise RuntimeError(f"SVCM-SP QP failed to solve, status={problem.status}")

    return {
        'gamma0': Q2 @ theta0.value,
        'gammaL': Q2 @ thetaL.value,
        'gammaE': Q2 @ thetaE.value,
        'Q2': Q2,
        'status': problem.status,
    }


def fit_svcm_sp(X, y, H, K, lambda0, lambdaL=None, lambdaE=None,
                solver_cache=None):
    """
    Fit the sign-preserving spatially varying coefficient model to a
    single (daily or pooled) dataset. See _solve_svcm_qp for the
    objective/constraints.
    """
    if lambdaL is None:
        lambdaL = lambda0
    if lambdaE is None:
        lambdaE = lambda0

    return _solve_svcm_qp(
        X, y, H, K, lambda0, lambdaL, lambdaE,
        solver_cache=solver_cache,
    )


def get_pooled_doy_set(target_doy, window, n_days_per_year=365):
    """
    Day-of-year values (1-indexed, circular/wraps at year boundary)
    within +/- window of target_doy.
    """
    offsets = np.arange(-window, window + 1)
    days = ((target_doy - 1 + offsets) % n_days_per_year) + 1
    return np.unique(days)


def build_pooled_design_matrix(df, station_row_map, B_full, target_doy, window=DOY_POOL_WINDOW):
    """
    Pool station-day observations from all years within +/- window
    days of target_doy, reusing the fixed per-station Bernstein basis
    (station geometry does not change day to day) to assemble the
    sparse pooled design matrix X = [B, B*LST, B*elev] and response y.
    """
    days = get_pooled_doy_set(target_doy, window)

    sub = df[df['doy'].isin(days) & df['station_id'].isin(station_row_map)].copy()

    rows = sub['station_id'].map(station_row_map).to_numpy()
    B_rows = B_full[rows]

    lst = sub['LST'].to_numpy()
    elev = sub['elevation'].to_numpy()
    y = sub['t2m'].to_numpy()

    X = hstack([
        B_rows,
        B_rows.multiply(lst[:, None]),
        B_rows.multiply(elev[:, None]),
    ]).tocsr()

    station_ids = sub['station_id'].to_numpy()

    return X, y, station_ids


def _reduced_gram_matrix(X, H, K, solver_cache=None):
    n_coef = H.shape[1]
    if solver_cache is None:
        solver_cache = prepare_solver_cache(H, K)
    Q2 = solver_cache['Q2']
    n_reduced = Q2.shape[1]
    X_blocks = [X[:, i * n_coef:(i + 1) * n_coef] for i in range(3)]
    gram = np.zeros((3 * n_reduced, 3 * n_reduced))

    for i, X_i in enumerate(X_blocks):
        for j, X_j in enumerate(X_blocks):
            block = Q2.T @ (X_i.T @ X_j).toarray() @ Q2
            gram[i * n_reduced:(i + 1) * n_reduced,
                 j * n_reduced:(j + 1) * n_reduced] = block

    return Q2, gram, solver_cache['K_reduced']


def _gcv_score(X, y, H, K, lambdas, solver_cache=None, gram=None):
    if solver_cache is None:
        solver_cache = prepare_solver_cache(H, K)
    if gram is None:
        _, gram, _ = _reduced_gram_matrix(X, H, K, solver_cache)
    Q2 = solver_cache['Q2']
    K_reduced = solver_cache['K_reduced']
    penalty = block_diag([
        lambdas[0] * K_reduced,
        lambdas[1] * K_reduced,
        lambdas[2] * K_reduced,
    ]).toarray()
    W = gram + penalty
    W = (W + W.T) / 2
    effective_df = np.trace(np.linalg.solve(W, gram))

    fit = fit_svcm_sp(X, y, H, K, *lambdas, solver_cache=solver_cache)
    gamma = np.concatenate([fit['gamma0'], fit['gammaL'], fit['gammaE']])
    residual = np.asarray(X @ gamma - y).ravel()
    denominator = max(1e-12, 1 - effective_df / len(y)) ** 2
    return np.mean(residual ** 2) / denominator


def select_smoothing_parameters(X, y, H, K, lambda_grid=LAMBDA_GRID,
                                max_passes=2, solver_cache=None):
    """
    Select separate roughness parameters with the paper's GCV
    criterion. Coordinate search avoids an unnecessarily large
    three-dimensional grid for production-sized pooled datasets.
    """
    if solver_cache is None:
        solver_cache = prepare_solver_cache(H, K)

    _, gram, _ = _reduced_gram_matrix(X, H, K, solver_cache)

    lambdas = np.full(3, float(np.median(lambda_grid)))
    best_score = _gcv_score(X, y, H, K, lambdas, solver_cache, gram)

    for _ in range(max_passes):
        improved = False
        for coefficient in range(3):
            candidate = lambdas.copy()
            scores = []
            for value in lambda_grid:
                candidate[coefficient] = value
                scores.append(_gcv_score(X, y, H, K, candidate, solver_cache, gram))
            index = int(np.argmin(scores))
            if scores[index] < best_score:
                lambdas[coefficient] = lambda_grid[index]
                best_score = scores[index]
                improved = True
        if not improved:
            break

    return tuple(lambdas), best_score


def build_date_design_matrix(df, station_row_map, B_full, target_date):
    """Build a sparse design matrix from one calendar date only."""
    target_date = np.datetime64(target_date, 'D')
    target_year = int(str(target_date)[:4])
    target_doy = int((target_date - np.datetime64(f'{target_year}-01-01')) / np.timedelta64(1, 'D')) + 1
    sub = df[
        (df['year'] == target_year) &
        (df['doy'] == target_doy) &
        df['station_id'].isin(station_row_map)
    ].copy()

    if sub.empty:
        raise ValueError(f'No complete station observations found for {target_date}')

    rows = sub['station_id'].map(station_row_map).to_numpy()
    B_rows = B_full[rows]
    lst = sub['LST'].to_numpy()
    elev = sub['elevation'].to_numpy()
    y = sub['t2m'].to_numpy()
    X = hstack([
        B_rows,
        B_rows.multiply(lst[:, None]),
        B_rows.multiply(elev[:, None]),
    ]).tocsr()
    return X, y, sub['station_id'].to_numpy()


def fit_date(df, station_row_map, B_full, H, K, target_date,
             lambda_grid=LAMBDA_GRID, solver_cache=None):
    """
    Fit SVCM-SP to one actual calendar date, preserving that year's
    temperature signal rather than averaging across years.
    """
    X, y, station_ids = build_date_design_matrix(df, station_row_map, B_full, target_date)

    if solver_cache is None:
        solver_cache = prepare_solver_cache(H, K)

    lambdas, gcv = select_smoothing_parameters(
        X, y, H, K, lambda_grid, solver_cache=solver_cache
    )

    fit = fit_svcm_sp(X, y, H, K, *lambdas, solver_cache=solver_cache)

    fit['lambdas'] = lambdas
    fit['gcv'] = gcv
    fit['date'] = str(target_date)
    fit['n_obs'] = X.shape[0]
    fit['n_stations'] = len(np.unique(station_ids))

    return fit


def barycentric_coordinates_allow_outside(tri, points):
    """
    Like barycentric_coordinates but marks out-of-hull points as
    invalid instead of raising, for target-grid evaluation where
    some edge cells may fall outside the fixed triangulation.
    """
    simplex = tri.find_simplex(points)
    valid = simplex >= 0
    simplex_safe = np.where(valid, simplex, 0)

    X = points - tri.transform[simplex_safe, 2]
    bary12 = np.einsum('nij,nj->ni', tri.transform[simplex_safe, :2], X)
    bary3 = 1 - bary12.sum(axis=1)
    bary = np.column_stack([bary12, bary3])

    return simplex_safe, bary, valid


def build_grid_basis(tri, easting, northing):
    """
    Precompute the fixed-triangulation basis at arbitrary target
    points (e.g. a prediction grid). Reusable across every day since
    both the triangulation and the target grid are fixed.
    """
    n_triangles = len(tri.simplices)
    points = np.column_stack([easting, northing])

    simplex, bary, valid = barycentric_coordinates_allow_outside(tri, points)
    B_local = bernstein_basis_degree3(bary)
    B_grid = build_full_basis_matrix(simplex, B_local, n_triangles)

    return B_grid, valid


def predict_ta_grid(fit, B_grid, valid, lst, elev):
    """
    Evaluate the fitted coefficient surfaces at grid points and
    combine with that day's LST/elevation. Points outside the fixed
    triangulation's convex hull are set to NaN.
    """
    beta0 = B_grid @ fit['gamma0']
    betaLST = B_grid @ fit['gammaL']
    betaelev = B_grid @ fit['gammaE']

    ta = beta0 + betaLST * lst + betaelev * elev
    ta = np.where(valid, ta, np.nan)

    return ta


def project_lonlat_to_azimuthal(lon, lat):
    with open(DATA_PATH + 'azimuthal_projection.txt', 'r') as f:
        az_proj = CRS.from_proj4(f.read())
    transformer = Transformer.from_crs('EPSG:4326', az_proj, always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    return np.asarray(easting), np.asarray(northing)


def predict_date_map(fit, tri, lon, lat, lst, elev):
    """
    lon, lat: 1-D coordinate arrays of the target grid (degrees).
    lst, elev: 2-D (lat, lon) arrays for the target day, on the same grid.
    Returns a 2-D (lat, lon) array of predicted Ta, NaN outside the triangulation hull.
    """
    lon2d, lat2d = np.meshgrid(lon, lat)
    easting, northing = project_lonlat_to_azimuthal(lon2d.ravel(), lat2d.ravel())

    B_grid, valid = build_grid_basis(tri, easting, northing)
    ta = predict_ta_grid(fit, B_grid, valid, lst.ravel(), elev.ravel())

    return ta.reshape(lat2d.shape)


if __name__ == '__main__':
    tri, points = get_delaunay_triangulation(mdf)
    simplex, bary = barycentric_coordinates(tri, points)
    B = bernstein_basis_degree3(bary)

    #%%

    edge_to_triangles = defaultdict(list)

    for t, vertices in enumerate(tri.simplices):

        v0, v1, v2 = vertices

        edges = [
            tuple(sorted((v0, v1))),
            tuple(sorted((v0, v2))),
            tuple(sorted((v1, v2))),
        ]

        for edge in edges:
            edge_to_triangles[edge].append(t)

    shared_edges = {
        edge: triangles
        for edge, triangles in edge_to_triangles.items()
        if len(triangles) == 2
    }
    #%%

    H = build_continuity_matrix(tri)
    K = build_energy_penalty_matrix(tri)
    solver_cache = prepare_solver_cache(H, K)

    n_triangles = len(tri.simplices)
    B_full = build_full_basis_matrix(simplex, B, n_triangles)
    station_row_map = {sid: i for i, sid in enumerate(mdf['id'].to_numpy())}

    import matplotlib.pyplot as plt
    import xarray as xr
    from ufuncs import get_amazon_basin_mask

    TARGET_DATE = '2003-06-09'
    TARGET_YEAR = int(TARGET_DATE[:4])
    TARGET_DOY = int(
        (np.datetime64(TARGET_DATE) - np.datetime64(f'{TARGET_YEAR}-01-01')) /
        np.timedelta64(1, 'D')
    ) + 1

    fit = fit_date(
        df, station_row_map, B_full, H, K, TARGET_DATE,
        solver_cache=solver_cache,
    )

    basin = get_amazon_basin_mask(buffer_km=0)['vector']
    west, south, east, north = basin.total_bounds

    lst_da = xr.open_zarr(LST_DIR + f'zz_lst_{TARGET_YEAR}.zarr')['LST'].sel(
        lat=slice(south, north), lon=slice(west, east)
    ) / 10
    elev_da = xr.open_dataset(
        STORAGE_DIR + '../elevation/orig_resolution_regridded/elevation_regridded.nc'
    )['elevation'].interp(lat=lst_da['lat'], lon=lst_da['lon'])

    lst_day = lst_da.sel(doy=TARGET_DOY).values
    elev_grid = elev_da.values

    ta_map = predict_date_map(fit, tri, lst_da['lon'].values, lst_da['lat'].values, lst_day, elev_grid)

    ta_da = xr.DataArray(
        ta_map,
        coords={'lat': lst_da['lat'], 'lon': lst_da['lon']},
        dims=('lat', 'lon'),
        name='t2m',
    )

    ta_da.plot(cmap='inferno', vmin=10, vmax=35)
    plt.title(f'Predicted t2m, day-of-year {TARGET_DOY}, {TARGET_YEAR}')
    plt.show()
    plt.savefig(f't2m_{TARGET_DOY}_{TARGET_YEAR}.png')

