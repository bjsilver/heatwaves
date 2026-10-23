"""
Predict tmax with the final GRF on the full LST grid, a single zarr
(time, lat, lon), chunked (365, 256, 256), NaN outside the Amazon basin.

This script runs with fixed configuration values defined at the top of the
module instead of argparse flags.
"""
#%%
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree
from constants import LST_DIR, STORAGE_DIR, GRF_RESULTS_DIR, ELEV_PATH, TMAX_RECHUNKED_FILE
from grf_data import get_climatologically_valid_data
from grf_evaluation import retrieve_final_model_results
from grf_fit import FEATURES
from obs_data import define_local_azimuthal_projection
from ufuncs import get_amazon_basin_mask, _check_coords_match, ensure_match_coords

OUT_DIR = os.path.join(GRF_RESULTS_DIR, 'predictions')
BLOCK = 256  # LST spatial chunk size
DAYS = 365  # noleap year = output time chunk
SHARD = 256  # pixels per accumulation lock
STATIC_FEATURES = {'elevation'}
TIME_FEATURES = {'doy_sin', 'doy_cos', 'day'}

# Fixed runtime configuration for the production script.
YEARS = list(range(2003, 2021))  # inclusive
SDIR = 'all_elev'
MODEL_TYPE = 'rf'
WORKERS = 70
CONCURRENT_BLOCKS = 4
PIXELS_PER_BATCH = 4096
MIN_WEIGHT = 0.0
BLOCK_START = 0
BLOCK_STOP = None
INIT_ONLY = False
OVERWRITE = False

def open_lst():
    return xr.open_zarr(LST_DIR + 'regridded_rechunked.zarr')['LST'] / 10


def load_static(year):
    """Lat/lon, elevation and boolean validity mask on the LST grid."""
    grid = xr.open_zarr(TMAX_RECHUNKED_FILE)['Ta_TMAX'][0]
    elev = xr.open_dataset(ELEV_PATH)['elevation']
    elev = elev.assign_coords(lat=grid['lat'], lon=grid['lon'])
    mask = get_amazon_basin_mask()['raster']
    
    grid, elev = ensure_match_coords(grid, elev)
    grid, mask = ensure_match_coords(grid, mask)
    _check_coords_match([grid, elev, mask])


    valid = grid.notnull().where(mask==0)
    valid = xr.where(valid==1, True, False)
    return (grid['lat'].values, grid['lon'].values, elev, valid)


def time_features(years):
    """
    Concatenated per-day features for all years (noleap, 365 days/year),
    consistent with add_aux_time_features. 'day' is relative to the first
    training date, as during fitting.
    """
    clim = get_climatologically_valid_data()
    dt = clim['datetime'] if 'datetime' in clim else clim.index.get_level_values('datetime')
    day0 = pd.DatetimeIndex(dt).min()

    parts = []
    for year in years:
        t = xr.date_range(start=f'{year}-01-01', end=f'{year}-12-31',
                          freq='D', calendar='noleap')
        parts.append(pd.DatetimeIndex([x.strftime('%Y-%m-%d') for x in t]))
    tindex = parts[0].append(parts[1:]) if len(parts) > 1 else parts[0]
    doy = tindex.dayofyear
    return {
        'doy_sin': np.sin(2 * np.pi * doy / 365),
        'doy_cos': np.cos(2 * np.pi * doy / 365),
        'day': (tindex - day0).days.astype(float),
    }


def noleap_time_index(years):
    """CFTimeIndex of every day of every year on the noleap calendar."""
    parts = [xr.date_range(start=f'{y}-01-01', end=f'{y}-12-31', freq='D',
                           calendar='noleap', use_cftime=True) for y in years]
    return parts[0].append(parts[1:]) if len(parts) > 1 else parts[0]


def init_store(path, years, lat, lon):
    """Create one empty zarr store (metadata only), chunked (365, 256, 256)."""
    time = noleap_time_index(years)
    template = xr.Dataset(
        {'tmax': (('time', 'lat', 'lon'),
                  da.full((len(time), len(lat), len(lon)), np.nan, dtype='float32',
                          chunks=(DAYS, BLOCK, BLOCK)))},
        coords={'time': time, 'lat': lat, 'lon': lon},
    )
    template.to_zarr(path, mode='w', compute=False)


# ----------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------

def knn_weights(tree, coords, k, min_weight=0.0):
    """
    Adaptive bisquare weights matching PyGRFBuilder, normalised to sum 1.
    min_weight > 0 drops stations below that normalised weight (approximation).
    """
    d, idx = tree.query(coords, k=k, workers=-1)
    d, idx = d.reshape(-1, k), idx.reshape(-1, k)
    h = d[:, -1:] * 1.0000001
    ratio = np.divide(d, h, out=np.zeros_like(d), where=h > 0)
    w = np.maximum(1.0 - ratio ** 2, 0.0) ** 2
    zero = h[:, 0] == 0
    w[zero] = 0.0
    w[zero, 0] = 1.0
    w /= w.sum(axis=1, keepdims=True)
    if min_weight > 0:
        w[w < min_weight] = 0.0
        w /= w.sum(axis=1, keepdims=True)
    return idx.astype(np.int32), w


def forest_predict(rf, X):
    """
    Mean of the trees' predictions. Equivalent to RandomForestRegressor.predict
    for single-output forests but without its per-call joblib machinery, which
    adds overhead and emits warnings when called from many threads.
    X must be C-contiguous float32.
    """
    estimators = getattr(rf, 'estimators_', None)
    if estimators is None or getattr(rf, 'n_outputs_', 1) != 1:
        return rf.predict(X)
    total = np.zeros(len(X), dtype=np.float64)
    for tree in estimators:
        total += tree.predict(X, check_input=False)
    return total / len(estimators)


def predict_batch(model, tree, az_xy, lst_pt, elev_p, tfeat, executor,
                  min_weight=0.0, max_rows=1_000_000):
    """
    az_xy  : (P, 2) projected pixel coords
    lst_pt : (P, T) float32 LST, pixel-major
    elev_p : (P,)   elevation
    tfeat  : dict of (T,) time features
    executor : shared ThreadPoolExecutor that runs the forest predictions
    Returns (P, T) float32 tmax.
    """
    P, T = lst_pt.shape
    k = min(int(model.band_width), len(model.local_models))
    idx, w = knn_weights(tree, az_xy, k, min_weight)
    if not model.predict_weighted:
        w = (w > 0) / np.maximum((w > 0).sum(axis=1, keepdims=True), 1)
    keep = w > 0

    # Invert pixel->stations into station->pixels (pixels stay ascending
    # within each station thanks to the stable sort).
    pix = np.broadcast_to(np.arange(P, dtype=np.int32)[:, None], idx.shape)[keep]
    stn, wts = idx[keep], w[keep]
    order = np.argsort(stn, kind='stable')
    pix, stn, wts = pix[order], stn[order], wts[order]
    stations, starts = np.unique(stn, return_index=True)
    bounds = np.append(starts, len(stn))

    out = np.zeros((P, T), dtype=np.float64)
    # Sharded locks: tasks only contend if they touch the same pixel range.
    n_shards = -(-P // SHARD)
    locks = [threading.Lock() for _ in range(n_shards)]
    shard_edges = np.arange(n_shards + 1) * SHARD
    step = max(1, max_rows // T)  # bounds the feature matrix per predict call

    def build_X(p):
        cols = []
        for f in FEATURES:
            if f == 'LST':
                cols.append(lst_pt[p].ravel())
            elif f in STATIC_FEATURES:
                cols.append(np.repeat(elev_p[p], T))
            elif f in TIME_FEATURES:
                cols.append(np.tile(tfeat[f], len(p)))
            else:
                raise KeyError(f'Unsupported feature {f}')
        return np.ascontiguousarray(np.stack(cols, axis=1), dtype=np.float32)

    def task(j, a, b):
        p, wj = pix[a:b], wts[a:b]
        pred = forest_predict(model.local_models[stations[j]], build_X(p)).reshape(len(p), T)
        pred *= wj[:, None]
        cuts = np.searchsorted(p, shard_edges)
        for i in range(n_shards):
            lo, hi = cuts[(i + j) % n_shards], cuts[(i + j) % n_shards + 1]
            if hi > lo:
                with locks[(i + j) % n_shards]:
                    out[p[lo:hi]] += pred[lo:hi]

    futures = [
        executor.submit(task, j, a, min(a + step, bounds[j + 1]))
        for j in range(len(stations))
        for a in range(bounds[j], bounds[j + 1], step)
    ]
    for f in futures:
        f.result()  # propagates exceptions

    return out.astype(np.float32)


def available_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


def main():

    print('Years:', YEARS, '-', YEARS[-1], f'({len(YEARS)})')
    lat, lon, elevation, valid = load_static(YEARS[0])
    nlat, nlon = valid.shape
    nbx = -(-nlon // BLOCK)
    nblocks = -(-nlat // BLOCK) * nbx

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f'grf_tmax_{YEARS[0]}_{YEARS[-1]}.zarr')
    if OVERWRITE or not os.path.exists(out_path):
        init_store(out_path, YEARS, lat, lon)
        print('Initialised', out_path)
    if INIT_ONLY:
        return

    done_dir = os.path.join(OUT_DIR, f'done_{YEARS[0]}_{YEARS[-1]}')
    os.makedirs(done_dir, exist_ok=True)

    model = retrieve_final_model_results(sdir=SDIR, model_type=MODEL_TYPE)['model']
    if model.kernel != 'adaptive':
        raise ValueError("Requires kernel='adaptive'.")
    for m in model.local_models:
        m.n_jobs = 1
    tree = cKDTree(np.asarray(model.station_coords, dtype=np.float64))

    az_proj = define_local_azimuthal_projection(recalculate=False)
    tfeat = time_features(YEARS)
    stop = nblocks if BLOCK_STOP is None else min(BLOCK_STOP, nblocks)

    def block_slices(b):
        r0, c0 = (b // nbx) * BLOCK, (b % nbx) * BLOCK
        return (slice(r0, min(r0 + BLOCK, nlat)), slice(c0, min(c0 + BLOCK, nlon)))

    todo = [b for b in range(BLOCK_START, stop)
            if valid[block_slices(b)].any()
            and not os.path.exists(os.path.join(done_dir, f'{b}'))]
    print(f'{len(todo)} blocks to process; {WORKERS} predict threads, '
          f'{CONCURRENT_BLOCKS} concurrent blocks', flush=True)

    def process_block(b, executor):
        rs, cs = block_slices(b)
        vb = np.asarray(valid[rs, cs]).astype(bool)
        # pyproj transformers are not guaranteed thread-safe: one per call
        to_az = Transformer.from_crs(CRS.from_epsg(4326), az_proj, always_xy=True)
        lon2d, lat2d = np.meshgrid(lon[cs], lat[rs])
        x, y = to_az.transform(lon2d[vb], lat2d[vb])
        az_xy = np.column_stack([x, y])
        elev_p = elevation.values[rs, cs][vb]

       # Select the current spatial block lazily.
        lst_block = open_lst().isel(lat=rs, lon=cs)

        # Find the local row/column indices of valid pixels.
        row_idx, col_idx = np.nonzero(vb)
        P = len(row_idx)
        T = lst_block.sizes["time"]

        pred = np.empty((P, T), dtype=np.float32)

        # Process batches of pixels, loading only their LST values.
        for s in range(0, P, PIXELS_PER_BATCH):
            e = min(s + PIXELS_PER_BATCH, P)

            # Paired indexing selects the required pixels, not a rectangular
            # section of latitude rows.
            rows = xr.DataArray(row_idx[s:e], dims="pixel")
            cols = xr.DataArray(col_idx[s:e], dims="pixel")

            lst_batch = lst_block.isel(lat=rows, lon=cols).compute().values
            # Shape: (time, batch_pixels)

            pred[s:e] = predict_batch(
                model, tree, az_xy[s:e],
                np.ascontiguousarray(lst_batch.T, dtype=np.float32),
                elev_p[s:e], tfeat, executor, MIN_WEIGHT
            )

        del lst_block

        for i in range(len(YEARS)):
            t0 = i * DAYS
            full = np.full((DAYS,) + vb.shape, np.nan, dtype=np.float32)
            full[:, vb] = pred[:, t0:t0 + DAYS].T
            # downcast dtype to float32
            full = full.astype(np.float32)
            xr.Dataset({'tmax': (('time', 'lat', 'lon'), full)}).to_zarr(
                out_path, region={'time': slice(t0, t0 + DAYS), 'lat': rs, 'lon': cs})
        open(os.path.join(done_dir, f'{b}'), 'w').close()
        return b, P

    # Separate pools: block threads only orchestrate (read, submit, wait,
    # write); the predict pool does the CPU work, so they cannot deadlock.
    with ThreadPoolExecutor(WORKERS) as predict_pool, \
            ThreadPoolExecutor(CONCURRENT_BLOCKS) as block_pool:
        futures = [block_pool.submit(process_block, b, predict_pool) for b in todo]
        for n, f in enumerate(as_completed(futures), 1):
            b, P = f.result()
            print(f'[{n}/{len(todo)}] block {b} done ({P} pixels)', flush=True)


#%%
if __name__ == '__main__':
    main()
