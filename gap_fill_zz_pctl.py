#%%
import xarray as xr
from constants import STORAGE_DIR, HANSEN_DIR
import os
from dask.diagnostics import ProgressBar
import numpy as np
from ufuncs import local_cluster
from scipy.spatial import cKDTree
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
# import lru_cache
from functools import lru_cache
from ufuncs_plot import _add_gridlines

PCT_PATH = os.path.join(STORAGE_DIR, 'percentiles', 'zz_tmax_95pc.zarr')
OUT_PATH = os.path.join(STORAGE_DIR, 'percentiles', 'zz_tmax_95pc_gapfilled.zarr')
HANSEN_PATH = os.path.join(HANSEN_DIR, 'hansen_tc_2000_2024_rechunked.zarr')
LOSS_THRESHOLD = 10
SPATIAL_DIMS = ('lat', 'lon')
IDW_K = 256 # 
IDW_POWER = 1.0
DISTANCE_UPPER_BOUND = 0.25  # degrees, ~28 km at equator
LOOKUP_QUERY_BATCH = 200_000

N_WORKERS = 1
THREADS_PER_WORKER = 32
WORKER_MEMORY_LIMIT = '400GB'
DOY_BATCH_SIZE = 30


def load_data(area_slicer=None):
    pct = xr.open_zarr(PCT_PATH, consolidated=True)['Ta_TMAX_p95']

    reduce_dims = [dim for dim in pct.dims if dim not in SPATIAL_DIMS]
    if reduce_dims:
        mask = pct.notnull().any(reduce_dims).astype('int8')
    else:
        mask = pct.notnull().astype('int8')

    ds = xr.open_zarr(HANSEN_PATH, consolidated=False)
    tc = ds['tree_cover']
    loss = -(tc - tc[0])
    loss = loss.chunk({'time': -1})
    loss = loss.assign_coords({'lat': pct.lat, 'lon': pct.lon})

    if area_slicer is not None:
        return pct.loc[area_slicer], mask.loc[area_slicer], loss.loc[area_slicer]
    else:
        return pct, mask, loss

def _flat_xy_from_indices(flat_idx, lon_1d, lat_1d):
    """Return projected x/y coordinates for flat (lat, lon) indices without 2D meshgrids."""
    nlon = lon_1d.size
    ilat = flat_idx // nlon
    ilon = flat_idx % nlon
    lat_sel = lat_1d[ilat]
    lon_sel = lon_1d[ilon]
    x = lon_sel * np.cos(np.deg2rad(lat_sel))
    y = lat_sel
    return np.column_stack((x, y)).astype(np.float32, copy=False)


def build_idw_lookup(source_mask, target_mask, lon, lat, k=8, power=2.0,
                     distance_upper_bound=np.inf, query_batch_size=LOOKUP_QUERY_BATCH):
    """Precompute IDW neighbour indices/weights for fixed source and target geometry."""
    source_mask = np.asarray(source_mask, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)

    if source_mask.shape != target_mask.shape:
        raise ValueError('source_mask and target_mask must have identical shape')

    if np.any(source_mask & target_mask):
        raise ValueError('source_mask and target_mask must be disjoint')

    n_source = int(source_mask.sum())
    n_target = int(target_mask.sum())

    if n_source == 0:
        raise ValueError('No source pixels available for IDW interpolation')

    if n_target == 0:
        return {
            'source_mask': source_mask,
            'target_mask': target_mask,
            'source_flat_idx': np.flatnonzero(source_mask.ravel()).astype(np.int64, copy=False),
            'target_flat_batches': [],
            'idx_batches': [],
            'weights_batches': [],
        }

    lon_1d = np.asarray(lon, dtype=np.float64)
    lat_1d = np.asarray(lat, dtype=np.float64)

    source_flat_idx = np.flatnonzero(source_mask.ravel()).astype(np.int64, copy=False)
    target_flat_idx = np.flatnonzero(target_mask.ravel()).astype(np.int64, copy=False)

    source_xy = _flat_xy_from_indices(source_flat_idx, lon_1d, lat_1d)

    k_eff = max(1, min(int(k), n_source))
    tree = cKDTree(source_xy)

    idx_batches = []
    weights_batches = []
    target_flat_batches = []
    batch_size = max(1, int(query_batch_size))

    for start in range(0, n_target, batch_size):
        end = min(start + batch_size, n_target)
        tflat = target_flat_idx[start:end]
        target_xy = _flat_xy_from_indices(tflat, lon_1d, lat_1d)

        dist, idx = tree.query(target_xy, k=k_eff, distance_upper_bound=distance_upper_bound)
        if k_eff == 1:
            dist = dist[:, np.newaxis]
            idx = idx[:, np.newaxis]

        # Valid only if finite distance AND index points to real source
        valid = np.isfinite(dist) & (idx < n_source)

        # Replace invalid indices so later source_values[idx] is always safe
        idx_safe = np.where(valid, idx, 0)

        weights = np.zeros_like(dist, dtype=np.float64)
        weights[valid] = 1.0 / np.maximum(dist[valid], 1e-12) ** power
        wsum = weights.sum(axis=1, keepdims=True)
        weights = np.divide(weights, wsum, out=np.zeros_like(weights), where=wsum > 0)

        target_flat_batches.append(tflat.astype(np.int64, copy=False))
        idx_batches.append(idx_safe.astype(np.int32, copy=False))
        weights_batches.append(weights.astype(np.float32, copy=False))

    return {
        'source_mask': source_mask,
        'target_mask': target_mask,
        'source_flat_idx': source_flat_idx,
        'target_flat_batches': target_flat_batches,
        'idx_batches': idx_batches,
        'weights_batches': weights_batches,
    }


def _idw_fill_2d(data_2d, lookup):
    """Fill one 2D field using precomputed IDW neighbours and weights."""
    arr = np.asarray(data_2d, dtype=np.float32)
    source_mask = lookup['source_mask']
    target_mask = lookup['target_mask']

    if source_mask.shape != arr.shape:
        raise ValueError(f'Lookup/source shape {source_mask.shape} != data shape {arr.shape}')

    if not np.any(target_mask):
        return arr

    flat = arr.ravel().copy()
    source_values = flat[lookup['source_flat_idx']]

    for tflat, idx, weights in zip(
        lookup['target_flat_batches'],
        lookup['idx_batches'],
        lookup['weights_batches']
    ):
        neighbour_values = source_values[idx]

        # Robustness: if a source value is NaN for a day, renormalize weights locally.
        finite = np.isfinite(neighbour_values)
        if np.all(finite):
            interp = np.sum(neighbour_values * weights, axis=1)
        else:
            w = np.where(finite, weights, 0.0)
            wsum = w.sum(axis=1)
            safe_wsum = np.where(wsum > 0.0, wsum, 1.0)
            interp = np.sum(np.where(finite, neighbour_values, 0.0) * w, axis=1) / safe_wsum
            interp = np.where(wsum > 0.0, interp, np.nan)

        flat[tflat] = interp.astype(np.float32, copy=False)

    return flat.reshape(arr.shape)


def interpolate_missing(da, lookup):
    filled = xr.apply_ufunc(
        _idw_fill_2d,
        da,
        input_core_dims=[['lat', 'lon']],
        output_core_dims=[['lat', 'lon']],
        dask='parallelized',
        vectorize=True,
        kwargs={'lookup': lookup},
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={"allow_rechunk": True}
    )

    filled = filled.transpose(*da.dims)
    filled.name = da.name
    filled.attrs = da.attrs
    return filled


def interpolate_missing_stream_to_zarr(da, lookup, out_path, doy_batch_size=1):
    """Process DOY in small batches and append to Zarr to keep memory bounded."""
    first_write = True
    name = da.name
    batch_size = max(1, int(doy_batch_size))

    for start in range(0, da.sizes['doy'], batch_size):
        end = min(start + batch_size, da.sizes['doy'])
        batch = da.isel(doy=slice(start, end)).astype(np.float32).compute()
        filled_batch = np.empty_like(batch.values, dtype=np.float32)
        for j in range(batch.sizes['doy']):
            filled_batch[j, :, :] = _idw_fill_2d(batch.isel(doy=j).values, lookup)

        batch_out = xr.DataArray(
            filled_batch,
            coords={
                'doy': batch['doy'].values,
                'lat': da['lat'].values,
                'lon': da['lon'].values,
            },
            dims=('doy', 'lat', 'lon'),
            name=name,
            attrs=da.attrs,
        ).to_dataset(name=name)

        if first_write:
            batch_out.to_zarr(out_path, mode='w', consolidated=False)
            first_write = False
        else:
            batch_out.to_zarr(out_path, mode='a', append_dim='doy', consolidated=False)


def _build_lookup_from_static_masks(pct_ok, mask, k=8, power=2.0,
                                    distance_upper_bound=np.inf,
                                    query_batch_size=LOOKUP_QUERY_BATCH):
    """Build source/target masks once and precompute IDW neighbours/weights."""
    sample = pct_ok.isel(doy=0).compute()
    fillable_mask = mask.astype(bool).compute().values

    source_mask = np.isfinite(sample.values)
    target_mask = np.isnan(sample.values) & fillable_mask

    return build_idw_lookup(
        source_mask=source_mask,
        target_mask=target_mask,
        lon=pct_ok['lon'].values,
        lat=pct_ok['lat'].values,
        k=k,
        power=power,
        distance_upper_bound=distance_upper_bound,
        query_batch_size=query_batch_size,
    )



def main():

    pct, mask, loss = load_data()

    # get the cells where all-time loss is less than Threshold %
    ok = loss.max('time') < LOSS_THRESHOLD
    pct_ok = pct.where(ok)
    pct_ok = pct_ok.chunk({'doy': 1})

    lookup = _build_lookup_from_static_masks(
        pct_ok,
        mask,
        k=IDW_K,
        power=IDW_POWER,
        distance_upper_bound=DISTANCE_UPPER_BOUND
    )

    interpolate_missing_stream_to_zarr(
        pct_ok,
        lookup,
        OUT_PATH,
        doy_batch_size=DOY_BATCH_SIZE,
    )

    print(f'Wrote gap-filled percentile data to {OUT_PATH}')

        

#%%
if __name__ == '__main__':
    with local_cluster(n_workers=N_WORKERS, processes=False, threads_per_worker=THREADS_PER_WORKER,
                       memory_limit=WORKER_MEMORY_LIMIT, dashboard_address=':8787'):
        main()


#%% plot the process


    
# def interp_tests():

#     westlimit=-60.2; southlimit =-10.5; eastlimit=-58.7; northlimit=-9
#     TEST_AREA_SLICER = dict(lat=slice(southlimit, northlimit), lon=slice(westlimit, eastlimit))

#     @lru_cache(maxsize=10)
#     def quick_interp_test(k, power, distance_upper_bound,
#                         loss_threshold):
#         pct, mask, loss = load_data(TEST_AREA_SLICER)
#         pct = pct / 10.0
#         ok = loss.max('time') < loss_threshold
#         pct_ok = pct.where(ok)
#         pct_ok = pct_ok.chunk({'doy': 1, 'lat': -1, 'lon': -1})

#         lookup = _build_lookup_from_static_masks(
#             pct_ok,
#             mask,
#             k=k,
#             power=power,
#             distance_upper_bound=distance_upper_bound
#         )

#         with ProgressBar():
#             filled = interpolate_missing(pct_ok, lookup).mean('doy').compute(processes=4)
#             pct_mean = pct.mean('doy').compute(processes=4)
#         return ok, pct_mean, filled

#     def quick_interp_test_plots(tests, pct_lims=None, diff_lims=None):

#         # run the tests for each parameter combination
#         results = []
#         for test in tests:
#             k = test['k']
#             power = test['power']
#             distance_upper_bound = test['distance_upper_bound']
#             loss_threshold = test['loss_threshold']
#             ok, pct_mean, filled = quick_interp_test(k, power, distance_upper_bound, loss_threshold)
#             results.append((k, power, distance_upper_bound, ok, pct_mean, filled))

#         # plot the results
#         nrows = len(tests)
#         fig, axes = plt.subplots(nrows, 3, figsize=(12, 5 * nrows), subplot_kw={'projection': ccrs.PlateCarree()})
#         axes = np.atleast_2d(axes)

#         cbar_kwargs = {'label': '', 'orientation': 'horizontal', 'shrink': .5, 'pad': 0.1}
#         # plot the result and difference from original pct_mean for each test
#         for i, (k, power, distance_upper_bound, ok, pct_mean, filled) in enumerate(results):
#             ax1 = axes[i, 0]
#             ax2 = axes[i, 1]
#             ax3 = axes[i, 2]

#             # Keep all map panels on identical bounds to avoid subtle misalignment.
#             extent = [
#                 float(pct_mean.lon.min()),
#                 float(pct_mean.lon.max()),
#                 float(pct_mean.lat.min()),
#                 float(pct_mean.lat.max()),
#             ]
#             for ax in (ax1, ax2, ax3):
#                 ax.set_extent(extent, crs=ccrs.PlateCarree())

#             # plot original pct_mean
#             pct_mean.plot.imshow(ax=ax1, cmap='magma', vmin=pct_lims[0], vmax=pct_lims[1],
#                                 cbar_kwargs=cbar_kwargs)
#             ax1.set_title('')
#             _add_gridlines(ax1, tick_interval=.5)
#             # plot result
#             filled.plot.imshow(ax=ax2, cmap='magma', vmin=pct_lims[0], vmax=pct_lims[1],
#                             cbar_kwargs=cbar_kwargs)
#             # annotate the parameters as small white text
#             ax2.text(0.5, 0.05, f'k={k}, power={power}, dist={distance_upper_bound}',
#                     transform=ax2.transAxes, color='white', fontsize=10, ha='center', va='bottom',
#                     fontweight='bold')
#             ax2.set_title('')   
#             _add_gridlines(ax2, tick_interval=.5)


#             # plot diff
#             diff = filled - pct_mean
#             diff.plot.imshow(ax=ax3, cmap='RdBu_r', vmin=diff_lims[0], vmax=diff_lims[1],
#                             cbar_kwargs=cbar_kwargs)
#             ax3.set_title('')
#             _add_gridlines(ax3, tick_interval=.5, color='black')

#             for axis in (ax1, ax2, ax3):
#                 axis.set_xlabel('')
#                 axis.set_ylabel('')
#         plt.subplots_adjust(hspace=0, wspace=.17)
            
#     quick_interp_test_plots([
#         {'k': 16, 'power': 2, 'distance_upper_bound': 0.25,
#         'loss_threshold': 10},
#         {'k': 256, 'power': 2, 'distance_upper_bound': 0.25,
#         'loss_threshold': 10},
#         {'k': 256, 'power': 1, 'distance_upper_bound': 0.25,

#         'loss_threshold': 10},
#         {'k': 256, 'power': 1, 'distance_upper_bound': 0.1,
#         'loss_threshold': 10}
#     ], pct_lims=(35, 36.3), diff_lims=(-0.4, 0.4))


# #%%%


# # def plot_gapfill_example():
# WLAT = -65
# slicer = {'lat': slice(-10.5, -9.5), 'lon': slice(WLAT, WLAT+1)}

# pct, mask, loss = load_data()
# pct = pct / 10.0
# ok = loss.max('time') < LOSS_THRESHOLD
# pct_ok = pct.where(ok)
# out_ds = xr.open_zarr(OUT_PATH, consolidated=False) / 10.0

# fig, axes = plt.subplots(2, 2, figsize=(12, 12), subplot_kw={'projection': ccrs.PlateCarree()})
# # plot the loss
# axes[0, 0].set_title('Max Loss < 10%')
# loss.loc[slicer].max('time').plot.imshow(ax=axes[0, 0], cmap='Reds', vmin=0, vmax=100,
#                                         cbar_kwargs={'label': 'Loss (%)', 'orientation': 'horizontal',
#                                                     'shrink':.5, 'pad':0.05})
# pct.loc[slicer].mean('doy').plot.imshow(ax=axes[0, 1], cmap='magma', vmin=34, vmax=36,
#                                     cbar_kwargs={'label': 'Tmax 95th Percentile (°C)', 'orientation': 'horizontal',
#                                                     'shrink':.5, 'pad':0.05})
# pct.loc[slicer].where(ok).mean('doy').plot.imshow(ax=axes[1, 0], cmap='magma', vmin=34, vmax=36,
#                                     cbar_kwargs={'label': 'Tmax 95th Percentile (°C)', 'orientation': 'horizontal',
#                                                     'shrink':.5, 'pad':0.05})
# (xr.where(mask, np.nan, 1)).loc[slicer].plot.imshow(ax=axes[1,0], cmap='viridis', alpha=0.5, add_colorbar=False)

# out_ds['Ta_TMAX_p95'].loc[slicer].mean('doy').plot.imshow(ax=axes[1, 1], cmap='magma', vmin=34, vmax=36,
#                                     cbar_kwargs={'label': 'Tmax 95th Percentile (°C)', 'orientation': 'horizontal',
#                                                     'shrink':.5, 'pad':0.05})

# plt.tight_layout()

# # plot_gapfill_example()

# #%%

