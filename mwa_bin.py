#%%
import os
import json
import numpy as np
import pandas as pd
import xarray as xr
from constants import MWA_DAILY_DIR
from ufuncs import local_cluster
from glob import glob

COARSEN_FACTORS = [None, 2, 5, 10, 25]

_IND_VAR_BINS = {
    'cc_loss': np.arange(0, 110, 10),
    'forest_loss_age': np.arange(0, 18, 1),
}


westlimit=-69.7; southlimit=-13.7; eastlimit=-46.4; northlimit=5.6
RESULTS_BBOX = {'lat':slice(southlimit, northlimit), 'lon':slice(westlimit, eastlimit)}

def retrieve_ind_dep_results(sname, ind_var_name, dep_var_name):

    # get file paths for all versions of the combined zarr files
    zarr_filepaths = sorted(glob(os.path.join(MWA_DAILY_DIR, f'{sname}_mwa_results*.zarr')))
    if not zarr_filepaths:
        raise FileNotFoundError(f'No combined MWA result stores found for {sname}')

    all_inds = [xr.open_zarr(path, consolidated=False)[ind_var_name + '_cdiff'] for path in zarr_filepaths]
    all_deps = [xr.open_zarr(path, consolidated=False)[dep_var_name + '_cdiff'] for path in zarr_filepaths]


    # Concatenate across versions (if multiple)
    if len(all_inds) > 1:
        ind = xr.concat(all_inds, dim='version', join='outer')
        dep = xr.concat(all_deps, dim='version', join='outer')
    else:
        ind = all_inds[0]
        dep = all_deps[0]

    return ind, dep

def get_sname(dep_var_name, ind_var_name, driver_ind_var_name, aggregation, coarsen_factor=None):
    """Construct a standardized sname for binned statistics results."""
    ind_tag = '+'.join(ind_var_name) if isinstance(ind_var_name, list) else ind_var_name
    sname = f'{dep_var_name}_vs_{ind_tag}_driver_{driver_ind_var_name}_{aggregation}'
    if coarsen_factor:
        sname += f'_coarsened{coarsen_factor}'
    return sname


def _get_bins(ind_var_name):
    bins = _IND_VAR_BINS.get(ind_var_name)
    if bins is None:
        raise ValueError(f"No binning specified for variable '{ind_var_name}'")
    return bins


def _as_versioned(arr):
    if 'version' in arr.dims:
        return arr
    return arr.expand_dims(version=[0])


def _year_path(base_dir, year):
    return os.path.join(base_dir, f'year={int(year)}')


def export_binned_members(
    dep_var_name,
    ind_var_name,
    sname,
    out_dir=None,
    compression='snappy',
):
    """Export one row per valid pixel-year member in each independent-variable bin.

    Output columns:
    year, version, lat, lon, ind_value, dep_value, bin_left, bin_right, bin_label.
    """
    bins = _get_bins(ind_var_name)
    ind, dep = retrieve_ind_dep_results(sname, ind_var_name, dep_var_name)
    ind = _as_versioned(ind.loc[RESULTS_BBOX])
    dep = _as_versioned(dep.loc[RESULTS_BBOX])

    if out_dir is None:
        out_dir = os.path.join(MWA_DAILY_DIR, f'{sname}_binned_members_parquet')
    os.makedirs(out_dir, exist_ok=True)

    run_rows = []

    for version in ind['version'].values:
        ind_v = ind.sel(version=version)
        dep_v = dep.sel(version=version)

        lat_vals = ind_v['lat'].values
        lon_vals = ind_v['lon'].values
        lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing='ij')
        lat_flat = lat_grid.ravel()
        lon_flat = lon_grid.ravel()

        for year in ind_v['year'].values:
            ind_y = np.asarray(ind_v.sel(year=year).values).ravel()
            dep_y = np.asarray(dep_v.sel(year=year).values).ravel()

            valid = np.isfinite(ind_y) & np.isfinite(dep_y)
            if not np.any(valid):
                continue

            ind_valid = ind_y[valid]
            dep_valid = dep_y[valid]
            lat_valid = lat_flat[valid]
            lon_valid = lon_flat[valid]

            binned = pd.cut(ind_valid, bins=bins, right=True)
            codes = binned.codes
            valid_bins = codes >= 0
            if not np.any(valid_bins):
                continue

            codes = codes[valid_bins]
            intervals = binned.categories
            left = intervals.left.values[codes]
            right = intervals.right.values[codes]
            labels = np.array([str(intervals[idx]) for idx in codes], dtype=object)

            dep_kept = dep_valid[valid_bins]
            ind_kept = ind_valid[valid_bins]
            lat_kept = lat_valid[valid_bins]
            lon_kept = lon_valid[valid_bins]

            out_df = pd.DataFrame(
                {
                    'year': np.full(dep_kept.shape[0], int(year), dtype=np.int16),
                    'version': np.full(dep_kept.shape[0], int(version), dtype=np.int16),
                    'lat': lat_kept.astype(np.float32),
                    'lon': lon_kept.astype(np.float32),
                    'ind_value': ind_kept.astype(np.float32),
                    'dep_value': dep_kept.astype(np.float32),
                    'bin_left': left.astype(np.float32),
                    'bin_right': right.astype(np.float32),
                    'bin_label': labels,
                }
            )

            year_dir = _year_path(out_dir, year)
            os.makedirs(year_dir, exist_ok=True)
            out_path = os.path.join(year_dir, f'part_version={int(version)}.parquet')

            try:
                out_df.to_parquet(out_path, index=False, compression=compression)
            except Exception as exc:
                raise RuntimeError(
                    'Failed to write parquet. Ensure pyarrow or fastparquet is installed.'
                ) from exc

            run_rows.append(
                {
                    'sname': sname,
                    'year': int(year),
                    'version': int(version),
                    'n_members': int(out_df.shape[0]),
                    'path': out_path,
                }
            )
            print(f'Wrote {out_df.shape[0]} rows to {out_path}')

    metadata = {
        'sname': sname,
        'dep_var_name': dep_var_name,
        'ind_var_name': ind_var_name,
        'bins': bins.tolist(),
        'bbox': {
            'westlimit': westlimit,
            'southlimit': southlimit,
            'eastlimit': eastlimit,
            'northlimit': northlimit,
        },
        'columns': [
            'year',
            'version',
            'lat',
            'lon',
            'ind_value',
            'dep_value',
            'bin_left',
            'bin_right',
            'bin_label',
        ],
    }
    with open(os.path.join(out_dir, '_metadata.json'), 'w', encoding='utf-8') as fobj:
        json.dump(metadata, fobj, indent=2)

    summary_df = pd.DataFrame(run_rows).sort_values(['year', 'version']) if run_rows else pd.DataFrame()
    if not summary_df.empty:
        summary_path = os.path.join(out_dir, '_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f'Saved run summary to {summary_path}')

    return summary_df


def calc_members_all_coarsens(
    dep_var_name,
    ind_var_names,
    driver_ind_var_name,
    ind_var_name_for_bins,
    aggregation='annual_mean',
    coarsen_factors=COARSEN_FACTORS,
):
    """Export row-level binned members for all requested coarsen factors."""
    all_summaries = {}
    for coarsen_factor in coarsen_factors:
        sname = get_sname(dep_var_name, ind_var_names, driver_ind_var_name, aggregation, coarsen_factor)
        print(f'Exporting row-level binned members for {sname}...')
        summary_df = export_binned_members(
            dep_var_name=dep_var_name,
            ind_var_name=ind_var_name_for_bins,
            sname=sname,
        )
        all_summaries[sname] = summary_df
    return all_summaries


def load_binned_members(sname, years=None, columns=None):
    """Load one or more yearly parquet partitions for interactive analysis."""
    root = os.path.join(MWA_DAILY_DIR, f'{sname}_binned_members_parquet')
    if not os.path.isdir(root):
        raise FileNotFoundError(f'No parquet output folder found at {root}')

    if years is None:
        year_dirs = sorted(glob(os.path.join(root, 'year=*')))
    else:
        year_dirs = [os.path.join(root, f'year={int(year)}') for year in years]

    files = []
    for year_dir in year_dirs:
        files.extend(sorted(glob(os.path.join(year_dir, '*.parquet'))))

    if not files:
        raise FileNotFoundError(f'No parquet partitions found for {sname}')

    return pd.concat([pd.read_parquet(path, columns=columns) for path in files], ignore_index=True)


if __name__ == '__main__':
    with local_cluster(
        n_workers=24,
        threads_per_worker=1,
        memory_limit='6GB',
        suppress_dask_warnings=False,
        use_dask_dashboard=True,
    ):
        # calc_members_all_coarsens(
        #     dep_var_name='tmax',
        #     ind_var_names=['cc_loss'],
        #     driver_ind_var_name='cc_loss',
        #     ind_var_name_for_bins='cc_loss',
        #     aggregation='annual_mean',
        #     coarsen_factors=COARSEN_FACTORS,
        # )
        calc_members_all_coarsens(
            dep_var_name='heatwaves',
            ind_var_names=['cc_loss'],
            driver_ind_var_name='cc_loss',
            ind_var_name_for_bins='cc_loss',
            aggregation='annual_mean',
            coarsen_factors=COARSEN_FACTORS,
        )
        calc_members_all_coarsens(
            dep_var_name='heatwaves_gapfilled',
            ind_var_names=['cc_loss'],
            driver_ind_var_name='cc_loss',
            ind_var_name_for_bins='cc_loss',
            aggregation='annual_mean',
            coarsen_factors=COARSEN_FACTORS,
        )