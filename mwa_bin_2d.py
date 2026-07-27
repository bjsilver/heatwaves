#%%
import os
import json
import numpy as np
import pandas as pd
import xarray as xr
from constants import MWA_DAILY_DIR, IND_VAR_BINS
from ufuncs import local_cluster
from glob import glob

COARSEN_FACTORS = [None, 2, 5, 10, 25]


def retrieve_ind_dep_results(
    dep_var_name,
    ind_var_names,
    driver_ind_var_name,
    aggregation,
    coarsen_factor,
):
    """Load independent/dependent MWA outputs for two independent variables."""
    if len(ind_var_names) != 2:
        raise ValueError('mwa_bin_2d.py expects exactly two independent variables.')

    ind_das = []
    dep = None

    for ind_var_name in ind_var_names:
        sname = get_sname(
            dep_var_name,
            ind_var_name,
            driver_ind_var_name,
            aggregation,
            coarsen_factor,
        )

        zarr_filepaths = sorted(glob(os.path.join(MWA_DAILY_DIR, f'{sname}_mwa_results*.zarr')))
        if not zarr_filepaths:
            raise FileNotFoundError(f'No combined MWA result stores found for {sname}')

        all_inds = [xr.open_zarr(path, consolidated=False)[ind_var_name + '_cdiff'] for path in zarr_filepaths]
        all_deps = [xr.open_zarr(path, consolidated=False)[dep_var_name + '_cdiff'] for path in zarr_filepaths]

        if len(all_inds) > 1:
            ind_da = xr.concat(all_inds, dim='version', join='outer')
            dep_da = xr.concat(all_deps, dim='version', join='outer')
        else:
            ind_da = all_inds[0]
            dep_da = all_deps[0]

        if dep is None:
            dep = dep_da

        ind_das.append(ind_da)

    ind = xr.Dataset({name: da for name, da in zip(ind_var_names, ind_das)})
    return ind, dep


def get_sname(dep_var_name, ind_var_name, driver_ind_var_name, aggregation, coarsen_factor=None):
    """Construct a standardized sname for binned statistics results."""
    sname = f'{dep_var_name}_vs_{ind_var_name}_driver_{driver_ind_var_name}_{aggregation}'
    if coarsen_factor:
        sname += f'_coarsened{coarsen_factor}'
    return sname


def _get_bins(ind_var_names):
    bins = {}
    for ind_var_name in ind_var_names:
        bins[ind_var_name] = IND_VAR_BINS.get(ind_var_name)
        if bins[ind_var_name] is None:
            raise ValueError(f"No binning specified for variable '{ind_var_name}'")
    return bins


def _as_versioned(arr):
    if 'version' in arr.dims:
        return arr
    return arr.expand_dims(version=[0])


def _year_path(base_dir, year):
    return os.path.join(base_dir, f'year={int(year)}')


def export_binned_members_2d(
    dep_var_name,
    ind_var_names,
    driver_ind_var_name,
    aggregation='annual_mean',
    coarsen_factor=None,
    out_dir=None,
    compression='snappy',
):
    """Export one row per valid pixel-year member for joint 2D independent-variable bins."""
    if len(ind_var_names) != 2:
        raise ValueError('export_binned_members_2d expects exactly two independent variables.')

    bins = _get_bins(ind_var_names)
    ind, dep = retrieve_ind_dep_results(
        dep_var_name=dep_var_name,
        ind_var_names=ind_var_names,
        driver_ind_var_name=driver_ind_var_name,
        aggregation=aggregation,
        coarsen_factor=coarsen_factor,
    )

    ind = _as_versioned(ind)
    dep = _as_versioned(dep)

    ind1_name, ind2_name = ind_var_names
    ind1_bins = bins[ind1_name]
    ind2_bins = bins[ind2_name]

    sname = get_sname(
        dep_var_name,
        '+'.join(ind_var_names),
        driver_ind_var_name,
        aggregation,
        coarsen_factor,
    )

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
            dep_y = np.asarray(dep_v.sel(year=year).values).ravel()
            ind1_y = np.asarray(ind_v[ind1_name].sel(year=year).values).ravel()
            ind2_y = np.asarray(ind_v[ind2_name].sel(year=year).values).ravel()

            valid = np.isfinite(dep_y) & np.isfinite(ind1_y) & np.isfinite(ind2_y)
            if not np.any(valid):
                continue

            ind1_valid = ind1_y[valid]
            ind2_valid = ind2_y[valid]
            dep_valid = dep_y[valid]
            lat_valid = lat_flat[valid]
            lon_valid = lon_flat[valid]

            binned1 = pd.cut(ind1_valid, bins=ind1_bins, right=True)
            binned2 = pd.cut(ind2_valid, bins=ind2_bins, right=True)
            codes1 = binned1.codes
            codes2 = binned2.codes

            valid_bins = (codes1 >= 0) & (codes2 >= 0)
            if not np.any(valid_bins):
                continue

            codes1 = codes1[valid_bins]
            codes2 = codes2[valid_bins]

            intervals1 = binned1.categories
            intervals2 = binned2.categories

            left1 = intervals1.left.values[codes1]
            right1 = intervals1.right.values[codes1]
            labels1 = np.array([str(intervals1[idx]) for idx in codes1], dtype=object)

            left2 = intervals2.left.values[codes2]
            right2 = intervals2.right.values[codes2]
            labels2 = np.array([str(intervals2[idx]) for idx in codes2], dtype=object)

            dep_kept = dep_valid[valid_bins]
            ind1_kept = ind1_valid[valid_bins]
            ind2_kept = ind2_valid[valid_bins]
            lat_kept = lat_valid[valid_bins]
            lon_kept = lon_valid[valid_bins]

            out_df = pd.DataFrame(
                {
                    'year': np.full(dep_kept.shape[0], int(year), dtype=np.int16),
                    'version': np.full(dep_kept.shape[0], int(version), dtype=np.int16),
                    'lat': lat_kept.astype(np.float32),
                    'lon': lon_kept.astype(np.float32),
                    f'{ind1_name}_value': ind1_kept.astype(np.float32),
                    f'{ind2_name}_value': ind2_kept.astype(np.float32),
                    'dep_value': dep_kept.astype(np.float32),
                    f'{ind1_name}_bin_left': left1.astype(np.float32),
                    f'{ind1_name}_bin_right': right1.astype(np.float32),
                    f'{ind1_name}_bin_label': labels1,
                    f'{ind2_name}_bin_left': left2.astype(np.float32),
                    f'{ind2_name}_bin_right': right2.astype(np.float32),
                    f'{ind2_name}_bin_label': labels2,
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
        'ind_var_names': list(ind_var_names),
        'bins': {key: val.tolist() for key, val in bins.items()},
        'columns': [
            'year',
            'version',
            'lat',
            'lon',
            f'{ind1_name}_value',
            f'{ind2_name}_value',
            'dep_value',
            f'{ind1_name}_bin_left',
            f'{ind1_name}_bin_right',
            f'{ind1_name}_bin_label',
            f'{ind2_name}_bin_left',
            f'{ind2_name}_bin_right',
            f'{ind2_name}_bin_label',
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


def calc_members_all_coarsens_2d(
    dep_var_name,
    ind_var_names,
    driver_ind_var_name,
    aggregation='annual_mean',
    coarsen_factors=COARSEN_FACTORS,
):
    """Export row-level binned members for a two-variable setup across coarsen factors."""
    all_summaries = {}
    for coarsen_factor in coarsen_factors:
        sname = get_sname(
            dep_var_name,
            '+'.join(ind_var_names),
            driver_ind_var_name,
            aggregation,
            coarsen_factor,
        )
        print(f'Exporting 2D row-level binned members for {sname}...')
        summary_df = export_binned_members_2d(
            dep_var_name=dep_var_name,
            ind_var_names=ind_var_names,
            driver_ind_var_name=driver_ind_var_name,
            aggregation=aggregation,
            coarsen_factor=coarsen_factor,
        )
        all_summaries[sname] = summary_df
    return all_summaries


def load_binned_members_2d(dep_var_name, ind_var_names, driver_ind_var_name, aggregation, coarsen_factor=None, years=None, columns=None):
    """Load one or more yearly parquet partitions for interactive 2D-bin analysis."""
    sname = get_sname(
        dep_var_name,
        '+'.join(ind_var_names),
        driver_ind_var_name,
        aggregation,
        coarsen_factor,
    )
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


#%%
if __name__ == '__main__':
    with local_cluster(
        n_workers=24,
        threads_per_worker=1,
        memory_limit='6GB',
        suppress_dask_warnings=False,
        use_dask_dashboard=True,
    ):
        calc_members_all_coarsens_2d(
            dep_var_name='tmax',
            ind_var_names=['forest_loss_age', 'cc_loss'],
            driver_ind_var_name='cc_loss',
            aggregation='annual_mean',
            coarsen_factors=COARSEN_FACTORS,
        )

       