#%%
import os
import re
from glob import glob

import cartopy.crs as ccrs
import contextily as ctx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from constants import FIGDIR, STORAGE_DIR

MWA_DAILY_DIR = STORAGE_DIR + 'mwa_results/'
MWA_FIGDIR = FIGDIR + 'mwa_daily_results/'

COARSEN_FACTORS = [None, 2, 5, 10, 25]


def build_stats_sname(
    dep_var_name,
    ind_var_names,
    aggregation,
    driver_ind_var_name=None,
    coarsen_factor=None,
):
    if len(ind_var_names) != 2:
        raise ValueError('ind_var_names must contain exactly two variables')

    if driver_ind_var_name is None:
        driver_ind_var_name = ind_var_names[0]

    ind_combo = '+'.join(ind_var_names)
    sname = f'{dep_var_name}_vs_{ind_combo}_driver_{driver_ind_var_name}_{aggregation}'
    if coarsen_factor:
        sname += f'_coarsened{coarsen_factor}'
    return sname


def extract_coarsen_factor(text):
    match = re.search(r'_coarsened(\d+)', text)
    return int(match.group(1)) if match else None


def resolve_member_dirs(
    dep_var_name,
    ind_var_names,
    aggregation,
    driver_ind_var_name=None,
    coarsen_factor=None,
):
    sname = build_stats_sname(
        dep_var_name=dep_var_name,
        ind_var_names=ind_var_names,
        aggregation=aggregation,
        driver_ind_var_name=driver_ind_var_name,
        coarsen_factor=coarsen_factor,
    )

    exact_dir = os.path.join(MWA_DAILY_DIR, f'{sname}_binned_members_parquet')
    if coarsen_factor is not None:
        if not os.path.isdir(exact_dir):
            raise FileNotFoundError(f'No binned member parquet folder found at {exact_dir}')
        return sname, [exact_dir]

    stem = sname.split('_coarsened')[0]
    pattern = os.path.join(MWA_DAILY_DIR, f'{stem}*_binned_members_parquet')
    matches = sorted(path for path in glob(pattern) if os.path.isdir(path))
    if not matches:
        if os.path.isdir(exact_dir):
            return sname, [exact_dir]
        raise FileNotFoundError(f'No binned member parquet folders found for pattern {pattern}')

    return sname, matches


def _parquet_partitions(members_dir, years=None):
    if years is None:
        return sorted(glob(os.path.join(members_dir, 'year=*', '*.parquet')))

    parquet_paths = []
    for year in years:
        year_dir = os.path.join(members_dir, f'year={int(year)}')
        parquet_paths.extend(sorted(glob(os.path.join(year_dir, '*.parquet'))))
    return parquet_paths


def _parse_bin_text(bin_text):
    left, right = bin_text.split('-')
    return float(left), float(right)


def _schema(ind_var_names):
    ind1_name, ind2_name = ind_var_names
    schema = {
        'ind1_name': ind1_name,
        'ind2_name': ind2_name,
        'ind1_value': f'{ind1_name}_value',
        'ind2_value': f'{ind2_name}_value',
        'ind1_bin_left': f'{ind1_name}_bin_left',
        'ind1_bin_right': f'{ind1_name}_bin_right',
        'ind1_bin_label': f'{ind1_name}_bin_label',
        'ind2_bin_left': f'{ind2_name}_bin_left',
        'ind2_bin_right': f'{ind2_name}_bin_right',
        'ind2_bin_label': f'{ind2_name}_bin_label',
    }
    return schema


def members_dir_to_df(members_dir, ind_var_names, years=None):
    parquet_paths = _parquet_partitions(members_dir, years=years)
    if not parquet_paths:
        raise FileNotFoundError(f'No parquet partitions found under {members_dir}')

    s = _schema(ind_var_names)
    required_cols = [
        'year',
        'version',
        'lat',
        'lon',
        'dep_value',
        s['ind1_value'],
        s['ind2_value'],
        s['ind1_bin_left'],
        s['ind1_bin_right'],
        s['ind1_bin_label'],
        s['ind2_bin_left'],
        s['ind2_bin_right'],
        s['ind2_bin_label'],
    ]

    df = pd.concat(
        [pd.read_parquet(path, columns=required_cols) for path in parquet_paths],
        ignore_index=True,
    )

    group_cols = [
        s['ind1_bin_left'],
        s['ind1_bin_right'],
        s['ind2_bin_left'],
        s['ind2_bin_right'],
    ]

    statsdf = (
        df.groupby(group_cols)['dep_value']
        .agg(
            count='size',
            sum='sum',
            mean='mean',
            median='median',
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            min='min',
            max='max',
        )
        .reset_index()
    )

    statsdf['sumsq'] = (
        df.groupby(group_cols)['dep_value']
        .apply(lambda x: np.square(x.to_numpy(dtype=np.float64)).sum())
        .values
    )
    variance = statsdf['sumsq'] / statsdf['count'] - np.square(statsdf['mean'])
    statsdf['std'] = np.sqrt(np.maximum(variance, 0.0))
    statsdf['se'] = statsdf['std'] / np.sqrt(statsdf['count'])
    statsdf['ci_lower'] = statsdf['mean'] - 1.96 * statsdf['se']
    statsdf['ci_upper'] = statsdf['mean'] + 1.96 * statsdf['se']
    statsdf['yerr_lower'] = statsdf['mean'] - statsdf['ci_lower']
    statsdf['yerr_upper'] = statsdf['ci_upper'] - statsdf['mean']
    statsdf['median_yerr_lower'] = statsdf['median'] - statsdf['q25']
    statsdf['median_yerr_upper'] = statsdf['q75'] - statsdf['median']

    return df, statsdf


def plot_binned_heatmap(
    ind_var_names,
    dep_var_name,
    aggregation,
    coarsen_factor=None,
    driver_ind_var_name=None,
    years=None,
    stat='mean',
    cmap='viridis',
    min_count=100,
):
    sname, member_dirs = resolve_member_dirs(
        dep_var_name=dep_var_name,
        ind_var_names=ind_var_names,
        aggregation=aggregation,
        driver_ind_var_name=driver_ind_var_name,
        coarsen_factor=coarsen_factor,
    )

    _, statsdf = members_dir_to_df(member_dirs[0], ind_var_names=ind_var_names, years=years)
    s = _schema(ind_var_names)

    if stat not in statsdf.columns:
        raise ValueError(f"Unsupported stat '{stat}'. Available: {sorted(statsdf.columns.tolist())}")

    mat = statsdf.pivot_table(
        index=s['ind2_bin_left'],
        columns=s['ind1_bin_left'],
        values=stat,
        aggfunc='mean',
    ).sort_index().sort_index(axis=1)

    count = statsdf.pivot_table(
        index=s['ind2_bin_left'],
        columns=s['ind1_bin_left'],
        values='count',
        aggfunc='sum',
    ).sort_index().sort_index(axis=1)
    count = count.fillna(0)

    mat = mat.where(count >= min_count, np.nan)
    mat = mat.dropna(how='all', axis=0).dropna(how='all', axis=1)

    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(mat.values, cmap=cmap, aspect='auto', origin='lower')

    xlabels = [f'{v:g}' for v in mat.columns.values]
    ylabels = [f'{v:g}' for v in mat.index.values]
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha='right')
    ax.set_yticklabels(ylabels)

    ax.set_xlabel(f"{s['ind1_name']} bin left")
    ax.set_ylabel(f"{s['ind2_name']} bin left")
    ax.set_title(f'{dep_var_name} ({aggregation.replace("_", " ")}) - {stat} coarsened={coarsen_factor}')
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label(f'{stat} dep_value')

    fig.tight_layout()
    os.makedirs(MWA_FIGDIR, exist_ok=True)
    suffix = '' if years is None else f"_years_{min(years)}-{max(years)}"
    cf = 'original' if coarsen_factor is None else f'coarsened{coarsen_factor}'
    outname = f'{sname}_{cf}_{stat}_heatmap{suffix}.png'
    outpath = os.path.join(MWA_FIGDIR, outname)
    fig.savefig(outpath, dpi=350, bbox_inches='tight')
    print(f'Saved figure to {outpath}')

    return fig, ax


if __name__ == '__main__':
    plot_binned_heatmap(
        ind_var_names=['forest_loss_age', 'cc_loss'],
        dep_var_name='tmax',
        aggregation='annual_mean',
        coarsen_factor=2,
        driver_ind_var_name='cc_loss',
        stat='mean',
    )
