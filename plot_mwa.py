#%%
import os
import re
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import FIGDIR, STORAGE_DIR

MWA_DAILY_DIR = STORAGE_DIR + 'mwa_results/'
MWA_FIGDIR = FIGDIR + 'mwa_daily_results/'

COARSEN_FACTORS = [None, 2, 5, 10, 25]


def build_stats_sname(
    dep_var_name,
    ind_var_name,
    aggregation,
    ind_var_names=None,
    driver_ind_var_name=None,
    coarsen_factor=None,
):
    if ind_var_names is None:
        ind_var_names = [ind_var_name]
    ind_tag = '+'.join(ind_var_names)
    if driver_ind_var_name is None:
        driver_ind_var_name = ind_var_name

    sname = f'{dep_var_name}_vs_{ind_tag}_driver_{driver_ind_var_name}_{aggregation}'
    if coarsen_factor:
        sname += f'_coarsened{coarsen_factor}'
    return sname


def extract_coarsen_factor(text):
    match = re.search(r'_coarsened(\d+)', text)
    return int(match.group(1)) if match else None


def resolve_member_dirs(
    dep_var_name,
    ind_var_name,
    aggregation,
    ind_var_names=None,
    driver_ind_var_name=None,
    coarsen_factor=None,
):
    sname = build_stats_sname(
        dep_var_name=dep_var_name,
        ind_var_name=ind_var_name,
        aggregation=aggregation,
        ind_var_names=ind_var_names,
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


def members_dir_to_df(members_dir, years=None):
    parquet_paths = _parquet_partitions(members_dir, years=years)
    if not parquet_paths:
        raise FileNotFoundError(f'No parquet partitions found under {members_dir}')

    cols = ['year', 'dep_value', 'bin_left', 'bin_right']
    df = pd.concat(
        [pd.read_parquet(path, columns=cols) for path in parquet_paths],
        ignore_index=True,
    )

    statsdf = (
        df.groupby(['bin_left', 'bin_right'])['dep_value']
        .agg(
            count='size',
            sum='sum',
            mean='mean',
            median='median',
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
            min='min',
            max='max',
        )
        .reset_index()
    )
    statsdf['sumsq'] = (
        df.groupby(['bin_left', 'bin_right'])['dep_value']
        .apply(lambda s: np.square(s.to_numpy(dtype=np.float64)).sum())
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

    intervals = pd.IntervalIndex.from_arrays(
        statsdf['bin_left'],
        statsdf['bin_right'],
        closed='right',
    )
    statsdf = statsdf.set_index(intervals).sort_index()

    return df, statsdf


def format_count(count):
    if count >= 1e6:
        return f'{count/1e6:.1f}M'
    if count >= 1e3:
        return f'{count/1e3:.1f}k'
    return f'{count:.0f}'


def plot_binned_results(
    ind_var_name,
    dep_var_name,
    aggregation,
    coarsen_factor=None,
    ind_var_names=None,
    driver_ind_var_name=None,
    years=None,
    stat='mean',
    ylabel='neighbour mean temperature change (degC)',
    xlabel='canopy cover loss bin (%)',
):
    sname, member_dirs = resolve_member_dirs(
        dep_var_name=dep_var_name,
        ind_var_name=ind_var_name,
        aggregation=aggregation,
        ind_var_names=ind_var_names,
        driver_ind_var_name=driver_ind_var_name,
        coarsen_factor=coarsen_factor,
    )

    _, statsdf = members_dir_to_df(member_dirs[0], years=years)

    if stat not in statsdf.columns:
        raise ValueError(f"Unsupported stat '{stat}'. Available: {sorted(statsdf.columns.tolist())}")

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_kwargs = {'ax': ax, 'color': 'steelblue'}
    if stat == 'mean':
        plot_kwargs['yerr'] = [statsdf['yerr_lower'], statsdf['yerr_upper']]
        plot_kwargs['capsize'] = 4
    elif stat == 'median':
        plot_kwargs['yerr'] = [statsdf['median_yerr_lower'], statsdf['median_yerr_upper']]
        plot_kwargs['capsize'] = 4
    statsdf[stat].plot.bar(**plot_kwargs)

    xticklabels = [f'{int(interval.left)}-{int(interval.right)}' for interval in statsdf.index]

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticklabels(xticklabels, rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    bars = ax.patches
    _, y_upper = ax.get_ylim()
    for bar, count in zip(bars, statsdf['count']):
        if not np.isnan(count):
            count_text = format_count(count)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_upper * 0.95,
                count_text,
                ha='center',
                va='bottom',
                fontsize=10,
            )

    if coarsen_factor is not None:
        ax.set_title(f'coarsen_factor: {coarsen_factor}')

    os.makedirs(MWA_FIGDIR, exist_ok=True)
    suffix = '' if years is None else f"_years_{min(years)}-{max(years)}"
    png_path = f'{sname}{suffix}.png'
    fig.savefig(os.path.join(MWA_FIGDIR, png_path), dpi=400, bbox_inches='tight')
    print(f'Saved figure to {os.path.join(MWA_FIGDIR, png_path)}')

    return fig, ax


def load_binned_results(
    dep_var_name,
    ind_var_name,
    aggregation,
    ind_var_names=None,
    driver_ind_var_name=None,
    coarsen_factor=None,
    years=None,
):
    sname, member_dirs = resolve_member_dirs(
        dep_var_name=dep_var_name,
        ind_var_name=ind_var_name,
        aggregation=aggregation,
        ind_var_names=ind_var_names,
        driver_ind_var_name=driver_ind_var_name,
        coarsen_factor=coarsen_factor,
    )

    dfs = {}
    for members_dir in member_dirs:
        _, statsdf = members_dir_to_df(members_dir, years=years)
        base = os.path.basename(members_dir).replace('_binned_members_parquet', '')
        cf = extract_coarsen_factor(base)
        dfs[cf] = statsdf

    return sname, dfs


def plot_mwa_results_all(
    ind_var_name,
    dep_var_name,
    aggregation,
    ind_var_names=None,
    driver_ind_var_name=None,
    years=None,
    stat='mean',
    ylabel=None,
    xlabel='canopy cover loss bin (%)',
):
    sname, dfs = load_binned_results(
        dep_var_name=dep_var_name,
        ind_var_name=ind_var_name,
        aggregation=aggregation,
        ind_var_names=ind_var_names,
        driver_ind_var_name=driver_ind_var_name,
        years=years,
    )

    cfs = sorted(dfs.keys(), key=lambda value: -1 if value is None else value)
    if not cfs:
        raise ValueError(f'No valid parquet outputs to plot for {sname}')

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.magma(np.linspace(0.25, 0.75, len(cfs)))

    all_bins = set()
    for df in dfs.values():
        all_bins.update(df.index)
    sorted_bins = sorted(all_bins, key=lambda interval: (interval.left, interval.right))

    x = np.arange(len(sorted_bins))

    for i, cf in enumerate(cfs):
        df = dfs[cf].reindex(sorted_bins)
        label = 'original' if cf is None else f'approx {cf} km'

        if stat not in df.columns:
            raise ValueError(f"Unsupported stat '{stat}'. Available: {sorted(df.columns.tolist())}")

        yerr = None
        if stat == 'mean':
            yerr = [df['yerr_lower'].values, df['yerr_upper'].values]
        elif stat == 'median':
            yerr = [df['median_yerr_lower'].values, df['median_yerr_upper'].values]

        ax.errorbar(
            x,
            df[stat].values,
            label=label,
            color=colors[i],
            alpha=0.85,
            yerr=yerr,
            capsize=2,
        )

    if ylabel is None:
        ylabel = f'mean difference in {dep_var_name}'

    xticklabels = [f'{int(interval.left)}-{int(interval.right)}' for interval in sorted_bins]

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(title='downsampled\nresolution', bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.set_title(f'{dep_var_name} ({aggregation.replace("_", " ")}) vs {ind_var_name}', fontsize=14)
    fig.tight_layout()

    os.makedirs(MWA_FIGDIR, exist_ok=True)
    suffix = '' if years is None else f"_years_{min(years)}-{max(years)}"
    png_path = f'{sname}_all_coarsen_factors{suffix}.png'
    fig.savefig(os.path.join(MWA_FIGDIR, png_path), dpi=400, bbox_inches='tight')
    print(f'Saved figure to {os.path.join(MWA_FIGDIR, png_path)}')

    return fig, ax


def plot_mwa_results(
    ind_var_name,
    dep_var_name,
    aggregation,
    xlabel,
    ylabel,
    ind_var_names=None,
    driver_ind_var_name=None,
    years=None,
    stat='mean',
):
    plot_mwa_results_all(
        ind_var_name=ind_var_name,
        dep_var_name=dep_var_name,
        aggregation=aggregation,
        ind_var_names=ind_var_names,
        driver_ind_var_name=driver_ind_var_name,
        years=years,
        stat=stat,
        xlabel=xlabel,
        ylabel=ylabel,
    )


#%%
if __name__ == '__main__':
    common_args = {
        'ind_var_name': 'cc_loss',
        'ind_var_names': ['cc_loss'],
        'driver_ind_var_name': 'cc_loss',
        'xlabel': 'canopy cover loss bin (%)',
    }

    plot_mwa_results(
        dep_var_name='tmax',
        aggregation='annual_mean',
        ylabel='mean difference in annual mean tmax (degC)',
        stat='mean',
        **common_args,
    )
