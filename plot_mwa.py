#%%
import os
import re
from glob import glob
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import contextily as ctx
from scipy.fftpack import tilbert

from constants import FIGDIR, STORAGE_DIR

MWA_DAILY_DIR = STORAGE_DIR + 'mwa_results/'
MWA_FIGDIR = FIGDIR + 'mwa_daily_results/'

COARSEN_FACTORS = [None, 2, 5, 10, 25]

AMAZON_SLICE = dict(westlimit=-65.7, southlimit=-18.48, eastlimit=-48.86, northlimit=-4.56)

def build_stats_sname(
    dep_var_name,
    ind_var_name,
    aggregation,
    driver_ind_var_name=None,
    coarsen_factor=None,
):
    if driver_ind_var_name is None:
        driver_ind_var_name = ind_var_name

    sname = f'{dep_var_name}_vs_{ind_var_name}_driver_{driver_ind_var_name}_{aggregation}'
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
    driver_ind_var_name=None,
    coarsen_factor=None,
):
    sname = build_stats_sname(
        dep_var_name=dep_var_name,
        ind_var_name=ind_var_name,
        aggregation=aggregation,
        driver_ind_var_name=driver_ind_var_name,
        coarsen_factor=coarsen_factor,
    )

    exact_dir = os.path.join(MWA_DAILY_DIR, f'{sname}_binned_members_parquet')
    
    if not os.path.isdir(exact_dir):
        raise FileNotFoundError(f'No binned member parquet folder found at {exact_dir}')
    return sname, exact_dir


def _parquet_partitions(members_dir, years=None):
    if years is None:
        return sorted(glob(os.path.join(members_dir, 'year=*', '*.parquet')))

    parquet_paths = []
    for year in years:
        year_dir = os.path.join(members_dir, f'year={int(year)}')
        parquet_paths.extend(sorted(glob(os.path.join(year_dir, '*.parquet'))))
    return parquet_paths


def member_dir_to_df(member_dir, years=None, area_slice=None):
    parquet_paths = _parquet_partitions(member_dir, years=years)
    if not parquet_paths:
        raise FileNotFoundError(f'No parquet partitions found under {member_dir}')

    cols = ['year', 'dep_diff', 'bin_left', 'bin_right', 'lat', 'lon', 'ind_diff',
            'ind_center', 'ind_neighbor', 'dep_center', 'dep_neighbor', 'bin_label']
    df = pd.concat(
        [pd.read_parquet(path, columns=cols) for path in parquet_paths],
        ignore_index=True,
    )

    
    if area_slice is not None:
        df = df[
            (df['lon'] >= area_slice['westlimit']) &
            (df['lon'] <= area_slice['eastlimit']) &
            (df['lat'] >= area_slice['southlimit']) &
            (df['lat'] <= area_slice['northlimit'])
        ]

    statsdf = (
        df.groupby(['bin_left', 'bin_right'])['dep_diff']
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
        df.groupby(['bin_left', 'bin_right'])['dep_diff']
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
    driver_ind_var_name=None,
    years=None,
    stat='mean',
    ylabel='neighbour mean temperature change (degC)',
    xlabel='canopy cover loss bin (%)',
):
    sname, member_dir = resolve_member_dirs(
        dep_var_name=dep_var_name,
        ind_var_name=ind_var_name,
        aggregation=aggregation,
        driver_ind_var_name=driver_ind_var_name,
        coarsen_factor=coarsen_factor,
    )

    _, statsdf = member_dir_to_df(member_dir, years=years)

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

def plot_map_results(
    ind_var_name,
    dep_var_name,
    aggregation,
    coarsen_factor=None,
    driver_ind_var_name=None,
    canopy_cover_loss_bin='90-100',
    cbar_label='mean difference in annual sum of heatwaves (heatwave days/year)'
):


    sname, member_dir = resolve_member_dirs(
            dep_var_name=dep_var_name,
            ind_var_name=ind_var_name,
            aggregation=aggregation,
            driver_ind_var_name=driver_ind_var_name,
            coarsen_factor=coarsen_factor,
        )

    df, _ = member_dir_to_df(member_dir, years=None)
    df['lon'] = df['lon'].round(0)
    df['lat'] = df['lat'].round(0)
    # select canopy cover loss bin
    interval = pd.Interval(left=int(canopy_cover_loss_bin.split('-')[0]), right=int(canopy_cover_loss_bin.split('-')[1]), closed='right')
    df = df[df['bin_label'] == str(interval)]
    # take mean across years
    mdf = df.groupby(['lat', 'lon'])['dep_diff'].mean()
    cdf = df.groupby(['lat', 'lon'])['dep_diff'].count()

    cmap = plt.get_cmap('coolwarm')
    llim, ulim = mdf.quantile([0.05, 0.95])
    lim = max(abs(llim), abs(ulim))
    norm = plt.Normalize(vmin=-lim, vmax=lim)
    size_norm = plt.Normalize(vmin=cdf.min(), vmax=cdf.max())
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(projection=ccrs.PlateCarree())
    sc = ax.scatter(
        mdf.index.get_level_values('lon'),
        mdf.index.get_level_values('lat'),
        c=mdf.values,
        cmap=cmap,
        norm=norm,
        s=size_norm(cdf.values)*1000,
        edgecolor='none',
        linewidth=0.5,
        alpha=.8
    )
    ax.coastlines()
    ctx.add_basemap(ax, crs=ccrs.PlateCarree(), source=ctx.providers.CartoDB.Voyager, alpha=0.5)
    # add colorbar
    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.02, aspect=30)
    cbar.set_label(cbar_label, fontsize=12)
    ax.set_title(f'{dep_var_name} ({aggregation.replace("_", " ")}) vs {ind_var_name} - coarsen_factor: {coarsen_factor} - bin: {canopy_cover_loss_bin}', fontsize=14)


def plot_hist_results(
    ind_var_name,
    dep_var_name,
    aggregation,
    coarsen_factor=None,
    driver_ind_var_name=None,
    canopy_cover_loss_bin='90-100',
    bins=np.arange(-5, 5, 0.02),
    ylabel='neighbour mean temperature change (degC)',
    xlabel='canopy cover loss bin (%)',
):

    sname, member_dir = resolve_member_dirs(
        dep_var_name=dep_var_name,
        ind_var_name=ind_var_name,
        aggregation=aggregation,
        driver_ind_var_name=driver_ind_var_name,
        coarsen_factor=coarsen_factor,
    )

    df, _ = member_dir_to_df(member_dir, years=None)
    
    # convert string to interval
    bin_left, bin_right = map(int, canopy_cover_loss_bin.split('-'))
    bdf = df.where(df['bin_left'] == bin_left).where(df['bin_right'] == bin_right).dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    bdf['dep_diff'].plot.hist(ax=ax, color='steelblue', bins=bins, label=f'canopy cover loss bin: {canopy_cover_loss_bin}')
    ax.set_xlim(bdf['dep_diff'].quantile(.001), bdf['dep_diff'].quantile(.999))
    # plot the median
    median = bdf['dep_diff'].median()
    ax.axvline(median, color='red', linestyle='--', label=f'Median: {median:.2f}')
    # add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{dep_var_name} ({aggregation.replace("_", " ")}) vs {ind_var_name} - coarsen_factor: {coarsen_factor}')

    return fig, ax


def plot_hexbin_results(
    ind_var_name,
    dep_var_name,
    aggregation,
    coarsen_factor=None,
    driver_ind_var_name=None,
    years=None,
    area_slice=None,
    normalize_by_ind_diff=True,
    gridsize=30,
    cmap='viridis',
    reduce_C_function=np.mean,
    mincnt=1000,
    vmin=None,
    vmax=None,
    xlabel='canopy cover loss in center bin (%)',
    ylabel='canopy cover loss in neighbor bin (%)',
    cbar_label=None,
):
    sname, member_dir = resolve_member_dirs(
        dep_var_name=dep_var_name,
        ind_var_name=ind_var_name,
        aggregation=aggregation,
        driver_ind_var_name=driver_ind_var_name,
        coarsen_factor=coarsen_factor,
    )

    df, _ = member_dir_to_df(member_dir, years=years, area_slice=area_slice)
    if normalize_by_ind_diff:
        valid = df['ind_diff'] != 0
        cvals = (df.loc[valid, 'dep_diff'] / df.loc[valid, 'ind_diff']).to_numpy()
        xvals = df.loc[valid, 'ind_center'].to_numpy()
        yvals = df.loc[valid, 'ind_neighbor'].to_numpy()
        if cbar_label is None:
            cbar_label = f'{dep_var_name} difference per 1% {ind_var_name}'
    else:
        cvals = df['dep_diff'].to_numpy()
        xvals = df['ind_center'].to_numpy()
        yvals = df['ind_neighbor'].to_numpy()
        if cbar_label is None:
            cbar_label = f'mean {dep_var_name} difference'

    fig, ax = plt.subplots(figsize=(9, 8))
    hb = ax.hexbin(
        x=xvals,
        y=yvals,
        C=cvals,
        gridsize=gridsize,
        cmap=cmap,
        reduce_C_function=reduce_C_function,
        vmin=vmin,
        vmax=vmax,
        mincnt=mincnt,
    )
    plt.colorbar(hb, ax=ax, label=cbar_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    norm_label = 'normalized' if normalize_by_ind_diff else 'raw'
    ax.set_title(
        f'{dep_var_name} vs {ind_var_name} ({aggregation.replace("_", " ")}) '
        f'- coarsen_factor: {coarsen_factor} - {norm_label} hexbin'
    )

    return fig, ax


def plot_dep_response_vs_ind_diff(
    ind_var_name,
    dep_var_name,
    aggregation,
    coarsen_factor=None,
    driver_ind_var_name=None,
    years=None,
    area_slice=None,
    normalize_by_ind_diff=True,
    round_digits=0,
    xlabel=None,
    ylabel=None,
):
    sname, member_dir = resolve_member_dirs(
        dep_var_name=dep_var_name,
        ind_var_name=ind_var_name,
        aggregation=aggregation,
        driver_ind_var_name=driver_ind_var_name,
        coarsen_factor=coarsen_factor,
    )

    df, _ = member_dir_to_df(member_dir, years=years, area_slice=area_slice)
    if normalize_by_ind_diff:
        valid = df['ind_diff'] != 0
        response = df.loc[valid, 'dep_diff'] / df.loc[valid, 'ind_diff']
        ind_vals = df.loc[valid, 'ind_diff']
        if ylabel is None:
            ylabel = f'{dep_var_name} difference per 1% {ind_var_name}'
    else:
        response = df['dep_diff']
        ind_vals = df['ind_diff']
        if ylabel is None:
            ylabel = f'mean {dep_var_name} difference'

    summary = response.groupby(ind_vals.round(round_digits)).mean().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    summary.plot(ax=ax, marker='o', color='steelblue')

    if xlabel is None:
        xlabel = f'rounded {ind_var_name} difference (%)'

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    norm_label = 'normalized' if normalize_by_ind_diff else 'raw'
    ax.set_title(
        f'{dep_var_name} vs {ind_var_name} ({aggregation.replace("_", " ")}) '
        f'- coarsen_factor: {coarsen_factor} - {norm_label} grouped by ind_diff'
    )

    mean_response = response.mean()
    unit = 'per 1% change' if normalize_by_ind_diff else 'difference'
    print(f'Average {dep_var_name} response ({unit}): {mean_response:.5f}')

    return fig, ax, summary

def plot_trend_results(
    ind_var_name,
    dep_var_name,
    aggregation,
    coarsen_factor=None,
    driver_ind_var_name=None,
    ylabel='neighbour mean temperature change (degC)',
    xlabel='canopy cover loss bin (%)'):

    sname, member_dir = resolve_member_dirs(
        dep_var_name=dep_var_name,
        ind_var_name=ind_var_name,
        aggregation=aggregation,
        driver_ind_var_name=driver_ind_var_name,
        coarsen_factor=coarsen_factor,
    )

    df, _ = member_dir_to_df(member_dir, years=None)
    # make interval columns from bin_left and bin_right
    df['bin_interval'] = pd.IntervalIndex.from_arrays(df['bin_left'], df['bin_right'], closed='right')
    
    # group by year and compute the mean of dep_diff
    dep_per_ccloss = df['dep_diff'] / df['ind_diff']
    trend_df = dep_per_ccloss.groupby(df.year).mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    trend_df.plot(ax=ax, color='steelblue', marker='o')
    ax.set_xlabel('year', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_title(f'{dep_var_name} ({aggregation.replace("_", " ")}) vs {ind_var_name} - coarsen_factor: {coarsen_factor}')
    ax.set_xticks(range(int(trend_df.index.min()), int(trend_df.index.max()) + 1, 3))

    return fig, ax  

# plot_trend_results(
#     ind_var_name='cc_loss',
#     dep_var_name='tmax',
#     aggregation='annual_mean',
#     coarsen_factor=None,
#     driver_ind_var_name='cc_loss',
#     ylabel='neighbour mean temperature change (degC)',
#     xlabel='canopy cover loss bin (%)'
# )


def plot_binned_violin(
    ind_var_name,
    dep_var_name,
    aggregation,
    coarsen_factor,
    driver_ind_var_name=None,
    years=None,
    ylabel='distribution of dependent-variable differences',
    xlabel='canopy cover loss bin (%)',
    max_points_per_bin=None,
    show_p5_p95=True,
):
    """Plot one violin per bin for a single coarsen factor.

    This uses raw row-level values from the binned-members parquet output.
    """
    sname, member_dir = resolve_member_dirs(
        dep_var_name=dep_var_name,
        ind_var_name=ind_var_name,
        aggregation=aggregation,
        driver_ind_var_name=driver_ind_var_name,
        coarsen_factor=coarsen_factor,
    )
    raw_df, statsdf = member_dir_to_df(member_dir, years=years)

    grouped = raw_df.groupby(['bin_left', 'bin_right'])['dep_diff']
    violin_data = []
    xticklabels = []
    p5_vals = []
    p50_vals = []
    p95_vals = []

    for interval in statsdf.index:
        key = (float(interval.left), float(interval.right))
        if key not in grouped.groups:
            continue
        vals_full = grouped.get_group(key).to_numpy()
        p5, p50, p95 = np.percentile(vals_full, [5, 50, 95])
        p5_vals.append(p5)
        p50_vals.append(p50)
        p95_vals.append(p95)

        vals = vals_full
        if max_points_per_bin is not None and vals_full.size > max_points_per_bin:
            # Randomly downsample very large bins to keep plotting responsive.
            rng = np.random.default_rng(42)
            vals = rng.choice(vals_full, size=max_points_per_bin, replace=False)
        violin_data.append(vals)
        xticklabels.append(f'{int(interval.left)}-{int(interval.right)}')

    if not violin_data:
        raise ValueError(f'No data available for violin plot: {sname}')

    fig, ax = plt.subplots(figsize=(12, 6))
    vp = ax.violinplot(
        violin_data,
        positions=np.arange(1, len(violin_data) + 1),
        showmeans=False,
        showmedians=True,
        showextrema=False,
        widths=0.85,
    )
    for body in vp['bodies']:
        body.set_facecolor('steelblue')
        body.set_edgecolor('black')
        body.set_alpha(0.45)

    if show_p5_p95:
        positions = np.arange(1, len(violin_data) + 1)
        ax.vlines(
            positions,
            p5_vals,
            p95_vals,
            colors='black',
            linewidth=1.2,
            alpha=0.9,
            label='5th-95th percentile',
        )
        ax.scatter(
            positions,
            p50_vals,
            color='black',
            s=12,
            zorder=3,
            label='median',
        )
        ax.legend(loc='upper left')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(np.arange(1, len(xticklabels) + 1))
    ax.set_xticklabels(xticklabels, rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    cf_label = 'original' if coarsen_factor is None else f'coarsen_factor: {coarsen_factor}'
    ax.set_title(f'{dep_var_name} ({aggregation.replace("_", " ")}) vs {ind_var_name} - {cf_label}')
    fig.tight_layout()

    os.makedirs(MWA_FIGDIR, exist_ok=True)
    suffix = '' if years is None else f"_years_{min(years)}-{max(years)}"
    cf_suffix = 'original' if coarsen_factor is None else f'coarsened{coarsen_factor}'
    png_path = f'{sname}_{cf_suffix}_violin{suffix}.png'
    fig.savefig(os.path.join(MWA_FIGDIR, png_path), dpi=400, bbox_inches='tight')
    print(f'Saved figure to {os.path.join(MWA_FIGDIR, png_path)}')

    return fig, ax


def load_binned_results(
    dep_var_name,
    ind_var_name,
    aggregation,
    driver_ind_var_name=None,
    coarsen_factor=None,
    years=None,
):
    if driver_ind_var_name is None:
        driver_ind_var_name = ind_var_name

    # Find all matching coarsen factor directories
    base_sname = f'{dep_var_name}_vs_{ind_var_name}_driver_{driver_ind_var_name}_{aggregation}'
    pattern = os.path.join(MWA_DAILY_DIR, f'{base_sname}*_binned_members_parquet')
    member_dirs = sorted([p for p in glob(pattern) if os.path.isdir(p)])
    
    if not member_dirs:
        raise FileNotFoundError(f'No binned member parquet folders found for pattern {pattern}')

    dfs = {}
    for members_dir in member_dirs:
        _, statsdf = member_dir_to_df(members_dir, years=years)
        base = os.path.basename(members_dir).replace('_binned_members_parquet', '')
        cf = extract_coarsen_factor(base)
        dfs[cf] = statsdf

    return base_sname, dfs


def plot_mwa_results_all(
    ind_var_name,
    dep_var_name,
    aggregation,
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

#%%

def make_all_plots(ind_var_name, dep_var_name, aggregation, driver_ind_var_name=None, years=None):
    plot_mwa_results_all(
        ind_var_name=ind_var_name,
        dep_var_name=dep_var_name,
        aggregation=aggregation,
        driver_ind_var_name=driver_ind_var_name,
        years=years,
        stat='mean',
        ylabel=f'mean difference in {dep_var_name}',
    )

make_all_plots(
    ind_var_name='cc_loss',
    dep_var_name='tmax',
    aggregation='annual_mean',
    driver_ind_var_name='cc_loss',
    years=None,
)


#%%
sname, member_dir = resolve_member_dirs(
        dep_var_name='tmax',
        ind_var_name='cc_loss',
        aggregation='annual_mean',
        driver_ind_var_name='cc_loss',
        coarsen_factor=25,
    )

df, _ = member_dir_to_df(member_dir, years=None)

