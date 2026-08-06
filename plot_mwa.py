#%%
import os
import re
from glob import glob
from turtle import width
import cartopy.crs as ccrs
from matplotlib import cm
import matplotlib.pyplot as plt
from mercantile import xy_bounds
import numpy as np
import pandas as pd
import contextily as ctx
from constants import FIGDIR, STORAGE_DIR, IND_VAR_BINS, RESULTS_BBOX

MWA_DAILY_DIR = STORAGE_DIR + 'mwa_results/'
MWA_FIGDIR = FIGDIR + 'mwa_daily_results/'

COARSEN_FACTORS = [None, 2, 5, 10, 25]

# AMAZON_SLICE = dict(westlimit=-65.7, southlimit=-18.48, eastlimit=-48.86, northlimit=-4.56)

top_bins = {'cc_loss': '90-100', 'forest_loss_age': '16-17'}

pretty = {
    'heatwaves_gapfilled': 'gap-filled heatwaves',
    'heatwaves': 'heatwaves',
    'forest_loss_age': 'forest loss age',
    'cc_loss': 'canopy cover loss',
    'tmax': 'daily maximum temperature',
}

units = {
    'heatwaves_gapfilled': 'days/year',
    'heatwaves': 'days/year',
    'cc_loss': '%',
    'tmax': '°C',
    'forest_loss_age': 'years',
}


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

    stat_col_name = 'dep_diff'
    
    if area_slice is not None:
        df = df[
            (df['lon'] >= area_slice['westlimit']) &
            (df['lon'] <= area_slice['eastlimit']) &
            (df['lat'] >= area_slice['southlimit']) &
            (df['lat'] <= area_slice['northlimit'])
        ]

    statsdf = (
        df.groupby(['bin_left', 'bin_right'])[stat_col_name]
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
        df.groupby(['bin_left', 'bin_right'])[stat_col_name]
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
    area_slice=None,
    relative=False,
    print_stats=False,
):
    sname, member_dir = resolve_member_dirs(
        dep_var_name=dep_var_name,
        ind_var_name=ind_var_name,
        aggregation=aggregation,
        driver_ind_var_name=driver_ind_var_name,
        coarsen_factor=coarsen_factor,
    )

    df, statsdf = member_dir_to_df(member_dir, years=years, area_slice=area_slice)

    bin_means = (
        df.groupby(['bin_left', 'bin_right'])
        .agg(
            dep_neighbor_mean=('dep_neighbor', 'mean'),
            dep_center_mean=('dep_center', 'mean'),
        )
        .reset_index()
    )
    interval_index = statsdf.index
    statsdf = statsdf.merge(bin_means, on=['bin_left', 'bin_right'], how='left')
    statsdf.index = interval_index

    if relative:
        # Convert per-bin dep_diff stats to percent relative to that bin's
        # average dep_neighbor to avoid row-level divide-by-zero blowups.
        eps = 1e-12
        scale = np.where(np.abs(statsdf['dep_neighbor_mean']) > eps,
                         100.0 / statsdf['dep_neighbor_mean'],
                         np.nan)

        cols_to_scale = [
            'sum', 'mean', 'median', 'q25', 'q75', 'min', 'max',
            'std', 'se', 'ci_lower', 'ci_upper',
            'yerr_lower', 'yerr_upper',
            'median_yerr_lower', 'median_yerr_upper',
        ]
        for col in cols_to_scale:
            statsdf[col] = statsdf[col] * scale

    if stat not in statsdf.columns:
        raise ValueError(f"Unsupported stat '{stat}'. Available: {sorted(statsdf.columns.tolist())}")

    if print_stats:
        print('Key stats by bin:')
        print(f"stat column: {stat} | spread column: std")
        out = pd.DataFrame({
            'bin': [f'{int(interval.left)}-{int(interval.right)}' for interval in statsdf.index],
            'y': statsdf[stat].values,
            'std': statsdf['std'].values,
            'count': statsdf['count'].values,
            'dep_neighbor_mean': statsdf['dep_neighbor_mean'].values,
            'dep_center_mean': statsdf['dep_center_mean'].values,
        })
        print(out.to_string(index=False))

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

    if relative:
        ylabel = f'mean relative difference in {pretty[dep_var_name]} [%]'
    else:
        ylabel = f'mean difference in {pretty[dep_var_name]} [{units[dep_var_name]}]'

    ax.set_xlabel(f'{pretty[ind_var_name]} loss bin [%]', fontsize=12)
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
    rel_suffix = '_relative' if relative else ''
    png_path = f'{sname}{suffix}{rel_suffix}.png'
    fig.savefig(os.path.join(MWA_FIGDIR, png_path), dpi=400, bbox_inches='tight')
    print(f'Saved figure to {os.path.join(MWA_FIGDIR, png_path)}')

    return fig, ax

def plot_map_results(
    ind_var_name,
    dep_var_name,
    aggregation,
    coarsen_factor=None,
    driver_ind_var_name=None,
    area_slice=None,
    ind_bin='90-100',
    show_box=False
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
    interval = pd.Interval(left=int(ind_bin.split('-')[0]), right=int(ind_bin.split('-')[1]), closed='right')
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

    mapped_colors = cmap(norm(mdf.values))
    sc = ax.scatter(
    mdf.index.get_level_values("lon"),
    mdf.index.get_level_values("lat"),
    facecolors="none",
    edgecolors=mapped_colors,  # Pass mapped RGBA colors to edges
    s=size_norm(cdf.values) * 250,
    linewidths=1.5,  # Plural 'linewidths' & slightly thicker for visibility
    alpha=1,
    )
    ax.coastlines()
    ctx.add_basemap(ax, crs=ccrs.PlateCarree(), source=ctx.providers.CartoDB.Voyager, alpha=0.5)
    # add colorbar
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical',
                        shrink=0.5 )
    cbar.set_label(f'mean difference in\n{pretty[dep_var_name]} [{units[dep_var_name]}]', fontsize=10)
    ax.set_title('')

    if show_box and area_slice is not None:
        # Draw a rectangle around the specified area
        ax.add_patch(plt.Rectangle((area_slice['westlimit'], area_slice['southlimit']),
                                   area_slice['eastlimit'] - area_slice['westlimit'],
                                   area_slice['northlimit'] - area_slice['southlimit'],
                                   fill=False, edgecolor='black', linewidth=2, zorder=5))

    fig.savefig(os.path.join(MWA_FIGDIR, f'{sname}_map_{ind_bin}.png'), dpi=400, bbox_inches='tight')

def plot_hist_results(
    ind_var_name,
    dep_var_name,
    aggregation,
    coarsen_factor=None,
    driver_ind_var_name=None,
    ind_bin='90-100',
    bins=np.arange(-5, 5, 0.02),
    area_slice=None,

):

    sname, member_dir = resolve_member_dirs(
        dep_var_name=dep_var_name,
        ind_var_name=ind_var_name,
        aggregation=aggregation,
        driver_ind_var_name=driver_ind_var_name,
        coarsen_factor=coarsen_factor,
    )

    df, _ = member_dir_to_df(member_dir, years=None, area_slice=area_slice)
    
    # convert string to interval
    bin_left, bin_right = map(int, ind_bin.split('-'))
    bdf = df.where(df['bin_left'] == bin_left).where(df['bin_right'] == bin_right).dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    bdf['dep_diff'].plot.hist(ax=ax, color='steelblue', bins=bins, label=f'canopy cover loss bin: {ind_bin}')
    ax.set_xlim(bdf['dep_diff'].dropna().quantile(.001), bdf['dep_diff'].dropna().quantile(.999))
    # plot the median
    median = bdf['dep_diff'].median()
    ax.axvline(median, color='red', linestyle='--', label=f'Median: {median:.2f}')
    # add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()
    ax.set_xlabel(f'{pretty[ind_var_name]} loss bin [%]', fontsize=12)
    ax.set_ylabel(f'mean difference in {pretty[dep_var_name]} [{units[dep_var_name]}]', fontsize=12)
    ax.set_title(f'{pretty[dep_var_name]} ({aggregation.replace("_", " ")}) vs {pretty[ind_var_name]} - coarsen_factor: {coarsen_factor}')
    fig.savefig(os.path.join(MWA_FIGDIR, f'{sname}_hist_{ind_bin}.png'), dpi=400, bbox_inches='tight')
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

    else:
        cvals = df['dep_diff'].to_numpy()
        xvals = df['ind_center'].to_numpy()
        yvals = df['ind_neighbor'].to_numpy()


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
    plt.colorbar(hb, ax=ax, label=f'{pretty[dep_var_name]} difference per 1% {pretty[ind_var_name]}', orientation='vertical')
    ax.set_xlabel(f'{pretty[ind_var_name]} in centre bin [%]', fontsize=12)
    ax.set_ylabel(f'{pretty["cc_loss"]} in neighbor bin [%]', fontsize=12)

    norm_label = 'normalized' if normalize_by_ind_diff else 'raw'
    ax.set_title(
        f'{dep_var_name} vs {ind_var_name} ({aggregation.replace("_", " ")}) '
        f'- coarsen_factor: {coarsen_factor} - {norm_label} hexbin'
    )

    fig.savefig(os.path.join(MWA_FIGDIR, f'{sname}_hexbin.png'), dpi=400, bbox_inches='tight')

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
    xbin_cutoff=10,
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
        quantifier = 'per 1% change '
    else:
        response = df['dep_diff']
        ind_vals = df['ind_diff']
        quantifier = ''

    cuts = pd.cut(df['ind_diff'], IND_VAR_BINS[ind_var_name])
    summary = response.groupby(cuts).agg(['mean', 'count', 'std'])
    summary['se'] = summary['std'] / np.sqrt(summary['count'])

    fig, ax = plt.subplots(figsize=(10, 6))
    # summary.plot(ax=ax, marker='o', color='steelblue')

    summary['mean'].plot(ax=ax, marker='o', color='black')




    ax.set_xlabel(f'{pretty[ind_var_name]} difference [{units[ind_var_name]}]')
    ax.set_ylabel(f'{pretty[dep_var_name]} difference {quantifier}')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    ax.set_title('')

    mean_response = response.mean()
    unit = 'per 1% change' if normalize_by_ind_diff else 'difference'
    print(f'Average {pretty[dep_var_name]} response ({unit}): {mean_response.mean():.5f}')
    fig.savefig(os.path.join(MWA_FIGDIR, f'{sname}_dep_response_vs_ind_diff.png'), dpi=400, bbox_inches='tight')
    return fig, ax, summary

# ind_var_name = 'cc_loss'
# dep_var_name = 'tmax'
# aggregation = 'annual_mean'
# coarsen_factor = None
# driver_ind_var_name = 'cc_loss'
# years = None
# area_slice = RESULTS_BBOX
# normalize_by_ind_diff = True
# round_digits = 0
# xbin_cutoff = 10
# plot_dep_response_vs_ind_diff(
#     ind_var_name=ind_var_name,
#     dep_var_name=dep_var_name,
#     aggregation=aggregation,
#     coarsen_factor=coarsen_factor,
#     driver_ind_var_name=driver_ind_var_name,
#     years=years,
#     area_slice=area_slice,
#     normalize_by_ind_diff=normalize_by_ind_diff,
#     round_digits=round_digits,
#     xbin_cutoff=xbin_cutoff,
# )

def plot_trend_results(
    ind_var_name,
    dep_var_name,
    aggregation,
    coarsen_factor=None,
    driver_ind_var_name=None,

    area_slice=None):

    sname, member_dir = resolve_member_dirs(
        dep_var_name=dep_var_name,
        ind_var_name=ind_var_name,
        aggregation=aggregation,
        driver_ind_var_name=driver_ind_var_name,
        coarsen_factor=coarsen_factor,
    )

    df, _ = member_dir_to_df(member_dir, years=None, area_slice=area_slice)
    # make interval columns from bin_left and bin_right
    df['bin_interval'] = pd.IntervalIndex.from_arrays(df['bin_left'], df['bin_right'], closed='right')
    
    # group by year and compute the mean of dep_diff
    dep_per_ccloss = df['dep_diff'] / df['ind_diff']
    trend_df = dep_per_ccloss.groupby(df.year).mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    trend_df.plot(ax=ax, color='steelblue', marker='o')
    ax.set_xlabel('year', fontsize=12)
    ax.set_ylabel(f'mean {pretty[dep_var_name]} response per % {pretty[ind_var_name]}', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_title(f'{pretty[dep_var_name]} ({aggregation.replace("_", " ")}) vs {pretty[ind_var_name]} \n coarsen_factor: {coarsen_factor}')
    ax.set_xticks(range(int(trend_df.index.min()), int(trend_df.index.max()) + 1, 3))

    fig.savefig(os.path.join(MWA_FIGDIR, f'{sname}_trend.png'), dpi=400, bbox_inches='tight')
    return fig, ax  



def load_binned_results(
    dep_var_name,
    ind_var_name,
    aggregation,
    driver_ind_var_name=None,
    coarsen_factor=None,
    years=None,
    area_slice=None,
    relative=False,
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
        df, statsdf = member_dir_to_df(members_dir, years=years, area_slice=area_slice)

        bin_means = (
            df.groupby(['bin_left', 'bin_right'])
            .agg(
                dep_neighbor_mean=('dep_neighbor', 'mean'),
                dep_center_mean=('dep_center', 'mean'),
            )
            .reset_index()
        )
        interval_index = statsdf.index
        statsdf = statsdf.merge(bin_means, on=['bin_left', 'bin_right'], how='left')
        statsdf.index = interval_index

        if relative:
            # Convert per-bin dep_diff stats to percent relative to that bin's
            # average dep_neighbor to avoid divide-by-zero at row level.
            eps = 1e-12
            scale = np.where(np.abs(statsdf['dep_neighbor_mean']) > eps,
                             100.0 / statsdf['dep_neighbor_mean'],
                             np.nan)
            cols_to_scale = [
                'sum', 'mean', 'median', 'q25', 'q75', 'min', 'max',
                'std', 'se', 'ci_lower', 'ci_upper',
                'yerr_lower', 'yerr_upper',
                'median_yerr_lower', 'median_yerr_upper',
            ]
            for col in cols_to_scale:
                statsdf[col] = statsdf[col] * scale

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
    area_slice=None,
    min_count_per_bin=20,
    print_summary=False,
    plot_resolution_corr=True,
    relative=False,
    print_stats=False,
):
    sname, dfs = load_binned_results(
        dep_var_name=dep_var_name,
        ind_var_name=ind_var_name,
        aggregation=aggregation,
        driver_ind_var_name=driver_ind_var_name,
        years=years,
        area_slice=area_slice,
        relative=relative,
    )

    cfs = sorted(dfs.keys(), key=lambda value: -1 if value is None else value)
    if not cfs:
        raise ValueError(f'No valid parquet outputs to plot for {sname}')

    if not plot_resolution_corr:
        fig, ax = plt.subplots(figsize=(14, 6))
    else:
        fig, axes = plt.subplots(figsize=(14, 6), nrows=1, ncols=2,
                                 width_ratios=[3, 2], gridspec_kw={'wspace': 0.15})
        ax = axes[0]
    colors = plt.cm.magma(np.linspace(0.25, 0.75, len(cfs)))

    all_bins = set()
    for df in dfs.values():
        all_bins.update(df.index)
    sorted_bins = sorted(all_bins, key=lambda interval: (interval.left, interval.right))

    x = np.arange(len(sorted_bins))

    for i, cf in enumerate(cfs):
        df = dfs[cf].reindex(sorted_bins)
        label = 'original (~1 km)' if cf is None else f'~{cf} km'

        if stat not in df.columns:
            raise ValueError(f"Unsupported stat '{stat}'. Available: {sorted(df.columns.tolist())}")

        yerr = None
        if stat == 'mean':
            yerr = [df['yerr_lower'].values, df['yerr_upper'].values]
        elif stat == 'median':
            yerr = [df['median_yerr_lower'].values, df['median_yerr_upper'].values]

        y = df[stat].values
        # Mask bins with insufficient count
        y = np.ma.masked_where(df['count'].values < min_count_per_bin, y)
        ax.errorbar(
            x,
            y,
            label=label,
            color=colors[i],
            alpha=0.85,
            yerr=yerr,
            capsize=2,
        )

        if print_summary:
            print(f'Coarsen factor: {cf}')
            print(df[['count', stat, 'yerr_lower', 'yerr_upper']].to_string())

        if print_stats:
            print(f'Key stats for coarsen factor {cf}:')
            out = pd.DataFrame({
                'bin': [f'{int(interval.left)}-{int(interval.right)}' for interval in sorted_bins],
                'y': df[stat].values,
                'std': df['std'].values,
                'count': df['count'].values,
                'dep_neighbor_mean': df['dep_neighbor_mean'].values,
                'dep_center_mean': df['dep_center_mean'].values,
            })
            print(out.to_string(index=False))


    xticklabels = [f'{int(interval.left)}-{int(interval.right)}' for interval in sorted_bins]

    ax.set_xlabel(f'{pretty[ind_var_name]} bin [{units["cc_loss"]}]', fontsize=12)
    if relative:
        ax.set_ylabel(f'mean relative difference in {pretty[dep_var_name]} [%]', fontsize=12)
    else:
        ax.set_ylabel(f'mean difference in {pretty[dep_var_name]} [{units[dep_var_name]}]', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(title='downsampled\nresolution', bbox_to_anchor=(0, 1), loc='upper left')
    ax.set_title(f'{pretty[dep_var_name]} ({aggregation.replace("_", " ")}) vs {pretty[ind_var_name]}', fontsize=14)
    fig.tight_layout()


    if plot_resolution_corr:
        ax2 = axes[1]
        psr = pd.Series(index=cfs, dtype=float)
        for i, cf in enumerate(cfs):
            df = dfs[cf]
            if relative:
                nmean = df['mean'].mean()
            else:
                # get normalised mean
                nmean = (df['mean'] / df.index.mid).mean()
            ax2.plot(nmean, cf if cf is not None else 1, 
                     marker='o', color=colors[i])
            psr.loc[cf if cf is not None else 1] = nmean

        ax2.grid(linestyle='--', alpha=0.7)
        if relative:
            ax2.set_xlabel(f'mean relative difference in\n{pretty[dep_var_name]} [%]', fontsize=12)
        else:
            ax2.set_xlabel(f'normalised mean difference in\n{pretty[dep_var_name]}\n[{units[dep_var_name]} per % canopy cover loss]', fontsize=12)
        ax2.set_ylabel('downsampleed resolution (km)', fontsize=12)




    os.makedirs(MWA_FIGDIR, exist_ok=True)
    suffix = '' if years is None else f"_years_{min(years)}-{max(years)}"
    rel_suffix = '_relative' if relative else ''
    png_path = f'{sname}_all_coarsen_factors{suffix}{rel_suffix}.png'
    fig.savefig(os.path.join(MWA_FIGDIR, png_path), dpi=400, bbox_inches='tight')
    print(f'Saved figure to {os.path.join(MWA_FIGDIR, png_path)}')

    return fig, ax

# ind_var_name = 'cc_loss'
# dep_var_name = 'tmax'
# aggregation = 'annual_mean'
# driver_ind_var_name = 'cc_loss'
# years = None
# stat = 'mean'
# area_slice = RESULTS_BBOX
# print_summary = False
# min_count_per_bin = 20
# plot_resolution_corr = True
# plot_mwa_results_all(
#     ind_var_name=ind_var_name,
#     dep_var_name=dep_var_name,
#     aggregation=aggregation,
#     driver_ind_var_name=driver_ind_var_name,
#     years=years,
#     stat=stat,
#     area_slice=area_slice,
#     print_summary=print_summary,
#     min_count_per_bin=min_count_per_bin,
#     plot_resolution_corr=plot_resolution_corr
# )




def make_all_plots(ind_var_name, dep_var_name, aggregation, driver_ind_var_name=None, years=None,
                   area_slice=None):

    # plot_hist_results(
    #     ind_var_name=ind_var_name,
    #     dep_var_name=dep_var_name,
    #     aggregation=aggregation,
    #     driver_ind_var_name=driver_ind_var_name,
    #     area_slice=area_slice,
    #     ind_bin=top_bins[ind_var_name],
    # )


    plot_mwa_results_all(
        ind_var_name=ind_var_name,
        dep_var_name=dep_var_name,
        aggregation=aggregation,
        driver_ind_var_name=driver_ind_var_name,
        years=years,
        stat='mean',
        area_slice=area_slice,
        plot_resolution_corr=False,
        print_summary=True,
        min_count_per_bin=20,
        relative=False,
        print_stats=True
    )


    # plot_binned_results(
    #     ind_var_name=ind_var_name,
    #     dep_var_name=dep_var_name,
    #     aggregation=aggregation,
    #     driver_ind_var_name=driver_ind_var_name,
    #     years=years,
    #     stat='mean',
    #     area_slice=area_slice,
    #     relative=False,
    #     print_stats=True
    # )

    # plot_dep_response_vs_ind_diff(

    #     ind_var_name=ind_var_name,
    #     dep_var_name=dep_var_name,
    #     aggregation=aggregation,
    #     driver_ind_var_name=driver_ind_var_name,
    #     years=years,
    #     area_slice=area_slice,
    #     normalize_by_ind_diff=True,
    #     round_digits=0,

    # )

    # plot_map_results(
    #     ind_var_name=ind_var_name,
    #     dep_var_name=dep_var_name,
    #     aggregation=aggregation,
    #     driver_ind_var_name=driver_ind_var_name,
    #     ind_bin=top_bins[ind_var_name],
    #     show_box=True,
    #     area_slice=area_slice
    # )

    # plot_trend_results(
    #     ind_var_name=ind_var_name,
    #     dep_var_name=dep_var_name,
    #     aggregation=aggregation,
    #     driver_ind_var_name=driver_ind_var_name,
    #     area_slice=area_slice,
    # )

    # plot_hexbin_results(
    #     ind_var_name=ind_var_name,
    #     dep_var_name=dep_var_name,
    #     aggregation=aggregation,
    #     driver_ind_var_name=driver_ind_var_name,
    #     years=years,
    #     area_slice=area_slice,
    #     normalize_by_ind_diff=True,
    #     gridsize=30,
    #     cmap='viridis',
    #     reduce_C_function=np.mean,
    #     mincnt=1000,
    #     vmin=None,
    #     vmax=None,
    # )   



make_all_plots(
    ind_var_name='cc_loss',
    dep_var_name='tmax',
    aggregation='annual_mean',
    driver_ind_var_name='cc_loss',
    years=None,
    area_slice=RESULTS_BBOX
)

# make_all_plots(
#     ind_var_name='forest_loss_age',
#     dep_var_name='tmax',
#     aggregation='annual_mean',
#     driver_ind_var_name='forest_loss_age',
#     years=None,
#     area_slice=RESULTS_BBOX
# )

# make_all_plots(
#     ind_var_name='cc_loss',
#     dep_var_name='heatwaves_gapfilled',
#     aggregation='annual_sum',
#     driver_ind_var_name='cc_loss',
#     years=None,
#     area_slice=RESULTS_BBOX
# )


# #%%
# years = None
# dep_var_name = 'tmax'
# ind_var_name = 'cc_loss'
# aggregation = 'annual_mean'
# area_slice = RESULTS_BBOX
# driver_ind_var_name = 'cc_loss'
# coarsen_factor = 5

# sname, member_dir = resolve_member_dirs(
#     dep_var_name=dep_var_name,
#     ind_var_name=ind_var_name,
#     aggregation=aggregation,
#     driver_ind_var_name=driver_ind_var_name,
#     coarsen_factor=coarsen_factor,
# )

# df, _ = member_dir_to_df(member_dir, years=years, area_slice=area_slice)
# df['bin_label'] = pd.IntervalIndex.from_arrays(df['bin_left'], df['bin_right'], closed='right')
# #%%
# res = {}
# for cc_bin in range(10, 100, 10):
#     print(cc_bin)
#     bdf = df.loc[(df.ind_neighbor >=cc_bin) & (df.ind_neighbor < cc_bin+10)]
#     binned = (bdf['dep_diff']).groupby(bdf.ind_diff.round(-1)).mean()
#     # binned.index = binned.index.mid
#     res[cc_bin] = binned


# #%%
# fig, ax = plt.subplots(figsize=(10, 6))
# cmap = plt.get_cmap('viridis')
# norm = plt.Normalize(vmin=min(res.keys()), vmax=max(res.keys()))
# for cc_bin, binned in res.items():
#     color = cmap(norm(cc_bin))
#     binned.plot(ax=ax, marker='o', color=color, label=f'{cc_bin}-{cc_bin+10}')

# ax.legend(title='canopy cover loss in neighbor bin (%)', bbox_to_anchor=(1.01, 1), loc='upper left')
# ax.set_xlabel(f'canopy cover loss (%)')
# ax.set_ylabel(f'mean difference in {dep_var_name}')

# #%%
# cdf = pd.DataFrame(res)
# fig, ax = plt.subplots(figsize=(10, 6))
# cdf.div(cdf.columns, axis=1).mean().plot(ax=ax)
# ax.set_ylabel(f'mean difference in {dep_var_name} per 1% canopy cover loss')
# ax.set_xlabel('canopy cover loss in center bin (%)')
