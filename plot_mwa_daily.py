#%%
import os
import pickle
import re

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from constants import FIGDIR, STORAGE_DIR
from glob import glob

MWA_DAILY_DIR = STORAGE_DIR + 'mwa_results/'
MWA_FIGDIR = FIGDIR + 'mwa_daily_results/'

COARSEN_FACTORS = [None, 2, 5, 10, 25]


def build_stats_sname(dep_var_name, ind_var_name, aggregation,
                      ind_var_names=None, driver_ind_var_name=None,
                      coarsen_factor=None):
    if ind_var_names is None:
        ind_var_names = [ind_var_name]
    ind_tag = '+'.join(ind_var_names)
    if driver_ind_var_name is None:
        driver_ind_var_name = ind_var_name

    sname = f'{dep_var_name}_vs_{ind_tag}_driver_{driver_ind_var_name}_{aggregation}'
    if coarsen_factor:
        sname += f'_coarsened{coarsen_factor}'
    return sname


def extract_coarsen_factor(path):
    match = re.search(r'_coarsened(\d+)_binned_stats\.P$', os.path.basename(path))
    return int(match.group(1)) if match else None


def resolve_stats_pickles(dep_var_name, ind_var_name, aggregation,
                          ind_var_names=None, driver_ind_var_name=None,
                          coarsen_factor=None):
    sname = build_stats_sname(
        dep_var_name=dep_var_name,
        ind_var_name=ind_var_name,
        aggregation=aggregation,
        ind_var_names=ind_var_names,
        driver_ind_var_name=driver_ind_var_name,
        coarsen_factor=coarsen_factor,
    )
    pattern = os.path.join(MWA_DAILY_DIR, f'{sname}_binned_stats.P')
    matches = sorted(glob(pattern))
    if not matches:
        raise FileNotFoundError(f'No stats pickle found for pattern {pattern}')
    return sname, matches





def format_count(count):
    if count >= 1e6:
        return f"{count/1e6:.1f}M"
    elif count >= 1e3:
        return f"{count/1e3:.1f}k"
    else:
        return f"{count:.0f}"

def pickle_to_df(pickle_path):
        # read pickle
        with open(pickle_path, 'rb') as f:
            ds = pickle.load(f)
            ds = ds.rename({'cc_loss_cdiff_bins':'bins'})

        df = ds.to_dataframe()

        df['se'] = df['std'] / np.sqrt(df['count'])
        df['ci_lower'] = df['mean'] - 1.96 * df['se']
        df['ci_upper'] = df['mean'] + 1.96 * df['se']
        
        # Calculate error bar heights
        df['yerr_lower'] = df['mean'] - df['ci_lower']
        df['yerr_upper'] = df['ci_upper'] - df['mean']

        meandf = pd.DataFrame(index=df.index.get_level_values('bins').unique().sort_values())
        meandf['sum'] = df.groupby('bins')['sum'].sum() 
        meandf['count'] = df.groupby('bins')['count'].sum()
        meandf['mean'] = meandf['sum'] / meandf['count']
        meandf['sumsq'] = df.groupby('bins')['sumsq'].sum()
        meandf['std'] = np.sqrt(meandf['sumsq'] / meandf['count'] - meandf['mean']**2)
        meandf['se'] = meandf['std'] / np.sqrt(meandf['count'])
        meandf['ci_lower'] = meandf['mean'] - 1.96 * meandf['se']
        meandf['ci_upper'] = meandf['mean'] + 1.96 * meandf['se']
        meandf['yerr_lower'] = meandf['mean'] - meandf['ci_lower']
        meandf['yerr_upper'] = meandf['ci_upper'] - meandf['mean']
        meandf = meandf.sort_index()

        return df, meandf

def plot_binned_results(ind_var_name, dep_var_name, aggregation, coarsen_factor=None,
                        ind_var_names=None, driver_ind_var_name=None,
                        ylabel='neighbour mean temperature change (°C)',
                        xlabel='canopy cover loss bin (%)'):

    sname, pickle_paths = resolve_stats_pickles(
        dep_var_name=dep_var_name,
        ind_var_name=ind_var_name,
        aggregation=aggregation,
        ind_var_names=ind_var_names,
        driver_ind_var_name=driver_ind_var_name,
        coarsen_factor=coarsen_factor,
    )
    pickle_path = pickle_paths[0]
    df, meandf = pickle_to_df(pickle_path)
    df = meandf

    # make readable xtick labels from pandas interval index
    xticklabels = [f"{int(interval.left)}-{int(interval.right)}" for interval in df.index]

    
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    df['mean'].plot.bar(ax=ax, color='steelblue', 
                        yerr=[df['yerr_lower'], df['yerr_upper']], capsize=4)

    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    # make nice x-tick labels from interval index
    ax.set_xticklabels(xticklabels, rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add pixel counts to top of axis
    bars = ax.patches
    _, yuplim = ax.get_ylim()
    for i, (bar, count) in enumerate(zip(bars, df['count'])):
        if not np.isnan(count):
            # convert count to scientific notation if > 1000
            count_text = format_count(count)
            ax.text(bar.get_x() + bar.get_width() / 2, yuplim*.95, count_text, ha='center', va='bottom', fontsize=10)
        
    png_path = f'{sname}.png'
    if png_path:
        fig.savefig(MWA_FIGDIR+png_path, dpi=400, bbox_inches='tight')
        print(f"Saved figure to {png_path}")

    if coarsen_factor is not None:
        ax.set_title(f"coarsen_factor: {coarsen_factor}")
    return fig, ax


def plot_mwa_results_all(ind_var_name, dep_var_name, aggregation,
                        ind_var_names=None, driver_ind_var_name=None,
                        ylabel=None,
                        xlabel='canopy cover loss bin (%)'):

    sname, pickle_paths = resolve_stats_pickles(
        dep_var_name=dep_var_name,
        ind_var_name=ind_var_name,
        aggregation=aggregation,
        ind_var_names=ind_var_names,
        driver_ind_var_name=driver_ind_var_name,
        coarsen_factor=None,
    )

    # If only the uncoarsened file was matched exactly, try to include sibling coarsened files.
    if len(pickle_paths) == 1:
        stem = sname
        if '_coarsened' in stem:
            stem = stem.split('_coarsened')[0]
        expanded = sorted(glob(os.path.join(MWA_DAILY_DIR, f'{stem}*_binned_stats.P')))
        if expanded:
            pickle_paths = expanded

    dfs = {}
    for pickle_path in pickle_paths:
        df, meandf = pickle_to_df(pickle_path)
        df = meandf
        coarsen_factor = extract_coarsen_factor(pickle_path)
        dfs[coarsen_factor] = df

    cfs = sorted(dfs.keys(), key=lambda value: -1 if value is None else value)
    n_cfs = len(cfs)
    if n_cfs == 0:
        raise ValueError(f"No valid CSVs to plot for {sname}")

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.magma(np.linspace(.25, .75, n_cfs))

    # Build a shared x-axis from the union of all df indexes
    bin_labels = set()
    for df in dfs.values():
        bin_labels.update(df.index)

    # make readable xtick labels from pandas interval index
    xticklabels = [f"{int(interval.left)}-{int(interval.right)}" for interval in sorted(bin_labels)]
    

    n_bins = len(bin_labels)

    bar_width = 0.8 / n_cfs
    x = np.arange(n_bins)

    for i, cf in enumerate(cfs):
        df = dfs[cf]
        label = 'original' if cf is None else f'≈{cf} km'
        

        

        ax.errorbar(
            x,
            df['mean'].values,
            label=label,
            color=colors[i],
            alpha=0.85,
            yerr=[df['yerr_lower'].values, df['yerr_upper'].values],
            capsize=2,
        )

    if ylabel is None:
        ylabel = f'mean difference in {dep_var_name}'

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(title='downsampled\nresolution', bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.set_title(f'{dep_var_name} vs {ind_var_name} by downsampled resolution')
    fig.tight_layout()

    png_path = f'{sname}_all_coarsen_factors.png'
    fig.savefig(MWA_FIGDIR + png_path, dpi=400, bbox_inches='tight')
    print(f"Saved figure to {MWA_FIGDIR + png_path}")

    return fig, ax




def plot_mwa_results(ind_var_name, dep_var_name, aggregation,
                     xlabel, ylabel,
                     ind_var_names=None, driver_ind_var_name=None,
                     coarsen_factors=COARSEN_FACTORS):
    

    # for coarsen_factor in coarsen_factors:
    #     plot_binned_results(ind_var_name=ind_var_name, dep_var_name=dep_var_name, coarsen_factor=coarsen_factor,
    #                         aggregation=aggregation,
    #                                  ylabel=ylabel,
    #                                  xlabel=xlabel)
    plot_mwa_results_all(ind_var_name=ind_var_name, dep_var_name=dep_var_name,
                         aggregation=aggregation,
                                 ind_var_names=ind_var_names,
                                 driver_ind_var_name=driver_ind_var_name,
                                 xlabel=xlabel,
                                 ylabel=ylabel)


if __name__ == '__main__':
    common_args = {
        'ind_var_name': 'cc_loss',
        'ind_var_names': ['cc_loss'],
        'driver_ind_var_name': 'cc_loss',
        'xlabel': 'canopy cover loss bin (%)',
    }

    # plot_mwa_results(
    #     dep_var_name='tmax',
    #     aggregation='annual_95pct',
    #     ylabel='mean difference in annual 95th percentile tmax (°C)',
    #     **common_args,
    # )

    plot_mwa_results(
        dep_var_name='tmax',
        aggregation='annual_mean',
        ylabel='mean difference in annual mean tmax (°C)',
        **common_args,
    )

    # plot_mwa_results(
    #     dep_var_name='heatwaves',
    #     aggregation='annual_sum',
    #     ylabel='mean difference in annual heatwave count',
    #     **common_args,
    # )

    # plot_mwa_results(
    #     dep_var_name='heatwaves_smoothed',
    #     aggregation='annual_sum',
    #     ylabel='mean difference in annual heatwave count',
    #     **common_args,
    # )