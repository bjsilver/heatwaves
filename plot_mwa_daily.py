#%%
import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from constants import FIGDIR, STORAGE_DIR
from glob import glob

MWA_DAILY_DIR = STORAGE_DIR + 'mwa_results/'
MWA_FIGDIR = FIGDIR + 'mwa_daily_results/'

COARSEN_FACTORS = [None, 2, 5, 10, 25]



def sort_bin_labels(bin_labels):
    """Sort bin labels like '0-10', '10-20', ... by their lower bound."""
    return sorted(bin_labels, key=lambda interval: int(str(interval).split('-')[0]))

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
        df = ds.to_pandas()

        df['se'] = df['std'] / np.sqrt(df['count'])
        df['ci_lower'] = df['mean'] - 1.96 * df['se']
        df['ci_upper'] = df['mean'] + 1.96 * df['se']
        
        # Calculate error bar heights
        df['yerr_lower'] = df['mean'] - df['ci_lower']
        df['yerr_upper'] = df['ci_upper'] - df['mean']

        return df

def plot_binned_results(ind_var_name, dep_var_name, time_res, coarsen_factor=None,
                        ylabel='neighbour mean temperature change (°C)',
                        xlabel='canopy cover loss bin (%)'):

    sname = f'{dep_var_name}_vs_{ind_var_name}_{time_res}'
    if coarsen_factor:
        sname += f'_coarsened{coarsen_factor}'
    pickle_path = os.path.join(MWA_DAILY_DIR, f'{sname}_binned_stats.P')
    df = pickle_to_df(pickle_path)
    
    
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


def plot_mwa_results_all(ind_var_name, dep_var_name, time_res,
                        ylabel=None,
                        xlabel=None):
    
    sname = f'{dep_var_name}_vs_{ind_var_name}_{time_res}'

    pickle_paths = sorted(glob(MWA_DAILY_DIR + f'{sname}*_stats.P'))
    if not pickle_paths:
        raise FileNotFoundError(f"No pickle files found for pattern: {MWA_DAILY_DIR}{sname}*_stats.P")

    dfs = {}
    for pickle_path in pickle_paths:
        df = pickle_to_df(pickle_path)
        # Extract coarsen factor from filename
        coarsen_factor = None
        if 'coarsened' in pickle_path:
            coarsen_factor = int(pickle_path.split('coarsened')[-1].split('_')[0])
        dfs[coarsen_factor] = df


    cfs = sorted(dfs.keys(), key=lambda value: -1 if value is None else value)
    n_cfs = len(cfs)
    if n_cfs == 0:
        raise ValueError(f"No valid CSVs to plot for {sname}")

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.Greys(np.linspace(.4, .8, n_cfs))

    # Build a shared x-axis from the union of all df indexes
    bin_labels = set()
    for df in dfs.values():
        # make readable xtick labels from pandas interval index
        xticklabels = [f"{int(interval.left)}-{int(interval.right)}" for interval in df.index]
        bin_labels.update(xticklabels)

    n_bins = len(bin_labels)

    bar_width = 0.8 / n_cfs
    x = np.arange(n_bins)

    for i, cf in enumerate(cfs):
        df = dfs[cf]
        offset = (i - n_cfs / 2 + 0.5) * bar_width
        label = 'original' if cf is None else f'≈{cf} km'
        

        

        ax.bar(
            x + offset,
            df['mean'].values,
            width=bar_width,
            label=label,
            color=colors[i],
            alpha=0.85,
            yerr=[df['yerr_lower'], df['yerr_upper']],
            capsize=4,
        )

    if ylabel is None:
        ylabel = f'mean difference in {dep_var_name}'


    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(sort_bin_labels(bin_labels), rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(title='downsampled\nresolution', bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.set_title(f'{dep_var_name} vs {ind_var_name} by downsampled resolution')
    fig.tight_layout()

    png_path = f'{sname}_all_coarsen_factors.png'
    fig.savefig(MWA_FIGDIR + png_path, dpi=400, bbox_inches='tight')
    print(f"Saved figure to {MWA_FIGDIR + png_path}")

    return fig, ax


def plot_mwa_results(ind_var_name, dep_var_name, time_res,
                     xlabel, ylabel,
                     coarsen_factors=COARSEN_FACTORS):
    
    for coarsen_factor in coarsen_factors:
        plot_binned_results(ind_var_name=ind_var_name, dep_var_name=dep_var_name, coarsen_factor=coarsen_factor,
                            time_res=time_res,
                                     ylabel=ylabel,
                                     xlabel=xlabel)
    plot_mwa_results_all(ind_var_name=ind_var_name, dep_var_name=dep_var_name,
                         time_res=time_res,
                            ylabel=ylabel,
                            xlabel=xlabel)


plot_mwa_results(ind_var_name='cc_loss', dep_var_name='tmax', time_res='annual_mean',
                 xlabel='canopy cover loss bin (%)',
                 ylabel='mean difference in annual mean tmax (°C)')

plot_mwa_results(ind_var_name='cc_loss', dep_var_name='heatwaves', time_res='annual_sum',
                 xlabel='canopy cover loss bin (%)',
                 ylabel='mean difference in annual heatwave count')

plot_mwa_results(ind_var_name='cc_loss', dep_var_name='heatwaves_smoothed', time_res='annual_sum',
                 xlabel='canopy cover loss bin (%)',
                 ylabel='mean difference in annual heatwave count')

plot_mwa_results(ind_var_name='cc_loss', dep_var_name='tmax', time_res='daily',
                 xlabel='canopy cover loss bin (%)',
                 ylabel='mean difference in daily tmax (°C)')