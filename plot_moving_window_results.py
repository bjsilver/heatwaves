#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from constants import FIGDIR, STORAGE_DIR
from glob import glob
MWA_DIR = STORAGE_DIR + 'moving_window_analysis/'
MWA_FIGDIR = FIGDIR + 'mwa_results/'

def sort_bin_labels(bin_labels):
    """Sort bin labels like '0-10', '10-20', ... by their lower bound."""
    return sorted(bin_labels, key=lambda interval: int(str(interval).split('-')[0]))


def plot_binned_results_from_csv(csv_path, png_path=None,
                                 cf=None,
                                 ylabel='neighbour mean temperature change (°C)',
                                 xlabel='canopy cover loss bin (%)'):
    """Read binned statistics from CSV and create a bar plot with error bars.
    
    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path to the CSV file containing bin statistics.
    png_path : str or pathlib.Path, optional
        Path to save the figure. If None, figure is not saved.
    cf : int, optional
        Coarsen factor used for the plot title.
    ylabel : str
        Label for the y-axis.
    xlabel : str
        Label for the x-axis.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Extract data
    bin_labels = df['bin'].values
    mean_diff = df['mean_diff'].values
    ci_lower = df['ci_lower'].values
    ci_upper = df['ci_upper'].values
    counts = df['count'].values
    
    # Calculate error bar heights
    yerr_lower = mean_diff - ci_lower
    yerr_upper = ci_upper - mean_diff
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(bin_labels)), mean_diff, color='steelblue',
                   yerr=[yerr_lower, yerr_upper], capsize=4)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add pixel counts on top of bars
    _, yuplim = ax.get_ylim()
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., yuplim*.96,
                f'n={int(count)}',
                ha='center', va='bottom', fontsize=9)
    
    if png_path:
        fig.savefig(MWA_FIGDIR+png_path, dpi=400, bbox_inches='tight')
        print(f"Saved figure to {png_path}")

    if cf is not None:
        ax.set_title(f"coarsen_factor: {cf}")
    return fig, ax


def plot_mwa_results(ind_var_name, dep_var_name, coarsen_factors=[None]):
    
    for coarsen_factor in coarsen_factors:
        spath = f'{dep_var_name}_vs_{ind_var_name}'
        if coarsen_factor is not None:
            spath += f'_coarsened{coarsen_factor}'
        csvpath = MWA_DIR + f'{spath}_stats.csv'
        figure_name = spath + '.png'
        plot_binned_results_from_csv(csv_path=csvpath, png_path=figure_name, cf=coarsen_factor,
                                     ylabel=f'mean difference in {dep_var_name}',
                                     xlabel=f'{ind_var_name} bin (%)')
        


#%%

if __name__ == "__main__":
    # Example: plot all coarsened results
    plot_mwa_results(ind_var_name='d_cc', dep_var_name='tmax_trend', coarsen_factors=[None, 2, 4, 8, 16])
    plot_mwa_results(ind_var_name='d_cc', dep_var_name='d_heatwaves', coarsen_factors=[None, 2, 4, 8, 16])
    plot_mwa_results(ind_var_name='d_cc', dep_var_name='d_temp', coarsen_factors=[None, 2, 4, 8, 16])
    plot_mwa_results(ind_var_name='d_cc', dep_var_name='heatwave_trends', coarsen_factors=[None, 2, 4, 8, 16])
    plot_mwa_results(ind_var_name='d_cc', dep_var_name='heatwave_trends_smooth', coarsen_factors=[None, 2, 4, 8, 16])

#%%

def plot_mwa_results_all(
    ind_var_name,
    dep_var_name,
    png_path=None,
    ylabel=None,
    xlabel='canopy cover loss bin (%)',
):
    """Plot all coarsen-factor CSV results for one variable pair on one chart."""
    spath = f'{dep_var_name}_vs_{ind_var_name}'
    csv_paths = sorted(glob(MWA_DIR + f'{spath}*_stats.csv'))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found for pattern: {MWA_DIR}{spath}*_stats.csv")

    dfs = {}
    for csv_path in csv_paths:
        stem = Path(csv_path).stem
        base_stem = stem.replace('_stats', '')

        if base_stem == spath:
            cf = None
        else:
            cf_str = base_stem.split(f'{spath}_')[-1]
            if not cf_str.startswith('coarsened'):
                continue
            cf = int(cf_str.replace('coarsened', ''))

        dfs[cf] = pd.read_csv(csv_path)

    cfs = sorted(dfs.keys(), key=lambda value: -1 if value is None else value)
    n_cfs = len(cfs)
    if n_cfs == 0:
        raise ValueError(f"No valid CSVs to plot for {spath}")

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, n_cfs))

    # Build a shared x-axis from the union of all bins.
    all_bin_labels = {
        bin_label
        for df in dfs.values()
        for bin_label in df['bin'].values
    }
    bin_labels = sort_bin_labels(all_bin_labels)
    n_bins = len(bin_labels)

    bar_width = 0.8 / n_cfs
    x = np.arange(n_bins)

    for i, cf in enumerate(cfs):
        df = dfs[cf]
        offset = (i - n_cfs / 2 + 0.5) * bar_width
        label = 'original' if cf is None else f'{cf}'
        aligned_mean_diff = df.set_index('bin').reindex(bin_labels)['mean_diff'].values

        # Calculate error bar heights.
        ci_lower = df.set_index('bin').reindex(bin_labels)['ci_lower'].values
        ci_upper = df.set_index('bin').reindex(bin_labels)['ci_upper'].values
        yerr_lower = aligned_mean_diff - ci_lower
        yerr_upper = ci_upper - aligned_mean_diff

        ax.bar(
            x + offset,
            aligned_mean_diff,
            width=bar_width,
            label=label,
            color=colors[i],
            alpha=0.85,
            yerr=[yerr_lower, yerr_upper],
            capsize=4,
        )

    if ylabel is None:
        ylabel = f'mean difference in {dep_var_name}'

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(title='coarsen factor', bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.set_title(f'{dep_var_name} vs {ind_var_name} by coarsen factor')
    fig.tight_layout()

    if png_path is None:
        png_path = f'{spath}_all_coarsen_factors.png'
    fig.savefig(MWA_FIGDIR + png_path, dpi=400, bbox_inches='tight')
    print(f"Saved figure to {MWA_FIGDIR + png_path}")

    return fig, ax

plot_mwa_results_all(ind_var_name='d_cc', dep_var_name='tmax_trend', png_path='tmax_trend_vs_d_cc_all_coarsen_factors.png')
plot_mwa_results_all(ind_var_name='d_cc', dep_var_name='d_heatwaves', png_path='d_heatwaves_vs_d_cc_all_coarsen_factors.png')
plot_mwa_results_all(ind_var_name='d_cc', dep_var_name='d_temp', png_path='d_temp_vs_d_cc_all_coarsen_factors.png')
plot_mwa_results_all(ind_var_name='d_cc', dep_var_name='heatwave_trends', png_path='heatwave_trends_vs_d_cc_all_coarsen_factors.png')
plot_mwa_results_all(ind_var_name='d_cc', dep_var_name='heatwave_trends_smooth', png_path='heatwave_trends_smooth_vs_d_cc_all_coarsen_factors.png')

#%%


# #%%

# slopes = pd.DataFrame(index=coarsen_factors, columns=['slope'])
# slopes.index = [0, 2,3,4, 5, 10, 15, 20, 25]


# def calculate_slope_from_csv(csv_path, slopes, coarsen_bin=None):
#     res = pd.read_csv(csv_path)
#     # calculate the slope of the mean_diff vs bin midpoints
#     bin_midpoints = [int(interval.split('-')[0]) + 5 for interval in res['bin']]
#     slope = np.polyfit(bin_midpoints, res['mean_diff'], 1)[0]
#     if coarsen_bin is not None:
#         slopes.loc[coarsen_bin, 'slope'] = slope
#     else:        
#         slopes.loc[0, 'slope'] = slope
#     return slopes

# slopes = calculate_slope_from_csv('./data/heatwave_trend_vs_cc_loss_bins_coarsened_stats.csv', slopes)
# for cf in coarsen_factors[1:]:
#     csv_file = f'./data/heatwave_trend_vs_cc_loss_bins_coarsened{cf}_stats.csv'
#     if Path(csv_file).exists():
#         slopes = calculate_slope_from_csv(csv_file, slopes, coarsen_bin=cf)

# #%%

# fig, ax = plt.subplots(figsize=(8, 5))
# ax.plot(slopes.index, slopes / slopes.loc[0], marker='o')
# ax.set_ylabel('relative difference in slope')
# ax.set_xlabel('coarsen factor')
# ax.grid()
# fig.savefig(FIGDIR + 'percentiles/heatwave_trend_vs_cc_loss_slope_vs_coarsen_factor.png', dpi=400, bbox_inches='tight')

# #%%
# print('In a 17 year period (2003-2020), 0.01 degree grids cells that experience 50-60% more canopy loss than their neighbours have an average increase of')
# print(f"{dfs[None].loc[dfs[None]['bin'] == '50-60', 'mean_diff'].values[0] * 17:.2f} heatwave days per year.")

# print('If the 50% forest loss occurs over a 0.1 degree grid cell (coarsen factor 10), the average increase in heatwave days is')
# print(f"{dfs[10].loc[dfs[10]['bin'] == '50-60', 'mean_diff'].values[0] * 17:.2f} heatwave days per year.")

# print('If the 50% forest loss occurs over a 0.25 degree grid cell (coarsen factor 25), the average increase in heatwave days is')
# print(f"{dfs[25].loc[dfs[25]['bin'] == '50-60', 'mean_diff'].values[0] * 17:.2f} heatwave days per year.")


# #%%

# plot_binned_results_from_csv('./data/tmax_trend_vs_cc_loss_bins_stats.csv', png_path='tmax_trend_vs_cc_loss.png')
# plot_binned_results_from_csv('./data/tmax_trend_jja_vs_cc_loss_bins_stats.csv', png_path='tmax_trend_jja_vs_cc_loss.png')
# plot_binned_results_from_csv('./data/tmax_trend_jja_vs_cc_loss_bins_coarsened10_stats.csv', png_path='tmax_trend_jja_vs_cc_loss_coarsened10.png')