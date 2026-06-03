#%%
"""Plot binned moving window analysis results from CSV files."""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from constants import FIGDIR


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
        fig.savefig(FIGDIR+'percentiles/'+png_path, dpi=400, bbox_inches='tight')
        print(f"Saved figure to {png_path}")

    if cf is not None:
        ax.set_title(f"coarsen_factor: {cf}")
    return fig, ax


if __name__ == "__main__":
    # Example: plot all coarsened results
    coarsen_factors = [None, 2,3,4,5, 10, 15, 20, 25]
    
    dfs = {}
    for cf in coarsen_factors:
        if cf is None:
            csv_file = 'heatwave_trend_vs_cc_loss_bins_coarsened_stats.csv'
            png_file = 'heatwave_trend_vs_cc_loss_bins_coarsened.png'
        else:
            csv_file = f'heatwave_trend_vs_cc_loss_bins_coarsened{cf}_stats.csv'
            png_file = f'heatwave_trend_vs_cc_loss_bins_coarsened{cf}.png'
        
        csv_path = Path('data') / csv_file
        if csv_path.exists():
            print(f"Plotting {csv_file}...")
            plot_binned_results_from_csv(
                csv_path,
                png_path=png_file,
                cf=cf,
                ylabel='mean difference in heatwave trend [heatwave days/year]',
                xlabel='canopy cover loss bin (%)'
            )
            df = pd.read_csv(csv_path)
            dfs[cf] = df
        else:
            print(f"CSV file not found: {csv_file}")
#%%
fig, ax = plt.subplots(figsize=(14, 6))

colors = plt.cm.tab10(np.linspace(0, 1, len(dfs)))
cf_keys = list(dfs.keys())
n_cfs = len(cf_keys)

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

for i, (cf, df) in enumerate(dfs.items()):
    offset = (i - n_cfs / 2 + 0.5) * bar_width
    label = f'{cf}' if cf is not None else 'original'
    aligned_mean_diff = df.set_index('bin').reindex(bin_labels)['mean_diff'].values
    ax.bar(x + offset, aligned_mean_diff, width=bar_width,
           label=label, color=colors[i], alpha=0.85)

ax.set_xlabel('canopy cover loss bin (%)', fontsize=12)
ax.set_ylabel('mean difference in heatwave trend [heatwave days/year]', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(bin_labels, rotation=0)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend(title='coarsen factor', bbox_to_anchor=(1.01, 1), loc='upper left')
ax.set_title('Heatwave trend change vs canopy cover loss by coarsen factor')
fig.tight_layout()
fig.savefig(FIGDIR + 'percentiles/heatwave_trend_vs_cc_loss_all_coarsen_factors.png', dpi=400, bbox_inches='tight')


#%%

slopes = pd.DataFrame(index=coarsen_factors, columns=['slope'])
slopes.index = [0, 2,3,4, 5, 10, 15, 20, 25]


def calculate_slope_from_csv(csv_path, slopes, coarsen_bin=None):
    res = pd.read_csv(csv_path)
    # calculate the slope of the mean_diff vs bin midpoints
    bin_midpoints = [int(interval.split('-')[0]) + 5 for interval in res['bin']]
    slope = np.polyfit(bin_midpoints, res['mean_diff'], 1)[0]
    if coarsen_bin is not None:
        slopes.loc[coarsen_bin, 'slope'] = slope
    else:        
        slopes.loc[0, 'slope'] = slope
    return slopes

slopes = calculate_slope_from_csv('./data/heatwave_trend_vs_cc_loss_bins_coarsened_stats.csv', slopes)
for cf in coarsen_factors[1:]:
    csv_file = f'./data/heatwave_trend_vs_cc_loss_bins_coarsened{cf}_stats.csv'
    if Path(csv_file).exists():
        slopes = calculate_slope_from_csv(csv_file, slopes, coarsen_bin=cf)

#%%

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(slopes.index, slopes / slopes.loc[0], marker='o')
ax.set_ylabel('relative difference in slope')
ax.set_xlabel('coarsen factor')
ax.grid()
fig.savefig(FIGDIR + 'percentiles/heatwave_trend_vs_cc_loss_slope_vs_coarsen_factor.png', dpi=400, bbox_inches='tight')

#%%
print('In a 17 year period (2003-2020), 0.01 degree grids cells that experience 50-60% more canopy loss than their neighbours have an average increase of')
print(f"{dfs[None].loc[dfs[None]['bin'] == '50-60', 'mean_diff'].values[0] * 17:.2f} heatwave days per year.")

print('If the 50% forest loss occurs over a 0.1 degree grid cell (coarsen factor 10), the average increase in heatwave days is')
print(f"{dfs[10].loc[dfs[10]['bin'] == '50-60', 'mean_diff'].values[0] * 17:.2f} heatwave days per year.")

print('If the 50% forest loss occurs over a 0.25 degree grid cell (coarsen factor 25), the average increase in heatwave days is')
print(f"{dfs[25].loc[dfs[25]['bin'] == '50-60', 'mean_diff'].values[0] * 17:.2f} heatwave days per year.")


#%%

plot_binned_results_from_csv('./data/tmax_trend_vs_cc_loss_bins_stats.csv', png_path='tmax_trend_vs_cc_loss.png')
plot_binned_results_from_csv('./data/tmax_trend_jja_vs_cc_loss_bins_stats.csv', png_path='tmax_trend_jja_vs_cc_loss.png')
plot_binned_results_from_csv('./data/tmax_trend_jja_vs_cc_loss_bins_coarsened10_stats.csv', png_path='tmax_trend_jja_vs_cc_loss_coarsened10.png')