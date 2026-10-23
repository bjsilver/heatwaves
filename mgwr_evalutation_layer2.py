#%%
from constants import  MGWR_RESULTS_DIR, TMAX_RECHUNKED_FILE, STORAGE_DIR
import pickle as pkl
from tqdm import tqdm
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr
from ufuncs import local_cluster
from obs_data import get_harmonised_data
import os


def retrieve_pickled_results():

    with open(f'{MGWR_RESULTS_DIR}/layer2/grf_models.pkl', 'rb') as f:
        results = pkl.load(f)
    return results

results = retrieve_pickled_results()
predictions = results['predictions']
# %%

predictions['tmax_pred'] = predictions['tmax_clim'] + predictions['pred_resid']

comparisons = (
    ['tmax', 'pred_zz'],
    ['tmax', 'tmax_clim'],
    ['tmax', 'tmax_pred'],
)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for comp, ax in zip(comparisons, axes):
    x = predictions[comp[0]]
    y = predictions[comp[1]]

    ax.scatter(x, y, alpha=.5, color='black', marker='.',
               s=.01)

    ax.set_xlim(15, 45)
    ax.set_ylim(15, 45)
    # ensure equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # 1:1 line
    ax.plot([15, 45], [15, 45], color='red', linestyle='--')

    # best fit line
    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m*x + b, color='blue', linestyle='-')

    # print r2 on each
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    ax.text(0.05, 0.95, f'$R^2$ = {r2:.3f}', transform=ax.transAxes, fontsize=12,
            verticalalignment='top')
    # print rmse on each
    rmse = np.sqrt(np.mean((x - y) ** 2))
    ax.text(0.05, 0.90, f'RMSE = {rmse:.3f}', transform=ax.transAxes, fontsize=12,
            verticalalignment='top')