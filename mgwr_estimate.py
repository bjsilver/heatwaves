#%%
import os
from constants import DOY_POOL_WINDOW, MGWR_RESULTS_DIR, METEOSTAT_DATA_PATH, DATE_SLICE
import numpy as np
from mgwr.gwr import MGWR
from mgwr.sel_bw import Sel_BW
import pandas as pd
import pickle as pkl
from joblib import Parallel, delayed
from mgwr_data import get_climatologically_valid_data, doy_gaussian_mean


def fit_layer1_model(df, target_doy):
    doy_set = doy_gaussian_mean(
        df,
        target_doy,
        window=DOY_POOL_WINDOW,
        n_days_per_year=365
    )

    doy_set = doy_set.dropna(subset=['tmax', 'LST', 'elevation', 'easting', 'northing'])

        
    coords = doy_set[['easting', 'northing']].values
    y = doy_set['tmax'].astype(float).values.reshape(-1, 1)
    X = doy_set[['LST', 'elevation']].values
    
    selector = Sel_BW(coords, y, X, multi=True, constant=True,
                      n_jobs=1)

    selector.search(bw_min=2)

    model = MGWR(coords, y, X, selector, n_jobs=1)
    results = model.fit()

    # pickle the selector, model and results
    with open(f'{MGWR_RESULTS_DIR}layer1/mgwr_model_doy_{target_doy}.pkl', 'wb') as f:
        pkl.dump({
            'selector': selector,
            'model': model,
            'results': results,
            'df': doy_set
        }, f)

    print(f'MGWR model for DOY {target_doy} fitted and saved to {MGWR_RESULTS_DIR}layer1/mgwr_model_doy_{target_doy}.pkl')
    return target_doy

#%%

def fit_all_layer1_models():
    df = get_climatologically_valid_data(recompute=False)
    target_doys = range(1, 366)
    Parallel(n_jobs=16)(delayed(fit_layer1_model)(df, doy) for doy in target_doys)



#%%
if __name__ == '__main__':
    fit_all_layer1_models()


#%%
df = get_climatologically_valid_data(recompute=False)
df = df.where(df.elevation < 300)
df = df.dropna(subset=['tmax', 'LST'])

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(df['LST'], df['tmax'], alpha=0.01,
           marker='.', color='black', label='Data points')

r2 = np.corrcoef(df['LST'], df['tmax'])[0, 1] ** 2
# calc significance of correlation
from scipy.stats import pearsonr
r, p = pearsonr(df['LST'], df['tmax'])
ax.set_title(f'LST vs tmax (R2={r2:.3f}, p={p:.3e})')
ax.set_xlabel('LST (°C)')
ax.set_ylabel('tmax (°C)')
# at any given station is there even a decent correlation
# between LST and tmax? 
# if not, then MGWR is not going to work well