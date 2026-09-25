#%%
import os
from constants import DOY_POOL_WINDOW, MGWR_RESULTS_DIR, METEOSTAT_DATA_PATH, DATE_SLICE
import numpy as np
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
import pandas as pd
import pickle as pkl
from joblib import Parallel, delayed
from mgwr_data import get_climatologically_valid_data, doy_gaussian_mean
from tqdm import tqdm



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
    X = doy_set[['LST']].values
    
    selector = Sel_BW(coords, y, X, multi=False, constant=True,
                      n_jobs=1)

    # print(f'Selecting bandwidth for DOY {target_doy}...')
    bw = selector.search(bw_min=2)


    # print(f'Fitting GWR model for DOY {target_doy} with optimal bandwidth...')
    model = GWR(coords, y, X, bw, n_jobs=1)
    results = model.fit()

    # pickle the selector, model and results
    with open(f'{MGWR_RESULTS_DIR}layer1/mgwr_model_doy_{target_doy}.pkl', 'wb') as f:
        pkl.dump({
            'model': model,
            'bw': bw,
            'results': results,
            'df': doy_set
        }, f)

    # print(f'GWR model for DOY {target_doy} fitted and saved to {MGWR_RESULTS_DIR}layer1/mgwr_model_doy_{target_doy}.pkl')
    return target_doy

#%%

def fit_all_layer1_models():
    df = get_climatologically_valid_data(recompute=False)
    target_doys = range(1, 366)

    print('Fitting GWR models for all DOYs...')
    for doy in tqdm(target_doys, desc='Fitting GWR models for all DOYs'):
        fit_layer1_model(df, doy)

#%%
if __name__ == '__main__':
    fit_all_layer1_models()