#%%
from constants import  MGWR_RESULTS_DIR
import pickle as pkl
from tqdm import tqdm
import pandas as pd
import numpy as np
import geopandas as gpd

def retrieve_pickled_results():
    results = {}
    for doy in range(1, 366):
        try:
            with open(f'{MGWR_RESULTS_DIR}/layer1/mgwr_model_doy_{doy}.pkl', 'rb') as f:
                data = pkl.load(f)
                results[doy] = data
        except FileNotFoundError:
            print(f'MGWR model for DOY {doy} not found in {MGWR_RESULTS_DIR}mgwr_model_doy_{doy}.pkl')
    return results

results = retrieve_pickled_results()

#%%

def get_metrics(results):
    metrics = ['R2', 'adj_R2', 'resid_ss', 'aicc']
    df = pd.DataFrame(index=range(1, 366), columns=metrics)
    for doy, data in results.items():
        res = data['results']
        for metric in metrics:
            df.loc[doy, metric] = getattr(res, metric)
        residuals = res.resid_response.flatten()
        df.loc[doy, 'RMSE'] = np.sqrt(np.mean(residuals**2))
        df.loc[doy, 'MAE'] = np.mean(np.abs(residuals))
    return df

df = get_metrics(results)

#%%
df = pd.DataFrame(index=range(1, 366), columns=['intercept', 'LST', 'elevation'])
for doy, data in results.items():
    selector = data['selector']
    bws =  pd.Series(selector.bw[0], index=['intercept', 'LST', 'elevation'])
    # make multiindex with doy and variable
    df.loc[doy] = bws
#%%
with open('./data/azimuthal_projection.txt') as f:
    azimuthal_projection = f.read().strip()

df = pd.DataFrame(index=range(1, 366), columns=['LST', 'elevation'])
for doy, data in results.items():
    res = data['results']
    # coefficient std
    std =  pd.DataFrame(res.params[:, 1:]).std().values
    mean = pd.DataFrame(res.params[:, 1:]).mean().values
    cv  = std / mean
    df.loc[doy] = cv