#%%
from svcm_sp_data import load_svcm_data, get_valid_meteostat_stations
from constants import MIN_VALID_YEARS, MIN_VALID_DAYS_PER_YEAR, DOY_POOL_WINDOW, MGWR_RESULTS_DIR
import numpy as np
from scipy.sparse import hstack
from mgwr.gwr import MGWR
from mgwr.sel_bw import Sel_BW
from mgwr.gwr import GWR, MGWR
import pandas as pd
import pickle as pkl
from joblib import Parallel, delayed

def doy_gaussian_mean(
    df,
    target_doy,
    window=10,
    sigma=None,
    n_days_per_year=365,
):
    if sigma is None:
        sigma = window

    offsets = np.arange(-window, window + 1)

    # Circular DOY corresponding to each offset
    days = ((target_doy - 1 + offsets) % n_days_per_year) + 1

    weights = np.exp(-0.5 * (offsets / sigma) ** 2)

    weight_lookup = pd.Series(weights, index=days)
    offset_lookup = pd.Series(offsets, index=days)

    # Select observations within the Gaussian window
    doy_df = df[df['doy'].isin(days)].copy()

    # Weight associated with each observed DOY
    doy_df['_weight'] = doy_df['doy'].map(weight_lookup)

    # Offset relative to target DOY
    doy_df['_offset'] = doy_df['doy'].map(offset_lookup)

    # Assign each observation to the year corresponding to the
    # target DOY it contributes to.
    #
    # Negative offsets wrap backwards across New Year:
    # e.g. target DOY 2, observed DOY 365 -> target year = year + 1
    #
    # Positive offsets wrap forwards across New Year:
    # e.g. target DOY 364, observed DOY 1 -> target year = year - 1
    doy_df['_target_year'] = doy_df['year']

    doy_df.loc[
        (doy_df['_offset'] < 0) &
        (doy_df['doy'] > target_doy),
        '_target_year'
    ] = doy_df['year'] + 1

    doy_df.loc[
        (doy_df['_offset'] > 0) &
        (doy_df['doy'] < target_doy),
        '_target_year'
    ] = doy_df['year'] - 1

    # Weighted values
    doy_df['_t2m_weighted'] = doy_df['t2m'] * doy_df['_weight']
    doy_df['_LST_weighted'] = doy_df['LST'] * doy_df['_weight']

    # Weighted sums by station and target year
    weighted_sum = (
        doy_df
        .groupby(['station_id', '_target_year'])
        [['_t2m_weighted', '_LST_weighted']]
        .sum()
    )

    # Sum only weights corresponding to valid observations
    weight_sum = (
        doy_df
        .assign(
            _t2m_weight=doy_df['_weight'].where(doy_df['t2m'].notna()),
            _LST_weight=doy_df['_weight'].where(doy_df['LST'].notna())
        )
        .groupby(['station_id', '_target_year'])
        [['_t2m_weight', '_LST_weight']]
        .sum()
    )

    # Gaussian-weighted annual values
    annual = pd.DataFrame(index=weighted_sum.index)

    annual['t2m'] = (
        weighted_sum['_t2m_weighted']
        / weight_sum['_t2m_weight']
    )

    annual['LST'] = (
        weighted_sum['_LST_weighted']
        / weight_sum['_LST_weight']
    )

    # Average across years to produce the climatological value
    result = (
        annual
        .groupby('station_id')[['t2m', 'LST']]
        .mean()
        .reset_index()
    )

    # Add static station information
    station_info = (
        df[
            ['station_id', 'elevation', 'easting', 'northing']
        ]
        .drop_duplicates('station_id')
    )

    result = result.merge(
        station_info,
        on='station_id',
        how='left'
    )

    return result

#%%
df = load_svcm_data()
mdf = get_valid_meteostat_stations(min_valid_years=MIN_VALID_YEARS, min_valid_days_per_year=MIN_VALID_DAYS_PER_YEAR)




#%%

def fit_model(target_doy):
    doy_set = doy_gaussian_mean(
        df,
        target_doy,
        window=DOY_POOL_WINDOW,
        n_days_per_year=365
    )

        
    coords = doy_set[['easting', 'northing']].values
    y = doy_set['t2m'].astype(float).values.reshape(-1, 1)
    X = doy_set[['LST', 'elevation']].values

    selector = Sel_BW(coords, y, X, multi=True, constant=True,
                      n_jobs=1)

    print(f'Selecting bandwidth for DOY {target_doy}...')
    selector.search(bw_min=2)

    print(f'Fitting MGWR model for DOY {target_doy} with optimal bandwidth...')
    model = MGWR(coords, y, X, selector, n_jobs=1)
    results = model.fit()

    # pickle the selector, model and results
    with open(f'{MGWR_RESULTS_DIR}mgwr_model_doy_{target_doy}.pkl', 'wb') as f:
        pkl.dump({
            'selector': selector,
            'model': model,
            'results': results
        }, f)

    print(f'MGWR model for DOY {target_doy} fitted and saved to {MGWR_RESULTS_DIR}mgwr_model_doy_{target_doy}.pkl')
    return target_doy

if __name__ == '__main__':
    # parallelize the fitting over all target DOYs
    target_doys = range(1, 366)
    Parallel(n_jobs=8)(delayed(fit_model)(doy) for doy in target_doys)

