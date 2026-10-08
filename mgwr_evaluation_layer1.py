#%%
from constants import  MGWR_RESULTS_DIR, TMAX_RECHUNKED_FILE, STORAGE_DIR, DATE_SLICE
import pickle as pkl
from tqdm import tqdm
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr
from ufuncs import get_amazon_basin_mask, local_cluster
from obs_data import get_harmonised_data
import os
from ufuncs import get_amazon_basin_mask

def retrieve_pickled_results(test_run=None):
    results = {}
    for doy in range(1, 366):
        results_dir = f'{MGWR_RESULTS_DIR}/layer1/mgwr_model_doy_{doy}.pkl'
        if test_run is not None:
            results_dir = f'{MGWR_RESULTS_DIR}/layer1/tests/mgwr_model_doy_{doy}_{test_run}.pkl'
        try:
            with open(results_dir, 'rb') as f:
                data = pkl.load(f)
                results[doy] = data
        except FileNotFoundError:
            print(f'MGWR model for DOY {doy} not found in {MGWR_RESULTS_DIR}mgwr_model_doy_{doy}.pkl')
    return results


#resample to annual
def get_annual_mean_df(df):
    MIN_LEN_THRESH = round(365*.8)
    mean_dfs = []
    for station_id in df.index.get_level_values('station_id').unique():
        # use the second multiindex level to select
        station_df = df.xs(station_id, level='station_id')
        valid = station_df.resample('YS').count().mean(axis=1) > MIN_LEN_THRESH
        mean = station_df.resample('YS').mean()
        mean = mean[valid]
        mean_dfs.append(mean)

    mean_df = pd.concat(mean_dfs, axis=0)
    return mean_df

def retrieve_model_results(eval_type='unseen_data_predictions',
                           test_run=None):

    results = retrieve_pickled_results(test_run=test_run)
    dfs = []
    for doy in range(1, 366):

        r = results.get(doy)
        df = r[eval_type]

        if eval_type == 'unseen_data_predictions':
            df = df.reset_index().set_index(['datetime', 'station_id'])
        elif eval_type == 'k-fold_predictions':
            df = df.rename({'obs':'tmax_obs', 'pred':'tmax_pred'}, axis=1)
            df['doy'] = doy
            # prepend doy to the index to avoid duplicate indices when concatenating
            df = df.reset_index().set_index(['doy', 'station_id'])
        dfs.append(df)

    df = pd.concat(dfs, axis=0)
    df = df.dropna().sort_index()
    return df


def calc_evaluation_metrics(x, y):
    rmse = np.sqrt(np.mean((x - y) ** 2))
    mae = np.mean(np.abs(x - y))
    r2 = 1 - (np.sum((x - y) ** 2) / np.sum((x - np.mean(x)) ** 2))
    nmb = np.sum(y - x) / np.sum(x)
    return rmse, mae, r2, nmb

def format_evaluation_plot_ax(ax, x, y):
    
    
    ax.set_xlim(max(min(x), min(y)), min(max(x), max(y)))
    ax.set_ylim(max(min(x), min(y)), min(max(x), max(y)))
    # plot a 1:1 line
    ax.plot([min(x), max(x)], [min(x), max(x)], color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Observed Tmax (°C)')
    ax.set_ylabel('Predicted Tmax (°C)')
    
    # ensure axis is square
    ax.set_aspect('equal', adjustable='box')
    ax.grid()
    # make axis background grey
    ax.set_facecolor('lightgrey')


def evaluation_plot(x, y, title=''):


    # plot a density scatter plot of tmax_observed vs tmax_predicted
    fig, ax = plt.subplots(figsize=(8, 8))

    if len(df) < 20000:
        ax.plot(x, y, 'o', color='black', markersize=4, alpha=1)
    else:
        # plot a 2D histogram of tmax_observed vs tmax_predicted
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
        heatmap = np.where(heatmap == 0, np.nan, heatmap)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='magma',
                vmin=0, vmax=np.nanpercentile(heatmap, 99))

    
    rmse, mae, r2, nmb = calc_evaluation_metrics(x, y)
    format_evaluation_plot_ax(ax, x, y)

    # plot a text box with the RMSE, MAE, R2, and NMB values
    textstr = '\n'.join((
        f'RMSE: {rmse:.2f} °C',
        f'MAE: {mae:.2f} °C',
        f'R²: {r2:.2f}',
        f'NMB: {nmb:.2%}'
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=1)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
    heatmap = np.where(heatmap == 0, np.nan, heatmap)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]


    ax.set_title(title)

def interpolate_zz_to_stations(meta, recompute=False):

    if not recompute and os.path.exists(STORAGE_DIR+'/tmax/obs_interped/tmax_interpolated.zarr'):
        print('Loading precomputed interpolated ZZ data from zarr')
        interpolated_zz = xr.open_zarr(STORAGE_DIR+'/tmax/obs_interped/tmax_interpolated.zarr')['Ta_TMAX']
        return interpolated_zz

    zz = xr.open_zarr(TMAX_RECHUNKED_FILE)['Ta_TMAX']
    lat, lon = meta['lat'].values, meta['lon'].values

    
    with local_cluster(n_workers=6, threads_per_worker=4,
                    memory_limit='100GB', ) as cluster:
        interpolated_zz = zz.interp(lat=xr.DataArray(lat, dims='station_id'), 
                                        lon=xr.DataArray(lon, dims='station_id'))
        interpolated_zz = interpolated_zz.assign_coords(station_id=meta['station_id'].values)
        interpolated_zz = interpolated_zz.chunk({'time':-1})
        print(interpolated_zz.coords)
        interpolated_zz.to_zarr(STORAGE_DIR+'/tmax/obs_interped/tmax_interpolated.zarr', mode='w',
        zarr_format=2)

def merge_zz_into_df(df, zz_interped):
    zz = zz_interped.to_pandas()
    zz.index = pd.DatetimeIndex([pd.Timestamp(i.strftime()) for i in  zz.index])
    zz = zz.stack() / 10
    zz.name = 'tmax_pred_zz'
    zz.index.names = ['datetime', 'station_id']
    # merge into df as a new column, but drop zz where multiindex doesn't overlap
    df = df.merge(zz, how='left', left_index=True, right_index=True)
    return df

def evaluation_plot_with_zz(df):
    # plot a scatter plot of tmax_observed vs tmax_predicted and tmax
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(df['tmax_obs'], df['tmax_pred_zz'], alpha=0.5, s=10, color='blue', label='ZZ Predicted')
    ax.scatter(df['tmax_obs'], df['tmax_pred'], alpha=0.5, s=10, color='orange', label='MGWR Predicted')

    format_evaluation_plot_ax(ax, df['tmax_obs'], df['tmax_pred_zz'])

    # generate a legend with the evaluation metrics for each model
    rmse_zz, mae_zz, r2_zz, nmb_zz = calc_evaluation_metrics(df['tmax_obs'], df['tmax_pred_zz'])
    rmse_mgwr, mae_mgwr, r2_mgwr, nmb_mgwr = calc_evaluation_metrics(df['tmax_obs'], df['tmax_pred'])

    # plot a text box with the RMSE, MAE, R2, and NMB values for each model
    textstr_zz = '\n'.join((
        f'ZZ Predicted:',
        f'RMSE: {rmse_zz:.2f} °C',
        f'MAE: {mae_zz:.2f} °C',
        f'R²: {r2_zz:.2f}',
        f'NMB: {nmb_zz:.2%}',
        ))

    textstr_mgwr = '\n'.join((
        f'MGWR Predicted:',
        f'RMSE: {rmse_mgwr:.2f} °C',
        f'MAE: {mae_mgwr:.2f} °C',
        f'R²: {r2_mgwr:.2f}',
        f'NMB: {nmb_mgwr:.2%}',
        ))

    ax.text(0.05, 0.95, textstr_zz, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=1,
                                            edgecolor='blue'))
    ax.text(0.95, 0.05, textstr_mgwr, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=1,
                                            edgecolor='orange'))

#%%
if __name__ == '__main__':
    results = retrieve_pickled_results()
    df = retrieve_model_results(eval_type='k-fold_predictions')
    all_obs, meta, _ = get_harmonised_data()
    
    zz_interped = interpolate_zz_to_stations(meta, recompute=False)
    df = merge_zz_into_df(df, zz_interped)
    mean_df = get_annual_mean_df(df)
    evaluation_plot(mean_df, title='layer 1 model evaluation - annual mean')
    evaluation_plot(df, title='daily')
    evaluation_plot_with_zz(df)
    evaluation_plot_with_zz(mean_df)

    df = retrieve_model_results(eval_type='k-fold_predictions')
    diff = df['tmax_pred'] - df['tmax_obs']

    fig, ax = plt.subplots(figsize=(8, 8))
    for doy in range(1, 366):
        df_doy = df.xs(doy, level='doy')
        rmse, mae, r2, nmb = calc_evaluation_metrics(df_doy['tmax_obs'], df_doy['tmax_pred'])
        ax.plot(doy, rmse, 'o', color='black', markersize=4, alpha=1)
        ax.plot(doy, mae, 'o', color='blue', markersize=4, alpha=1)
        ax.plot(doy, r2, 'o', color='orange', markersize=4, alpha=1)


def get_zz_obs_df():
    all_obs, meta, _ = get_harmonised_data()
    # drop leap days
    all_obs = all_obs.loc[~((all_obs.index.month == 2) & (all_obs.index.day == 29))]
    obs = all_obs.stack()
    obs.index.names = ['datetime', 'station_id']

    zz_interped = interpolate_zz_to_stations(meta, recompute=False) / 10
    # convert cftime to datetime
    zz_interped.coords['time'] = pd.DatetimeIndex([pd.Timestamp(t.strftime()) for t in zz_interped.time.values])
    zz_interped = zz_interped.rename({'time':'datetime'})
    zz = zz_interped.to_pandas().stack()

    df = pd.merge(obs.rename('tmax_obs'), zz.rename('tmax_pred_zz'), how='left', left_index=True, right_index=True)
    df = df.dropna()
    return df, meta


def zz_evaluation():
    df, meta = get_zz_obs_df()
    # evaluation
    AMAZON_BASIN = get_amazon_basin_mask(buffer_km=0)['vector']
    basin_stations = meta[meta.to_crs('EPSG:4326').geometry.within(AMAZON_BASIN.geometry.iloc[0])]
    in_basin_stations = basin_stations['station_id'].values
    df_basin = df[df.index.get_level_values('station_id').isin(in_basin_stations)]
    evaluation_plot(df_basin['tmax_obs'], df_basin['tmax_pred_zz'], title='')

    # annual
    df_basin_annual = get_annual_mean_df(df_basin)
    evaluation_plot(df_basin_annual['tmax_obs'], df_basin_annual['tmax_pred_zz'], title='annual mean')





def compare_layer1_with_zz(df, all_obs, zz_interped):
    start_year = DATE_SLICE.start[:4]
    end_year = DATE_SLICE.stop[:4]
    tindex = xr.date_range(start=f'{start_year}-01-01', end=f'{end_year}-12-31', 
                        freq='D', calendar='noleap')
    # convert from cftime to pandas datetime
    tindex = pd.DatetimeIndex([pd.Timestamp(t.strftime()) for t in tindex])

    years = tindex.year.unique()
    df_dt = pd.concat([df] * len(years), keys=years, names=['year'])
    df_dt = df_dt.reset_index()
    df_dt['datetime'] = tindex.repeat(len(df.index.get_level_values('station_id').unique())).values
    df_dt = df_dt.drop(columns=['year', 'doy']).set_index(['datetime', 'station_id']).sort_index()

    df_dt = merge_zz_into_df(df_dt, zz_interped)


    obs = all_obs.stack()
    obs.index.names = ['datetime', 'station_id']
    df_dt = df_dt.drop('tmax_obs', axis=1).merge(obs.rename('tmax_obs'), how='left', left_index=True, right_index=True)

    # plot scatter of tmax_obs vs tmax_pred and of tmax_obs vs tmax_pred_zz
    evaluation_plot_with_zz(df_dt)



df = retrieve_model_results(eval_type='daily_k-fold_predictions')
zz_interped = interpolate_zz_to_stations(meta, recompute=False)
all_obs, meta, _ = get_harmonised_data()
evaluation_plot(x=df['obs'], y=df['pred'], title='layer 1 model evaluation - daily')
compare_layer1_with_zz(df, all_obs, zz_interped)
#%% temp test delete later

nmbs = pd.Series(dtype=float)
for doy, r in results.items():
    df = r['k-fold_predictions']
    nmb = np.sum(df['pred'] - df['obs']) / np.sum(df['obs'])
    nmbs.loc[doy] = nmb

nmbs.plot()

#%% zz evaluation
