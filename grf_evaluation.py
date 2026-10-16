#%%
from constants import  GRF_RESULTS_DIR, TMAX_RECHUNKED_FILE, STORAGE_DIR, DATE_SLICE, FIGDIR
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
from glob import glob
import cartopy.crs as ccrs
import contextily as ctx
import cartopy.feature as cf
from obs_data import define_local_azimuthal_projection
# import evaluation metrics
import sklearn

CRS = define_local_azimuthal_projection(recalculate=False)
AMAZON_BASIN = get_amazon_basin_mask(buffer_km=0)['vector']
figpath = FIGDIR + 'grf_evaluation/'
os.makedirs(figpath, exist_ok=True)

FEATURES = ['LST', 'elevation', 'doy_sin', 'doy_cos', 'day']

pretty_model = {'grf': 'Geographical Random Forest', 'zz': 'Zhang+Zhou',
                'xgb': 'Geographical XGBoost'}
LABEL = 'XGB'

def retrieve_pickled_results(sdir, model_type="rf"):

    fpaths = GRF_RESULTS_DIR + f'{model_type}/{sdir}/fold_*_results.pkl'
    fpaths = glob(fpaths)

    dfs = []
    fimps = []
    for fpath in fpaths:
        with open(fpath, 'rb') as f:
            file = pkl.load(f)
            results = file['predictions']
            fimp = file['feature_importances']
            dfs.append(results)
            fimps.append(fimp)

    df = pd.concat(dfs, axis=0)

    fimp = pd.DataFrame(fimps)
    
    return df


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
        # append the station_id as outer multiindex level
        mean['station_id'] = station_id
        mean = mean.reset_index().set_index(['station_id', 'datetime'])
        mean_dfs.append(mean)

    mean_df = pd.concat(mean_dfs, axis=0)
    return mean_df



def calc_evaluation_metrics(x, y):
    rmse = sklearn.metrics.root_mean_squared_error(x, y)
    mae = sklearn.metrics.mean_absolute_error(x, y)
    r2 = sklearn.metrics.r2_score(x, y)
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


def evaluation_plot(x, y, title, sfname):


    # plot a density scatter plot of tmax_observed vs tmax_predicted
    fig, ax = plt.subplots(figsize=(8, 8))

    if len(x) < 20000:
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
    fig.savefig(figpath + sfname+'.png', dpi=300, bbox_inches='tight')

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

def evaluation_plot_with_zz(obs, rf_pred, zz_pred, title, sfname):
    # plot a scatter plot of tmax_observed vs tmax_predicted and tmax
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(obs, zz_pred, alpha=0.5, s=10, color='blue', label='ZZ Predicted')
    ax.scatter(obs, rf_pred, alpha=0.5, s=10, color='orange', label='MGWR Predicted')

    format_evaluation_plot_ax(ax, obs, zz_pred)

    # generate a legend with the evaluation metrics for each model
    rmse_zz, mae_zz, r2_zz, nmb_zz = calc_evaluation_metrics(obs, zz_pred)
    rmse_mgwr, mae_mgwr, r2_mgwr, nmb_mgwr = calc_evaluation_metrics(obs, rf_pred)

    # plot a text box with the RMSE, MAE, R2, and NMB values for each model
    textstr_zz = '\n'.join((
        f'ZZ Predicted:',
        f'RMSE: {rmse_zz:.2f} °C',
        f'MAE: {mae_zz:.2f} °C',
        f'R²: {r2_zz:.2f}',
        f'NMB: {nmb_zz:.2%}',
        ))

    textstr_mgwr = '\n'.join((
        f'{LABEL} Predicted:',
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

    fig.savefig(figpath + sfname+'.png', dpi=300, bbox_inches='tight')
    

def plot_eval_metrics_map(df, meta, metric, suptitle,
                          spath_prefix):

    metric_equations = {
        'rmse': lambda x, y: sklearn.metrics.root_mean_squared_error(x, y),
        'mae': lambda x, y: sklearn.metrics.mean_absolute_error(x, y),
        'r2': lambda x, y: sklearn.metrics.r2_score(x, y),
        'nmb': lambda x, y: (np.sum(y - x) / np.sum(x))
    }

    cmap_by_metric = {'rmse':'RdYlGn_r', 
                      'mae':'RdYlGn_r', 
                      'r2':'RdYlGn', 
                      'nmb':'RdBu_r'}

    met_grf = df.groupby('station_id').apply(lambda x: metric_equations[metric](x['obs'], x['pred']))
    met_zz = df.groupby('station_id').apply(lambda x: metric_equations[metric](x['obs'], x['tmax_pred_zz']))

    gdf = meta.copy()
    gdf = gdf.set_index('station_id').join(met_grf.rename(f'{metric}_grf')).join(met_zz.rename(f'{metric}_zz'))

    fig = plt.figure(figsize=(14, 8))
    ax_grf = fig.add_axes(121, projection=CRS)
    ax_zz = fig.add_axes(122, projection=CRS)

    ulim = max(gdf[f'{metric}_grf'].quantile(.95), gdf[f'{metric}_zz'].quantile(.95))
    llim = min(gdf[f'{metric}_grf'].quantile(.05), gdf[f'{metric}_zz'].quantile(.05))
    if metric == 'nmb': # centre lims around 0 for nmb
        ulim = max(abs(llim), abs(ulim))
        llim = -ulim
    elif metric == 'r2':
        llim = -ulim


    for ax, model in zip([ax_grf, ax_zz], ['grf', 'zz']):
        # add satellite basemap
        ax.set_title(pretty_model[model], fontsize=16)
        ax.set_axis_off()
        ax.add_feature(cf.BORDERS, linewidth=0.5, edgecolor='black')
        ax.add_feature(cf.COASTLINE, linewidth=0.5)
        gdf.plot(ax=ax, column=f'{metric}_{model}', cmap=cmap_by_metric[metric], 
                markersize=50, legend=True, vmin=llim, vmax=ulim,
                edgecolor='white', linewidth=0.5,
                legend_kwds={'label': metric.upper(), 'shrink': 0.7,
                            'orientation': 'horizontal', 'pad': 0.01})
        
        ctx.add_basemap(ax, crs=CRS, source=ctx.providers.Esri.WorldImagery,
                        attribution='')
        AMAZON_BASIN.to_crs(crs=CRS).boundary.plot(ax=ax, color='red', linewidth=1,
                                zorder=10)

    fig.suptitle(suptitle, fontsize=18, y=.83)
    fig.savefig(figpath + f'{spath_prefix}_{metric}_evalmap.png', dpi=300, bbox_inches='tight')

def plot_eval_metrics_maps(df, meta, mean_df):
    daily_metrics = ['rmse', 'mae', 'r2', 'nmb']
    annual_metrics = ['rmse', 'mae', 'nmb']
    for metric in daily_metrics:
        plot_eval_metrics_map(df, meta, metric, suptitle=r'Daily T$\mathrm{_{max}}$ evaluation map',
                            spath_prefix='daily')
    for metric in annual_metrics:
        plot_eval_metrics_map(mean_df, meta, metric, suptitle=r'Annual mean T$\mathrm{_{max}}$ evaluation map',
                            spath_prefix='annual_mean')

def seasonal_evaluation_plot(df):
    df = df.copy()
    # drop all leap days
    df = df[~((df.index.get_level_values('datetime').month == 2) & (df.index.get_level_values('datetime').day == 29))]
    doy_mean = df.groupby(df.index.get_level_values('datetime').dayofyear).mean()

    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 1, 1])
    ax = fig.add_subplot(gs[0, :])
    # plot the observed and predicted tmax as a function of day of year
    ax.plot(doy_mean.index, doy_mean['obs'], label='Observed', color='black', lw=2)
    ax.plot(doy_mean.index, doy_mean['tmax_pred_zz'], label='ZZ predicted', color='blue', lw=2)
    ax.plot(doy_mean.index, doy_mean['pred'], label=f'{LABEL} predicted', color='orange', lw=2)
    ax.set_xlabel('day of year')
    ax.set_ylabel(r'$\mathrm{T_{max}}$ (°C)')
    ax.legend()

    # Calculate each metric across all daily station values in each season.
    dates = df.index.get_level_values('datetime')
    season = pd.Series(dates.month, index=df.index).map(
        {12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM',
        6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON'}
    )
    season_order = ['DJF', 'MAM', 'JJA', 'SON']
    metric_positions = {'RMSE': 0, 'MAE': 1, 'R²': 2, 'NMB': 3}
    seasonal_metrics = {metric: {'GRF': [], 'ZZ': []} for metric in metric_positions}

    for season_name in season_order:
        season_df = df.loc[season == season_name]
        for model, prediction_column in [('GRF', 'pred'), ('ZZ', 'tmax_pred_zz')]:
            valid = season_df[['obs', prediction_column]].dropna()
            metric_values = calc_evaluation_metrics(
                valid['obs'].to_numpy(), valid[prediction_column].to_numpy()
            )
            for metric, position in metric_positions.items():
                seasonal_metrics[metric][model].append(metric_values[position])

    for i, (metric, values) in enumerate(seasonal_metrics.items()):
        metric_ax = fig.add_subplot(gs[1 + i // 2, i % 2])
        x = np.arange(len(season_order))
        width = 0.36
        metric_ax.bar(x - width / 2, values['GRF'], width, label=LABEL, color='orange')
        metric_ax.bar(x + width / 2, values['ZZ'], width, label='ZZ', color='blue')
        metric_ax.set_xticks(x, season_order)
        metric_ax.set_title(f'Seasonal {metric}')
        metric_ax.set_ylabel(metric)
        metric_ax.grid(axis='y', alpha=0.3)
        metric_ax.legend()

    fig.tight_layout()
    fig.savefig(figpath + 'grf_seasonal_evaluation.png', dpi=300, bbox_inches='tight')

def trend_evaluation_plot(df):
    df = df.copy()
    tdf = df.reset_index().set_index('datetime')
    tdf = tdf.groupby('station_id').resample('YS').mean().reset_index()
    tdf = tdf.set_index(['datetime'])
    tdf = tdf.drop('station_id', axis=1)
    tdf = tdf.astype(float)

    # calc annual mean across all stations
    annual_mean = tdf.groupby('datetime').mean()

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 1, 1])
    ax = fig.add_subplot(gs[0, :])
    ax.plot(annual_mean.index, annual_mean['obs'], label='Observed', color='black', lw=1.5, marker='.')
    ax.plot(annual_mean.index, annual_mean['tmax_pred_zz'], label='ZZ predicted', color='blue', lw=1.5, marker='.')
    ax.plot(annual_mean.index, annual_mean['pred'], label=f'{LABEL} predicted', color='orange', lw=1.5, marker='.')
    ax.set_xlabel('Year')
    ax.set_ylabel(r'$\mathrm{T_{max}}$ (°C)')
    ax.set_title('Annual mean T$_{max}$ trend')
    ax.legend()

    years = np.sort(np.unique(annual_mean.index.year))
    metric_positions = {'RMSE': 0, 'MAE': 1, 'R²': 2, 'NMB': 3}
    annual_metrics = {metric: {'GRF': [], 'ZZ': []} for metric in metric_positions}

    for year in years:
        year_df = tdf[tdf.index.year == year]
        for model, prediction_column in [('GRF', 'pred'), ('ZZ', 'tmax_pred_zz')]:
            valid = year_df[['obs', prediction_column]].dropna()
            metric_values = calc_evaluation_metrics(
                valid['obs'].to_numpy(), valid[prediction_column].to_numpy()
            )
            for metric, position in metric_positions.items():
                annual_metrics[metric][model].append(metric_values[position])

    for i, (metric, values) in enumerate(annual_metrics.items()):
        metric_ax = fig.add_subplot(gs[1 + i // 2, i % 2])
        metric_ax.plot(years, values['GRF'], label=LABEL, color='orange', marker='o', lw=1.5)
        metric_ax.plot(years, values['ZZ'], label='ZZ', color='blue', marker='o', lw=1.5)
        metric_ax.set_title(f'Annual {metric}')
        metric_ax.set_ylabel(metric)
        metric_ax.grid(axis='y', alpha=0.3)
        if i in (2, 3):
            metric_ax.set_xlabel('Year')
        metric_ax.tick_params(axis='x', rotation=45)
        metric_ax.legend()

    fig.tight_layout()
    fig.savefig(figpath + 'grf_annual_trend_evaluation.png', dpi=300, bbox_inches='tight')


def retrieve_final_model_results(sdir, model_type="rf"):
    spath = os.path.join(GRF_RESULTS_DIR, f'{model_type}/{sdir}/grf_final_model.pkl')
    with open(spath, 'rb') as f:
        results = pkl.load(f)
    return results

def plot_feature_importance_maps(results):
    mod = results['model']

    feature_importances = pd.DataFrame(columns=FEATURES + ['x', 'y'])
    for i, local_model in enumerate(mod.local_models):
        station_id = mod.station_ids[i]
        fimp = local_model.feature_importances_
        x, y = mod.station_coords[i]
        feature_importances.loc[station_id] = list(fimp) + [x, y]

    gdf = gpd.GeoDataFrame(feature_importances, geometry=gpd.points_from_xy(feature_importances['x'], feature_importances['y']), crs=CRS)
    # sum doy_sin and doy_cos to get a single feature importance for day of year
    gdf['doy'] = gdf['doy_sin'] + gdf['doy_cos']
    gdf = gdf.drop(columns=['doy_sin', 'doy_cos', 'x', 'y'])



    pretty_features = {'LST': 'Land Surface Temperature', 'elevation': 'Elevation', 
                    'doy': 'Day of Year', 'day': 'Day'}

    fig = plt.figure(figsize=(14,  12))
    for i, feature in enumerate(gdf.columns.drop('geometry')):
        ax = fig.add_subplot(2, 2, i+1, projection=CRS)
        gdf.plot(ax=ax, column=feature, cmap='viridis', markersize=50, legend=True,
                legend_kwds={'label': f'{feature} feature importance', 'shrink': 0.7,
                            'orientation': 'horizontal', 'pad': 0.01},
                edgecolor='white', linewidth=0.5)
        AMAZON_BASIN.to_crs(crs=CRS).boundary.plot(ax=ax, color='red', linewidth=1,
                                zorder=10)
        ax.set_title(f'{pretty_features.get(feature, feature)} feature importance')
        ax.set_axis_off()
        ctx.add_basemap(ax, crs=CRS, source=ctx.providers.Esri.WorldImagery,
                        attribution='')
    fig.tight_layout()
    fig.savefig(figpath + 'grf_feature_importance_maps.png', dpi=300, bbox_inches='tight')


def plot_feature_response(sdir, model_type="rf"):
    global figpath
    figpath = os.path.join(FIGDIR, 'grf_evaluation', model_type, sdir) + '/'
    os.makedirs(figpath, exist_ok=True)

    final_model_results = retrieve_final_model_results(sdir=sdir, model_type=model_type)


    feature_responses = final_model_results['feature_responses']

    fig = plt.figure(figsize=(16, 12))
    axes = fig.subplots(2, 2)

    pretty_features = {'LST': 'land surface temperature', 'elevation': 'elevation',
                        'doy': 'day of year', 'day': 'day'}

    ymin = round(min([feature_responses[f]['prediction'].quantile(0.01).min() for f in ['LST', 'elevation', 'day', 'doy']]))
    ymax = round(max([feature_responses[f]['prediction'].quantile(0.99).max() for f in ['LST', 'elevation', 'day', 'doy']]))
    for i, feature in enumerate(['LST', 'elevation', 'day', 'doy']):
        ax = axes.flatten()[i]

        fdf = feature_responses[feature].sort_index().astype(float)
        x = fdf.reset_index()[feature]
        y = fdf['prediction'].reset_index()['prediction'].squeeze()

        # split into n intervals of equal number of samples, and plot the mean and std of each interval
        n_intervals = 15
        intervals = pd.cut(x, bins=n_intervals)

        y_mean = y.groupby(intervals).median()
        y_upper = y.groupby(intervals).quantile(0.75)
        y_lower = y.groupby(intervals).quantile(0.25)
        x_mean = x.groupby(intervals).median()

        ax.plot(x_mean, y_mean, color='blue', lw=2)
        ax.fill_between(x_mean, y_lower, y_upper, color='blue', alpha=0.3)
        ax.set_ylim(ymin, ymax)

        # set x label and title
        ax.set_xlabel(pretty_features.get(feature, feature))
        ax.set_title(f'Partial dependence of {pretty_features.get(feature, feature)} on predicted Tmax')

        # grid and background
        ax.grid(alpha=0.3)

    fig.savefig(figpath + f'grf_feature_response_{model_type}.png', dpi=300, bbox_inches='tight')

#%%

def evaluate_grf(sdir, model_type="rf"):

    global LABEL, figpath
    LABEL = {'rf': 'GRF', 'xgb': 'GXGB'}[model_type]
    pretty_model['grf'] = pretty_model.get(model_type, pretty_model['grf'])
    figpath = os.path.join(FIGDIR, 'grf_evaluation', model_type, sdir) + '/'
    os.makedirs(figpath, exist_ok=True)
    df = retrieve_pickled_results(sdir=sdir, model_type=model_type)
    all_obs, meta, _ = get_harmonised_data()
    
    zz_interped = interpolate_zz_to_stations(meta, recompute=False)
    df = merge_zz_into_df(df, zz_interped)
    mean_df = get_annual_mean_df(df)
    evaluation_plot(x=mean_df['obs'], y=mean_df['pred'], title='annual mean',
                    sfname='grf_eval_annual_mean')
    evaluation_plot(x=df['obs'], y=df['pred'], title='daily', sfname='grf_eval_daily')
    evaluation_plot_with_zz(obs=df['obs'], rf_pred=df['pred'], zz_pred=df['tmax_pred_zz'], title='daily with ZZ comparison', sfname='grf_eval_daily_with_zz')
    evaluation_plot_with_zz(obs=mean_df['obs'], rf_pred=mean_df['pred'], zz_pred=mean_df['tmax_pred_zz'], title='annual mean with ZZ comparison', sfname='grf_eval_annual_mean_with_zz')
    plot_eval_metrics_maps(df, meta, mean_df)
    seasonal_evaluation_plot(df)
    trend_evaluation_plot(df)

    final_model_results = retrieve_final_model_results(sdir='climvalid_obs', model_type=model_type)
    plot_feature_importance_maps(final_model_results)

evaluate_grf(sdir='low_corr', model_type='xgb')

