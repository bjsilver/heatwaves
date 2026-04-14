#%%
import xarray as xr
import pandas as pd
import numpy as np
from constants import MODIS_PATH, HANSEN_PATH
import matplotlib.pyplot as plt


AMAZON_BBOX = {'latitude':slice(-30, 20), 'longitude':slice(-82, -34.5)}

#%%
def get_modis_temperature_change(year_window):
    # NOTE: does not account for missing data in any way

    ds = xr.open_dataset(MODIS_PATH+'MOD11A2_LST_0.25deg_2001–2021.nc')['LST_Day_1km']
    years = np.unique(ds.time.dt.year)
    start_years = years[:year_window]
    end_years = years[-year_window:]
    modis_start = ds.sel(time=ds.time.dt.year.isin(start_years)).mean('time')
    modis_end = ds.sel(time=ds.time.dt.year.isin(end_years)).mean('time')
    temperature_change = modis_end - modis_start
    return temperature_change

def get_hansen_deforestation():
    ds = xr.open_mfdataset(HANSEN_PATH+'treeFrac*_0.25_hansen-landsat_mon_0.25deg_conserv.nc',
                        compat='override')
    defor = ds['treeFrac2003_0_05'] - ds['treeFrac2020_0_05'] 
    return defor.compute()

def ensure_match_coords(ds1, ds2):
    lat_coords = np.intersect1d(ds1.latitude.values, ds2.latitude.values)
    lon_coords = np.intersect1d(ds1.longitude.values, ds2.longitude.values)
    ds1 = ds1.sel(latitude=lat_coords, longitude=lon_coords)
    ds2 = ds2.sel(latitude=lat_coords, longitude=lon_coords)
    return ds1, ds2

def prepare_data():
    modis = get_modis_temperature_change(year_window=3)
    hansen = get_hansen_deforestation()

    modis = modis.sel(AMAZON_BBOX)
    hansen = hansen.sel(AMAZON_BBOX)

    modis, hansen = ensure_match_coords(modis, hansen)

    ds = xr.Dataset({
        'd_temp': modis,
        'd_cc': hansen
    })
    return ds.compute()

def generate_shifts(outer_radius):
    shifts = []
    for i in range(-outer_radius, outer_radius+1):
        for j in range(-outer_radius, outer_radius+1):
            if i == 0 and j == 0: # i.e. gets all co-ordinate combinations except centre
                continue
            shifts.append((i, j))
    return shifts

def moving_window_analysis(ds, window_radius):
    #TODO: do I need to add to the valid mask a minimum number of neighbours
    #TODO: do I need a threshold for the minimum CC loss difference to be included in the results? 

    results = xr.Dataset(coords=ds.coords)
    shifts = generate_shifts(window_radius)

    # create a ds with the window as a new dimension
    shifted_dss = [ds.shift(latitude=dx, longitude=dy) for dx, dy in shifts]
    nds = xr.concat(shifted_dss, dim=pd.RangeIndex(len(shifts), name='neighbour'))

    # only keep pixels where:
    valid = [
        ds['d_cc'] > 0, #  CC loss data present and >0
        ds['d_cc'] < nds['d_cc'].max('neighbour'), # at least one neighbour with higher CC loss
        ds['d_temp'].notnull(), # temperature change data present
        # how about edge pixels?
    ]
    valid_mask = np.logical_and.reduce(valid)

    # CC loss difference from centre for each neighbour
    cc_loss_diff = nds['d_cc'] - ds['d_cc']
    # get chance in CC for the neighbour with greatest CC loss difference
    cc_loss_diff_max = cc_loss_diff.max('neighbour')
    # get indices of neighbour with greatest CC loss difference
    max_diff_idx = cc_loss_diff.fillna(-np.inf).argmax('neighbour')
    # get temperature change for neighbour with greatest CC loss difference
    temp_change_diff = nds['d_temp'].isel(neighbour=max_diff_idx) - ds['d_temp']

    results['cc_loss'] = cc_loss_diff_max.where(valid_mask)
    results['temp_change'] = temp_change_diff.where(valid_mask)
    return results

ds = prepare_data()
results = moving_window_analysis(ds, window_radius=2)



# group by cc_loss_diff_max bins
bins = np.arange(0, 60, 5)
binned = results['temp_change'].groupby_bins(results['cc_loss'], bins).mean() 
n_in_bins = results['temp_change'].groupby_bins(results['cc_loss'], bins).count()

#%%
fig, ax = plt.subplots(figsize=(10, 6))
binned.to_series().plot(kind='bar', ax=ax, color='steelblue')
ax.set_xlabel('Canopy cover loss bin (%)', fontsize=12)
ax.set_ylabel('neighbour mean temperature change (°C)', fontsize=12)
bin_dim = next(dim for dim in binned.dims if dim.endswith('_bins'))
ax.set_xticklabels(
    [f'{int(interval.left)}-{int(interval.right)}' for interval in binned[bin_dim].values],
    rotation=0,
)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add pixel counts on top of bars
for i, (bar, count) in enumerate(zip(ax.patches, n_in_bins.values)):

    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'n={int(count)}',
            ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()