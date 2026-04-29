#%%
import rioxarray as rxr
import xarray as xr
import pandas as pd
import os

def open_hansen_canopy_cover():
    HANSEN_DIR = '/uolstore/Research/b/b0256/Users/earbsi/hansen/'

    if not os.path.exists(HANSEN_DIR + 'hansen_tc_2000_2024.nc'):

        da = rxr.open_rasterio(HANSEN_DIR + 'hansen_tc_2000_2024.tif', masked=True).squeeze()
        del da.attrs['long_name']
        da = da.drop('spatial_ref')
        da = da.rename({'band':'time', 'x':'lon', 'y':'lat'})
        da['time'] = [2000+i for i in range(da.sizes['time'])]
        da.name = 'tree_cover'
        encoding = {var: {'zlib': True, 'complevel': 5} for var in ['tree_cover']}
        da.to_netcdf(HANSEN_DIR + 'hansen_tc_2000_2024.nc', encoding=encoding)
    else:
        ds = xr.open_dataset(HANSEN_DIR + 'hansen_tc_2000_2024.nc',
                             chunks={'time':1})
        da = ds['tree_cover']
    return da

cc = open_hansen_canopy_cover()

#%%
# gets all co-ordinate combinations except centre
window_radius = 1
shifts = []
for i in range(-window_radius, window_radius+1):
    for j in range(-window_radius, window_radius+1):
        if i == 0 and j == 0: 
            continue
        shifts.append((i, j))

# create a ds with the window as a new dimension
shifted_dss = [cc.shift(lat=dx, lon=dy) for dx, dy in shifts]
nds = xr.concat([cc] + shifted_dss, dim=pd.RangeIndex(len(shifts)+1, name='neighbour'))

#%% get indices of loss
loss = (cc.diff('time') < 0)
# make mask of pre-loss data
pre_loss_mask = loss.cumsum('time') == 0


