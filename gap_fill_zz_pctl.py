#%%
import xarray as xr
from constants import STORAGE_DIR, HANSEN_DIR
import os
from dask.diagnostics import ProgressBar

TEST_SLICER = {'lat':slice(-5, 0), 'lon':slice(-60, -55)}

pct = xr.open_zarr(STORAGE_DIR + 'rechunked/zz_pct_rechunked.zarr',
                    consolidated=True)['Ta_TMAX_p95'] / 10
pct = pct[:, :-1, :-1] 

ds = xr.open_zarr(os.path.join(HANSEN_DIR, 'hansen_tc_2000_2024_rechunked.zarr'))
tc = ds['tree_cover']
loss = -(tc - tc[0])
loss = loss.chunk({'time': -1})
loss = loss.assign_coords({'lat': pct.lat, 'lon': pct.lon})

#%%
# get the cells where all-time loss is less than 20%
ok = loss.max('time') < 20
pct_ok = pct.where(ok)

with ProgressBar():
    pct_ok = pct_ok.loc[TEST_SLICER].compute(num_workers=32, memory_limit='12GB')


#%%
max_gap = 20

interped_lat = pct_ok[140].interpolate_na(dim='lat', method='spline', max_gap=max_gap,
                                         use_coordinate=False)
interped_lon = pct_ok[140].interpolate_na(dim='lon', method='spline', max_gap=max_gap,
                                         use_coordinate=False)

# combine
interped = pct_ok[140].copy()

# where interped is null and interped_lat is not null, and interped_lon is null, use interped_lat
interped = interped.where(~interped.isnull() | interped_lat.isnull() | ~interped_lon.isnull(), interped_lat)
# where interped is null and interped_lon is not null, and interped_lat is null, use interped_lon
interped = interped.where(~interped.isnull() | interped_lon.isnull() | ~interped_lat.isnull(), interped_lon)
# where interped is null and both interped_lat and interped_lon are not null, use the average of the two
interped = interped.where(~interped.isnull() | interped_lat.isnull() | interped_lon.isnull(), (interped_lat + interped_lon) / 2)
interped.plot.imshow()
