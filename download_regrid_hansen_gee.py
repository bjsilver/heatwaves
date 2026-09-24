#%%

import ee
import xarray as xr
from constants import ZZ_SOURCE_DIR

# 1. Initialize and Authenticate Earth Engine
ee.Authenticate(auth_mode='notebook')
ee.Initialize()
#%%

target = xr.open_dataset(ZZ_SOURCE_DIR + 'SouthAmerica_Ta_SVCMsp_2003TMAX_daily_tropics30N30S.nc')['Ta_TMAX']

target_crs = 'EPSG:4326'
lon_res = (target.lon[1] - target.lon[0]).item()
lat_res = (target.lat[1] - target.lat[0]).item()

# target.lon/lat are pixel CENTERS, but crsTransform anchors the top-left
# pixel CORNER, so shift back by half a pixel in each direction to align.
target_transform = [
    lon_res,
    0,
    target.lon[0].item() - lon_res / 2,
    0,
    lat_res,
    target.lat[0].item() - lat_res / 2,
]

print('Target CRS:', target_crs)
print('Target transform:', target_transform)

#%%

# 2. Setup Data and ROI (exact extent of the target Ta grid)
dataset = ee.Image('UMD/hansen/global_forest_change_2025_v1_13')
roi = ee.Geometry.BBox(
    target.lon.min().item() - abs(lon_res) / 2,
    target.lat.min().item() - abs(lat_res) / 2,
    target.lon.max().item() + abs(lon_res) / 2,
    target.lat.max().item() + abs(lat_res) / 2,
)

# Extract core bands
tc00 = dataset.select('treecover2000').clip(roi)
loss_year = dataset.select('lossyear').clip(roi)
gain = dataset.select('gain').clip(roi)

#%%



#%%

annual_list = [tc00.rename('year_2000')]

for year_val in range(1, 25):
    actual_year = 2000 + year_val

    forest_mask = loss_year.eq(0).Or(loss_year.gt(year_val))
    year_img = tc00.where(forest_mask.Not(), 0).rename(f'year_{actual_year}')

    annual_list.append(year_img)

annual_stack = ee.Image.cat(annual_list)

regridded_tc = (
    annual_stack
    .reduceResolution(
        reducer=ee.Reducer.mean(),
        maxPixels=2048
    )
    .reproject(
        crs=target_crs,
        crsTransform=target_transform
    )
)

regridded_gain = (
    gain
    .reduceResolution(
        reducer=ee.Reducer.mean(),
        maxPixels=2048
    )
    .reproject(
        crs=target_crs,
        crsTransform=target_transform
    )
    .rename('forest_gain_fraction')
)
# 5. Export Tasks
# Use 'fileFormatConfig' for GeoTIFF compression options
format_options = {
    'cloudOptimized': True
}

task_tc = ee.batch.Export.image.toDrive(
    image=regridded_tc,
    description='Hansen_Annual_Stack_001',
    folder='GEE_Exports',
    fileNamePrefix='hansen_tc_2000_2024',
    region=roi,
    crs=target_crs,
    crsTransform=target_transform,
    maxPixels=1e13,
    formatOptions=format_options
)

task_gain = ee.batch.Export.image.toDrive(
    image=regridded_gain,
    description='Hansen_Gain_001',
    folder='GEE_Exports',
    fileNamePrefix='hansen_gain_fraction',
    region=roi,
    crs=target_crs,
    crsTransform=target_transform,
    maxPixels=1e13,
    formatOptions=format_options
)

task_tc.start()
task_gain.start()


#%% export regridded loss

