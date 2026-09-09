#%%
import ee
import xarray as xr
import matplotlib.pyplot as plt
from constants import STORAGE_DIR, AREA_BBOX

# 1. Initialize and Authenticate Earth Engine
ee.Authenticate(auth_mode='notebook')
ee.Initialize()
#%%

# 2. Setup Data and ROI (South America / Amazon BBox)
dataset = ee.Image('UMD/hansen/global_forest_change_2025_v1_13')
roi = ee.Geometry.BBox(
    AREA_BBOX['westlimit'],
    AREA_BBOX['southlimit'],
    AREA_BBOX['eastlimit'],
    AREA_BBOX['northlimit']
)
# test small roi
#roi = ee.Geometry.BBox(-63, -6, -62, -5)

# Extract core bands
tc00 = dataset.select('treecover2000').clip(roi)
loss_year = dataset.select('lossyear').clip(roi)
gain = dataset.select('gain').clip(roi)


#%%

annual_list = [tc00.rename('year_2000')]

for year_val in range(1, 25):
    actual_year = 2000 + year_val

    forest_mask = loss_year.eq(0).Or(loss_year.gt(year_val))
    year_img = tc00.where(forest_mask.Not(), 0).rename(f'year_{actual_year}')

    annual_list.append(year_img)

annual_stack = ee.Image.cat(annual_list)

# 4. Regridding (0.01 degree)
tmax_collection = ee.ImageCollection(
    'projects/sat-io/open-datasets/global-daily-air-temp/latin_america'
)

tmax_img = (
    tmax_collection
    .filterDate('2003-01-01', '2003-01-02')
    .filter(ee.Filter.stringContains('system:index', 'TMAX'))
    .first()
)

tmax_proj = tmax_img.select('b1').projection()

target_crs = 'EPSG:4326'
target_transform = [
    0.008055117514264009,
    0,
    -154.88527132920055,
    0,
    -0.008055117514264009,
    28.075000000030002,
]

print('TMAX CRS:', target_crs)
print('TMAX transform:', target_transform)

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

