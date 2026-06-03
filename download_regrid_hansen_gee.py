#%%
import ee
import geemap
import xarray as xr
import matplotlib.pyplot as plt

# 1. Initialize and Authenticate Earth Engine
ee.Authenticate(auth_mode='notebook')
ee.Initialize()
geemap.ee_initialize()

#%%

# 2. Setup Data and ROI (South America / Amazon BBox)
dataset = ee.Image('UMD/hansen/global_forest_change_2025_v1_13')
roi = ee.Geometry.BBox(-82, -30, -34.5, 20) 
# test small roi
#roi = ee.Geometry.BBox(-63, -6, -62, -5)

# Extract core bands
tc00 = dataset.select('treecover2000').clip(roi)
loss_year = dataset.select('lossyear').clip(roi)
gain = dataset.select('gain').clip(roi)


#%%

# 3. Build the Annual Time Series (2000-2024)
annual_list = [tc00.rename('year_2000')]

for year_val in range(1, 25): 
    actual_year = 2000 + year_val
    # Logic: Keep forest if loss hasn't happened yet (loss_year == 0 or > current year)
    mask = loss_year.eq(0).where(loss_year.gt(year_val), 1)
    year_img = tc00.where(mask.Not(), 0).rename(f'year_{actual_year}')
    annual_list.append(year_img)

annual_stack = ee.Image.cat(annual_list)

# 4. Regridding (0.01 degree)
target_crs = 'EPSG:4326'
grid_size = 0.01 
target_transform = [grid_size, 0, -180.005, 0, -grid_size, 90.005]

regridded_tc = annual_stack.reduceResolution(
    reducer=ee.Reducer.mean(), 
    maxPixels=2048
).reproject(crs=target_crs, crsTransform=target_transform)

regridded_gain = gain.reduceResolution(
    reducer=ee.Reducer.mean(),
    maxPixels=2048
).reproject(crs=target_crs, crsTransform=target_transform).rename('forest_gain_fraction')

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

