#%%

import xarray as xr
import rioxarray as rxr

TIF_DIR = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/zhang_zhou_tifs/'
NC_OUTPUT_DIR = '/uolstore/Research/b/b0256/Users/earjba/ZhangandZhou/tropical_regions/'

# get a template dataset to reformat tifs to
tds = xr.open_dataset(NC_OUTPUT_DIR + 'SouthAmerica_Ta_SVCMsp_2007TMAX_daily_tropics30N30S.nc')

year = 2003

tif_path = TIF_DIR+f'{year}/gf_Day{year}_099.tif'
tif = rxr.open_rasterio(tif_path, masked=True).squeeze()
lltif = tif.rio.reproject('epsg:4326')

# slice to tds
sltif = lltif.loc[{'y':slice(tds.lat.max(), tds.lat.min()), 'x':slice(tds.lon.min(), tds.lon.max())}]

print(tds)
print(sltif)


tif.coarsen()