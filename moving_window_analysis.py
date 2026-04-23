import rioxarray as rxr
HANSEN_DIR = '/uolstore/Research/b/b0256/Users/earbsi/hansen/'

da = rxr.open_rasterio(HANSEN_DIR + 'hansen_tc_2000_2024.tif', masked=True).squeeze()
del da.attrs['long_name']
da = da.drop('spatial_ref')
da = da.rename({'band':'time'})
da['time'] = [2000+i for i in range(da.sizes['time'])]