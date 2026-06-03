# FILEPATHS


MODIS_PATH = '/uolstore/Research/b/b0122/earjba/MODIS_LST/modis_v6.1/processed_from_callum/'
ERA5_PATH = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/era5/'
STORAGE_DIR = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/'
HANSEN_DIR = '/uolstore/Research/b/b0256/Users/earbsi/hansen/'
PERCENTILES_DIR = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/percentiles/'
HANSEN_PATH = '/uolstore/Research/b/b0122/ee13c2s/datasets/hansen_v1.9/output/'

FIGDIR = '/uolstore/Research/b/b0256/Users/earbsi/figs/heatwaves/'

# paths to specific files
ZZ_ORIG_DIR = "/uolstore/Research/b/b0256/Users/earjba/ZhangandZhou/tropical_regions/"
ERA_RECHUNKED_PATH = STORAGE_DIR + 'rechunked/era5_skt_p95_1970-1999_rechunked.zarr'
ZZ_RECHUNKED_PATH = STORAGE_DIR + 'rechunked/zhang_zhou_regridded_rechunked.zarr'

PERCENTILE_KEYS = ['era_1970-1999', 'era_2003-2020', 'zz_2003-2020']


# areas
AMAZON_BBOX = {'latitude':slice(-30, 20), 'longitude':slice(-82, -34.5)}