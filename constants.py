import numpy as np

#%%

### RAW DATA PATHS ###
MODIS_PATH = '/uolstore/Research/b/b0122/earjba/MODIS_LST/modis_v6.1/processed_from_callum/'
ERA5_PATH = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/era5/'
STORAGE_DIR = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/'
HANSEN_DIR = '/uolstore/Research/b/b0256/Users/earbsi/hansen/'
PERCENTILES_DIR = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/percentiles/'
LST_DIR='/uolstore/Research/b/b0256/Users/earbsi/heatwaves/lst/'
MWA_DIR = STORAGE_DIR+ 'mwa_results/'
BORDERS_DIR = '/uolstore/Research/b/b0256/Users/earbsi/borders/'
ZZ_SOURCE_DIR = "/uolstore/Research/b/b0256/Users/earjba/ZhangandZhou/tropical_regions/"
MGWR_RESULTS_DIR = STORAGE_DIR + 'mgwr_results/'

### PROCESSED DATA ###
TMAX_RECHUNKED_PATH = STORAGE_DIR + 'tmax/rechunked/'
TMAX_RECHUNKED_FILE = TMAX_RECHUNKED_PATH + 'tmax_rechunked.zarr'


AREA_BBOX = {'westlimit':-82, 'eastlimit':-34.5, 'southlimit':-30, 'northlimit':20}


FIGDIR = '/uolstore/Research/b/b0256/Users/earbsi/figs/heatwaves/'

# paths to specific files
ZZ_ORIG_DIR = "/uolstore/Research/b/b0256/Users/earjba/ZhangandZhou/tropical_regions/"
ERA_RECHUNKED_PATH = STORAGE_DIR + 'rechunked/era5_skt_p95_1970-1999_rechunked.zarr'
ZZ_RECHUNKED_PATH = STORAGE_DIR + 'rechunked/zhang_zhou_regridded_rechunked.zarr'
COORD_DUMP_PATH = STORAGE_DIR + 'tmax_template_coords.pkl'
COORD_DUMP_PATH = './data/tmax_template_coords.pkl'

PERCENTILE_KEYS = ['zz_2003-2020', 'zz_2003-2020_smooth', 'zz_2003-2020_gapfilled', 'era5_2003-2020']

# chunk sizes
LAT_CHUNK_SIZE = 250
LON_CHUNK_SIZE = 250


RESULTS_BBOX = dict(westlimit=-70, southlimit=-20, eastlimit=-50, northlimit=0)


# heatwave thresholds
HEATDAY_TEMP_THRESH = 28.5
HEATWAVE_DAY_THRESH = 3

# binning used in mwa_bin*.py
IND_VAR_BINS = {
    'cc': np.arange(10, 110, 10),
}

### SVCM PARAMETERS ###
MIN_VALID_YEARS = 12
MIN_VALID_DAYS_PER_YEAR = 50

# +/- days pooled around a target day-of-year across all years (noleap, 1-365)
DOY_POOL_WINDOW = 10
# shared smoothing parameter grid searched over during station cross-validation
LAMBDA_GRID = np.logspace(-3, 2, 6)
CV_N_FOLDS = 5