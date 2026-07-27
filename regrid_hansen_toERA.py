#%%

from ufuncs import open_hansen
import cf_xarray
import xarray as xr
import xesmf as xe
from constants import ERA5_PATH, HANSEN_DIR
from ufuncs import local_cluster
from calc_era5_percentiles import load_era5_t2m
import os



def load_data():
    hansen = open_hansen()
    era = load_era5_t2m()
    era = era.to_dataset(name='t2m')

    hansen = hansen.to_dataset(name='tree_cover')
    hansen['mask'] = (hansen['tree_cover'] > 0).all('time').astype(int)

    hansen = hansen.cf.add_bounds(['lat', 'lon'])
    era = era.cf.add_bounds(['lat', 'lon'])
    return era, hansen

def load_regridder(era, hansen):
    if not os.path.exists(ERA5_PATH+'hansen_to_era5_regridder.nc'):
        regridder = xe.Regridder(hansen, era, method='conservative_normed', periodic=False)
        regridder.to_netcdf(ERA5_PATH+'hansen_to_era5_regridder.nc')
    else:
        regridder = xe.Regridder(hansen, era, method='conservative_normed', periodic=False, 
                                reuse_weights=True, filename=ERA5_PATH+'hansen_to_era5_regridder.nc')
    return regridder

def main():
    era, hansen = load_data()
    regridder = load_regridder(era, hansen)
    hansen_regridded = regridder(hansen)
    hansen_regridded.to_zarr(HANSEN_DIR+'hansen_regridded.zarr', mode='w')
#%%
if __name__ == '__main__':
    with local_cluster(n_workers=16, threads_per_worker=1, memory_limit='8GB'):
        main()

#%%
hansen_regridded = xr.open_zarr(HANSEN_DIR+'hansen_regridded.zarr')
diff = hansen_regridded['tree_cover'][-1] - hansen_regridded['tree_cover'][0]
diff.plot.imshow()