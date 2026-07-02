#%%
import xarray as xr
from ufuncs import local_cluster
from constants import STORAGE_DIR

SOURCE_DIR = "/uolstore/Research/b/b0256/Users/earjba/ZhangandZhou/tropical_regions/"



def get_zz_nanmask():

    yearrange = range(2003, 2021)
    ds0 = xr.open_mfdataset(f"{SOURCE_DIR}SouthAmerica_Ta_SVCMsp_{yearrange[0]}TMAX_daily_tropics30N30S.nc",
                                    chunks={'DOY':365, 'lat':250, 'lon':250})
    year_masks = []
    for year in yearrange:
        ds = xr.open_mfdataset(f"{SOURCE_DIR}SouthAmerica_Ta_SVCMsp_{year}TMAX_daily_tropics30N30S.nc",
                                    chunks={'DOY':-1, 'lat':1024, 'lon':1024})
        da = ds['Ta_TMAX']

        year_mask = da.notnull().all(dim='DOY')
        year_mask = year_mask.expand_dims('year')

        year_masks.append(year_mask)

    mask = xr.concat(year_masks, dim='year')
    mask = mask.to_dataset(name='mask')
    mask = mask.astype('int8')

    return mask

#%%
if __name__ == "__main__":
    with local_cluster(n_workers=8, threads_per_worker=4, memory_limit='66GB') as cluster:
        mask = get_zz_nanmask()
        mask.to_zarr(STORAGE_DIR + 'regrid/zz_tmax_nanmask.zarr', mode='w', consolidated=True)

