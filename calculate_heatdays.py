#%%
import xarray as xr
from dask.distributed import Client, LocalCluster
from calculate_tmax_percentiles import load_tmax_da

STORAGE_DIR = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/'


pctl = xr.open_zarr(STORAGE_DIR+'/percentiles/tmax_95pc.zarr',
                    consolidated=True)
pctl = pctl['Ta_TMAX_p95']
tmax = load_tmax_da()
# %%

if __name__ == '__main__':

    with LocalCluster(n_workers=8,       
        threads_per_worker=1,  
        memory_limit='15GB',  
        dashboard_address=':8787') as cluster:
            with Client(cluster) as client:

                heatdays = (tmax > 28.5) & (tmax > pctl)
                heatdays.name = 'heatdays'
                heatdays.to_zarr(STORAGE_DIR+'/heatdays/heatdays.zarr',
                                 consolidated=True)