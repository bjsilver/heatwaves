#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:01:05 2024

@author: ben
"""

import xarray as xr
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt

DPATH = '/nfs/b0256/Users/earbsi/heatwaves/percentile_chunks/'

def preprocess(doy):
    def pp(ds):
        ds = ds.loc[{'DOY':doy}]
        return ds
    return pp

ds = xr.open_mfdataset(DPATH+'percentiles_*.nc', combine='by_coords',
                       preprocess=preprocess(doy=201))
da = ds['Ta_TMAX_95pc']

with ProgressBar():
    da = da.load()
#%%

fig, ax = plt.subplots(figsize=[10,8]) 
da.plot.imshow(ax=ax, vmin=300, vmax=370, cmap='magma')
fig.savefig('/nfs/see-fs-02_users/earbsi/od/craft/figs/heatwaves/doy_201_95CI.png',
            dpi=600, bbox_inches='tight')