#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 09:33:48 2026

@author: ben
"""
import xarray as xr
from glob import glob
from calculate_tmax_percentiles import (get_amazon_chunks,
                                        check_if_land)
import matplotlib.pyplot as plt

PERCENTILES_DIR = '/uolstore/Research/b/b0122/earjba/heatwaves/percentiles_from_ben/'


chunk_paths = glob(PERCENTILES_DIR+'percentiles*.nc')

encoding = {'Ta_TMAX_95pc':{'zlib':True, 'complevel':5}}


    
    
#%% load_percentile_chunk

for chunk_path in chunk_paths:
    


    da = xr.open_dataarray(chunk_path)
    
    fig, ax = plt.subplots()
    da[0].plot(ax=ax)
    plt.show()
    plt.close()