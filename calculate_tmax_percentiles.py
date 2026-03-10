#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 09:38:34 2024

@author: ben
"""

import xarray as xr
from dask.diagnostics import ProgressBar
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
import time
import regionmask
import warnings


DPATH = '/scratch/earjba/'
PAD = 7
TEMP_DIR = '/scratch/earbsi/heatwaves/'
MP_CORES = 30

def check_if_land(chunk):
    
    lat_slicer, lon_slicer = chunk


    lon = np.arange(-179.5, 180, .1)
    lat = np.arange(-89.5, 90, .1)
    
    minlat, maxlat = lat_slicer.stop, lat_slicer.start
    minlon, maxlon = lon_slicer.start, lon_slicer.stop
    
    mask = regionmask.defined_regions.natural_earth_v4_1_0.land_110.mask(lon, lat)
    
    mask_in = mask.loc[{'lat':slice(minlat, maxlat), 'lon':slice(minlon, maxlon)}]
    
    if mask_in.count() > 0:
        return True
    else:
        return False



def pad_start_end_doys(da, pad=7):

    # take start and move to end
    end_pad = da.loc[{'DOY':slice(1, 1+pad-1)}]
    end_pad.coords['DOY'] = end_pad.DOY + 365
    
    # take end and move to before start
    start_pad = da.loc[{'DOY':slice(365-pad+1, 365)}]
    start_pad.coords['DOY'] = start_pad.DOY - 365
    
    da = xr.concat([start_pad, da, end_pad], dim='DOY')
    return da




def preprocess(lat_slicer, lon_slicer):
    def pp(ds):
        ds = ds.loc[{'lat':lat_slicer, 'lon':lon_slicer}]
        ds = ds.drop_vars('spatial_ref')
        ds.coords['DOY'] = range(1, 366)
        return ds
    return pp

def load_area_chunk(lat_slicer, lon_slicer):


    ds = xr.open_mfdataset(DPATH+'SouthAmerica_Ta_SVCMsp_*TMAX_daily_tropics30N30S.nc',
                           preprocess=preprocess(lat_slicer, lon_slicer), 
             
                           concat_dim='DOY', combine='nested',
                           chunks={'DOY':-1},
                           parallel=True)
    da = ds['Ta_TMAX']
    
    print('loading chunk...')
    with ProgressBar():
        da = da.load()
    
    da = da.sortby('DOY')
    
    da = pad_start_end_doys(da, pad=PAD)
    
    return da

def get_quantiles(doy_da):    
    doys_in = np.unique(doy_da.DOY)[PAD:-PAD]

    res_da = xr.DataArray(coords=doy_da.coords)
    res_da = res_da.drop_duplicates(dim='DOY')

    for doy in doys_in:
                
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            p = doy_da.loc[{'DOY':slice(doy-PAD, doy+PAD)}].quantile(dim='DOY', q=.95)
        p = p.drop_vars('quantile')
       
        p = p.assign_coords({'DOY':doys_in[PAD]})
        res_da.loc[{'DOY':doy}] = p
    
    res_da = res_da.dropna('DOY', how='all')
    return res_da


def calculate_percentiles_mp(da):
    
    
    doy_chunks = np.array_split(list(range(1,366)), MP_CORES)
    doy_das = []
    for doy_chunk in doy_chunks:
        
        doy_da = da.loc[{'DOY':slice(doy_chunk[0] - PAD,
                                       doy_chunk[-1] + PAD)}]
        doy_das.append(doy_da)
    
    
    print('multiprocessing...')
    with mp.Pool(MP_CORES) as pool:
        with tqdm(total=len(doy_das)) as pbar:
            results = []
            for result in pool.imap_unordered(get_quantiles, doy_das):
                results.append(result)
                pbar.update()

    
    
    percentiles = xr.concat(results, dim='DOY').sortby('DOY')
    
    percentiles.name = 'Ta_TMAX_95pc'
    return percentiles

def get_percentiles_chunk(chunk):
    
    lat_slicer, lon_slicer = chunk
    
    print('calculating percentiles for',
          f'lat: {lat_slicer.start}—{lat_slicer.stop}',
          f'lon: {lon_slicer.start}—{lon_slicer.stop}')
    
    start_time = time.time()
    
    da = load_area_chunk(lat_slicer, lon_slicer)
    percentiles = calculate_percentiles_mp(da)
    
    end_time = time.time()
    
    print(f'time taken for chunk {end_time-start_time:.1f} seconds')
    
    return percentiles

def get_amazon_chunks():

    chunks = []
    for lat in range(-20, 0, 2):
        for lon in range(-75, -45, 3):
            lat_slicer = slice(lat+2,lat)
            lon_slicer = slice(lon, lon+3)
            chunks.append((lat_slicer, lon_slicer))
            
    return chunks


#%%

if __name__ == '__main__':
            
    chunks = get_amazon_chunks()
    
    encoding = {'Ta_TMAX_95pc':{'zlib':True, 'complevel':5}}
    
    for chunk in chunks:
        
        if not check_if_land(chunk):
            print('no land in chunk...')
            continue
        
        percentiles = get_percentiles_chunk(chunk)
        
        fname = f'percentiles_{chunk[0].start}_{chunk[1].start}.nc'
        percentiles.to_netcdf(TEMP_DIR+fname, encoding=encoding)
    