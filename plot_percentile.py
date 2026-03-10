#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:01:05 2024

@author: ben
"""
#%%
import xarray as xr
from dask.diagnostics import ProgressBar
import multiprocessing as mp
import matplotlib.pyplot as plt
import os
import subprocess


#DPATH = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/percentile_chunks/'
DPATH = '/local/data/earbsi/scratch/percentile_chunks/'
FIGDIR = '/uolstore/home/users/earbsi/onedrive/craft/figs/heatwaves/'

CPU_COUNT = 30

# def preprocess(doy):
#     def pp(ds):
#         ds = ds.loc[{'DOY':doy}]
#         return ds
#     return pp

ds = xr.open_mfdataset(DPATH+'percentiles_*.nc', combine='by_coords')
da = ds['Ta_TMAX_95pc']

with ProgressBar():
    da = da.load()
#%%

# chunk da into CPU_COUNT chunks along the DOY dimension
das = [da.loc[{'DOY':doy}].load() for doy in range(1, 366)]

# make iterable of chunks of DOY
#%%

def plot_doy(doy_chunk):
    doy = int(doy_chunk.DOY)
    

    fig, ax = plt.subplots(figsize=[10,8]) 
    doyda = doy_chunk / 10
    doyda.plot.imshow(ax=ax, vmin=26, vmax=42, cmap='magma',
                      cbar_kwargs={'extend':'both',
                                    'label':'95th percentile of daily maximum temperature (°C)'})
    fig.savefig(FIGDIR+f'doy_{str(doy).zfill(3)}_Ta_TMAX_95pc.png',
                dpi=400, bbox_inches='tight')
    plt.close()
    print(f'Plotted DOY {doy}')

if __name__ == '__main__':
    with mp.Pool(CPU_COUNT) as pool:
        pool.map(plot_doy, das)
# %% make the DOY plots into a video using ffmpeg


input_pattern = FIGDIR + "doy_%03d_Ta_TMAX_95pc.png"
output_mp4 = FIGDIR + "doy_Ta_TMAX_95pc.mp4"

cmd = [
    "ffmpeg",
    "-y",
    "-framerate", "10",
    # "-start_number", "0",  # uncomment if files start at 000 instead of 001
    "-i", input_pattern,
    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    output_mp4
]

subprocess.run(cmd, check=True)
print(f"Created: {output_mp4}")