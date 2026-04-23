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
import numpy as np
import subprocess
from calculate_tmax_percentiles import load_tmax_da
from constants import FIGDIR
STORAGE_DIR = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/'

CPU_COUNT = 16

with ProgressBar():
    tmax = load_tmax_da().load()

pctl = xr.open_zarr(STORAGE_DIR+'/percentiles/tmax_95pc.zarr',
                    consolidated=True)
pctl = pctl['Ta_TMAX_p95']
pctl.coords['lat'] = np.where(np.isnan(pctl.lat.values), 0, pctl.lat.values)

#%%


# chunk da into CPU_COUNT chunks along the DOY dimension


das = [pctl.loc[{'doy':doy}].load() for doy in range(1, 366)]

# make iterable of chunks of DOY
#%%

def plot_doy(doy_chunk):
    doy = int(doy_chunk.doy)
    

    fig, ax = plt.subplots(figsize=[10,8]) 
    doyda = doy_chunk / 10
    doyda.plot.imshow(ax=ax, vmin=26, vmax=42, cmap='magma',
                      cbar_kwargs={'extend':'both',
                                    'label':'95th percentile of daily maximum temperature (°C)'})
    fig.savefig(FIGDIR+f'doy_pctls/doy_{str(doy).zfill(3)}_Ta_TMAX_95pc.png',
                dpi=400, bbox_inches='tight')
    plt.close()
    print(f'Plotted DOY {doy}')

if __name__ == '__main__':
    with mp.Pool(CPU_COUNT) as pool:
        pool.map(plot_doy, das)
# %% make the DOY plots into a video using ffmpeg


input_pattern = FIGDIR + "doy_pctls/doy_%03d_Ta_TMAX_95pc.png"
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