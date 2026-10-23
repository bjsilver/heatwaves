#%%
import dask
import xarray as xr
import rioxarray as rxr
import xesmf as xe
import numpy as np
import os
from constants import LST_DIR, AREA_BBOX, HANSEN_DIR, LAT_CHUNK_SIZE, LON_CHUNK_SIZE
from pyproj import Transformer
import joblib
import pandas as pd
import datetime
import cftime
import itertools

N_JOBS = 16  # Number of parallel jobs to run
YEARS = range(2003, 2021)  # Years to process
_WORKER_REGRIDDER = None


def get_source_grid():
    raster = rxr.open_rasterio(
        LST_DIR+'zhang_zhou_tifs/2003/gf_Day2003_001.tif'
    ).squeeze()
    westlimit = AREA_BBOX['westlimit'] -1
    eastlimit = AREA_BBOX['eastlimit'] +1
    southlimit = AREA_BBOX['southlimit'] -1
    northlimit = AREA_BBOX['northlimit'] +1
    try:
        x_coords = raster.x.values
        y_coords = raster.y.values
        to_source = Transformer.from_crs(
            "EPSG:4326", raster.rio.crs, always_xy=True
        )
        x_min, y_min, x_max, y_max = to_source.transform_bounds(
            westlimit, southlimit, eastlimit, northlimit, densify_pts=101
        )

        x_indices = np.flatnonzero((x_coords >= x_min) & (x_coords <= x_max))
        y_indices = np.flatnonzero((y_coords >= y_min) & (y_coords <= y_max))
        if not x_indices.size or not y_indices.size:
            raise ValueError("AREA_BBOX does not intersect the source raster")

        x_coords = x_coords[x_indices[0]:x_indices[-1] + 1]
        y_coords = y_coords[y_indices[0]:y_indices[-1] + 1]
        to_geographic = Transformer.from_crs(
            raster.rio.crs, "EPSG:4326", always_xy=True
        )
        longitude_mask = np.zeros(x_coords.size, dtype=bool)
        latitude_mask = np.zeros(y_coords.size, dtype=bool)

        # Scan in tiles so the mask does not require full-raster coordinate arrays.
        tile_size = 512
        for y_start in range(0, y_coords.size, tile_size):
            y_stop = min(y_start + tile_size, y_coords.size)
            for x_start in range(0, x_coords.size, tile_size):
                x_stop = min(x_start + tile_size, x_coords.size)
                grid_x, grid_y = np.meshgrid(
                    x_coords[x_start:x_stop],
                    y_coords[y_start:y_stop],
                )
                lons, lats = to_geographic.transform(grid_x, grid_y)
                in_bounds = (
                    (lons >= westlimit) & (lons <= eastlimit)
                    & (lats >= southlimit) & (lats <= northlimit)
                )
                latitude_mask[y_start:y_stop] |= in_bounds.any(axis=1)
                longitude_mask[x_start:x_stop] |= in_bounds.any(axis=0)

        valid_x = np.flatnonzero(longitude_mask)
        valid_y = np.flatnonzero(latitude_mask)
        if not valid_x.size or not valid_y.size:
            raise ValueError("AREA_BBOX does not intersect the source raster")

        x_coords = x_coords[valid_x[0]:valid_x[-1] + 1]
        y_coords = y_coords[valid_y[0]:valid_y[-1] + 1]
        grid_x, grid_y = np.meshgrid(x_coords, y_coords)
        lons, lats = to_geographic.transform(grid_x, grid_y)
        return x_coords, y_coords, lons, lats
    finally:
        raster.close()


def open_LST_tif(year, doy, x_coords, y_coords, lons, lats):
    da = xr.open_dataarray(LST_DIR+f'zhang_zhou_tifs/{year}/gf_Day{year}_{doy:03d}.tif',
                         engine='rasterio')
    da = da.squeeze()

    da = da.sel(y=y_coords, x=x_coords)
    da = da.assign_coords(lon=(('y', 'x'), lons), lat=(('y', 'x'), lats))

    da.name = 'LST'
    return da


def get_regridder(lons, lats):
    target = xr.open_zarr(HANSEN_DIR+'hansen_tc_2000_2024_rechunked.zarr')
    target = target.isel(time=0)
    try:
        source_grid = xr.Dataset({
            'lat': (('y', 'x'), lats),
            'lon': (('y', 'x'), lons),
        })
        target_grid = xr.Dataset({
            'lat': target['lat'],
            'lon': target['lon'],
        })

        regrid_path = LST_DIR+'zhang_zhou_tifs/regridder_LST_to_t2m_weights.nc'
        if not os.path.exists(regrid_path):
            regridder = xe.Regridder(source_grid, target_grid, 'bilinear')
            regridder.to_netcdf(regrid_path)
        else:
            regridder = xe.Regridder(
                source_grid, target_grid, 'bilinear', weights=regrid_path
            )
        return regridder
    finally:
        target.close()




#%%

def get_worker_regridder(lons, lats):
    global _WORKER_REGRIDDER

    if _WORKER_REGRIDDER is None:
        print("Initializing regridder for worker...")
        _WORKER_REGRIDDER = get_regridder(lons, lats)

    return _WORKER_REGRIDDER

def regridder_worker(years_doys, x_coords, y_coords, lons, lats):

    regridder = get_worker_regridder(lons, lats)

    das = []
    for year, doy in years_doys:
        lst = open_LST_tif(year, doy, x_coords, y_coords, lons, lats)
        try:
            lst_regridded = regridder(lst)
            # get the cftime noleap timestamp
            doy_dt = cftime.DatetimeNoLeap(int(year), 1, 1) + datetime.timedelta(days=int(doy) - 1)
            lst_regridded = lst_regridded.expand_dims(time=[doy_dt])
            das.append(lst_regridded.load())
        finally:
            lst.close()
            print(f"Finished regridding year {year}, doy {doy}")

    return das

def regrid_year(year, x_coords, y_coords, lons, lats):
    print(f"Regridding year {year}...")
    doys = range(1, 366)
    years_doys = list(itertools.product([year], doys))

    # split the doys for year into chunks for parallel processing
    chunks = np.array_split(years_doys, N_JOBS)

    # process each chunk in parallel
    results = joblib.Parallel(n_jobs=N_JOBS)(
        joblib.delayed(regridder_worker)(chunk, x_coords, y_coords, lons, lats)
        for chunk in chunks
    )
    # flatten the list of lists
    das = [item for sublist in results for item in sublist]
    return das

def rechunk_zarr(combined):
    # rechunk the combined dataset
    with dask.config.set({
                    "array.rechunk.method": "p2p",
                    "optimization.fuse.active": False,
                }):
        rechunked = combined.chunk({'time': 365, 'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})
    return rechunked

def write_year_to_zarr(das, year):
    print(f"Writing year {year} to zarr...")
    # concatenate along time dimension
    combined = xr.concat(das, dim='time')

    rechunked = rechunk_zarr(combined)
    rechunked.name = 'LST'

    # write to zarr
    if year == YEARS[0]:
        rechunked.to_zarr(LST_DIR+'regridded_rechunked.zarr', mode='w')
    else:
        rechunked.to_zarr(LST_DIR+'regridded_rechunked.zarr', mode='a', append_dim='time')

    print(f"Finished writing year {year} to zarr.")



def main():
    x_coords, y_coords, lons, lats = get_source_grid()
    # initialise the regridder
    regridder = get_regridder(lons, lats)
    del regridder

    get_regridder(lons, lats)  # Initialize the regridder
    for year in YEARS:
        results = regrid_year(year, x_coords, y_coords, lons, lats)
        write_year_to_zarr(results, year)

#%%
if __name__ == "__main__":
    main()

