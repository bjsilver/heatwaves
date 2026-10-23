#%%
import zarr
from pathlib import Path
import json
import shutil
import xarray as xr
import numpy as np
from constants import LST_DIR, STORAGE_DIR, PERCENTILE_KEYS, HANSEN_DIR, BORDERS_DIR, TMAX_RECHUNKED_PATH, LAT_CHUNK_SIZE, LON_CHUNK_SIZE, COORD_DUMP_PATH
from dask.distributed import Client, LocalCluster
import warnings
from contextlib import contextmanager
import regionmask
import geopandas as gpd
import os
import pickle
import itertools

def lon_to_180(ds):
    ds = ds.copy()
    ds['longitude'] = ((ds['longitude'] + 180) % 360) - 180
    ds = ds.sortby('longitude')
    return ds



def add_metadata_to_zarr(
    store_path,
    var_name="Ta_TMAX",
    dims=("time", "lat", "lon"),
    consolidate=True,
    orig_ds=None,
    orig_store_path=None,
    orig_consolidated=False,
):
    store = Path(store_path)
    var_dir = store / var_name
    var_dir.mkdir(exist_ok=True)

    # Move root-array payload under variable name
    for p in list(store.iterdir()):
        if p.name in {var_name, ".zgroup", ".zattrs", ".zmetadata"}:
            continue
        shutil.move(str(p), str(var_dir / p.name))

    # Root must be a group for xr.open_zarr
    (store / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    if not (store / ".zattrs").exists():
        (store / ".zattrs").write_text("{}")

    # Variable must declare dimension names for xarray
    attrs_path = var_dir / ".zattrs"
    attrs = json.loads(attrs_path.read_text()) if attrs_path.exists() else {}
    attrs["_ARRAY_DIMENSIONS"] = list(dims)
    attrs_path.write_text(json.dumps(attrs))

    opened_orig_ds = False
    if orig_ds is None and orig_store_path is not None:
        orig_ds = xr.open_zarr(orig_store_path, consolidated=orig_consolidated)
        opened_orig_ds = True

    if orig_ds is not None:
        coord_candidates = {
            "time": ["time"],
            "lat": ["lat", "latitude"],
            "lon": ["lon", "longitude"],
        }
        coord_vars = {}
        for dim in dims:
            names = coord_candidates.get(dim, [dim])
            for name in names:
                if name in orig_ds.coords:
                    coord_vars[dim] = orig_ds.coords[name]
                    break

        if coord_vars:
            xr.Dataset(coords=coord_vars).to_zarr(store_path, mode="a", consolidated=False)

    if opened_orig_ds:
        orig_ds.close()

    if consolidate:
        zarr.consolidate_metadata(str(store))


def ensure_match_coords(ds1, ds2, round_decimals=6):
    for ds in [ds1, ds2]:
        ds['lat'] = np.round(ds['lat'], round_decimals)
        ds['lon'] = np.round(ds['lon'], round_decimals)
    lat_coords = np.intersect1d(ds1.lat.values, ds2.lat.values)
    lon_coords = np.intersect1d(ds1.lon.values, ds2.lon.values)
    ds1 = ds1.sel(lat=lat_coords, lon=lon_coords)
    ds2 = ds2.sel(lat=lat_coords, lon=lon_coords)
    return ds1, ds2

def load_heat(category='heatdays'):
    dss = {}
    for key in PERCENTILE_KEYS:
        da = xr.open_zarr(STORAGE_DIR + f'heatdays/{category}_{key}.zarr', consolidated=False)[category]
        dss[key] = da
    ds = xr.Dataset(dss)
    ds = ds.assign_coords(lat=np.round(ds.lat, 2), lon=np.round(ds.lon, 2))
    return ds

def open_hansen():
    ds = xr.open_zarr(HANSEN_DIR + 'hansen_tc_2000_2024_rechunked.zarr', consolidated=False)['tree_cover']
    return ds

@contextmanager
def local_cluster(n_workers=16, threads_per_worker=1, memory_limit='20GB', processes=True,
                  suppress_dask_warnings=False, use_dask_dashboard=True, dashboard_address=':7878'):

    if suppress_dask_warnings:
        warnings.filterwarnings(
            "ignore", 
            message="Sending large graph of size.*", 
            category=UserWarning, 
            module="distributed.client"
        )

    if use_dask_dashboard:
        with LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes,
                        memory_limit=memory_limit, dashboard_address=dashboard_address) as cluster:
            with Client(cluster) as client:
                print(f'LocalCluster started: {client.dashboard_link}')
                yield client
    else:
        yield None

        
def get_amazon_basin_mask(buffer_km=None):
    lst = xr.open_zarr(LST_DIR+'regridded_rechunked.zarr')['LST'][0]
    amazon_basin = gpd.read_file(BORDERS_DIR+'mrb_basins.shp')
    if buffer_km is not None:
        amazon_basin = amazon_basin.to_crs(epsg=3857).buffer(buffer_km * 1000).to_crs(epsg=4326)
    # create a mask for the Amazon basin using the hansen grid
    amazon_mask = regionmask.Regions([amazon_basin.geometry.iloc[0]]).mask(lst.lon, lst.lat)
    return {'raster': amazon_mask, 'vector': amazon_basin}

def get_target_lat_lons():
# load only first time chunk of tmax to use as a template for the rechunked hansen dataset
    tmax = xr.open_zarr(TMAX_RECHUNKED_PATH+'tmax_rechunked.zarr', 
                        chunks={'time': 1, 'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})
    lats = tmax.lat.values
    lons = tmax.lon.values
    return lats, lons



def clip_to_even_chunks(ds, lat_chunk_size=LAT_CHUNK_SIZE, lon_chunk_size=LON_CHUNK_SIZE):

    # ensure that lat and lon are both ascendung
    ds = ds.sortby('lat')
    ds = ds.sortby('lon')

    lat_full_chunks = ds['lat'].size // lat_chunk_size
    lon_full_chunks = ds['lon'].size // lon_chunk_size

    ds = ds.isel(lat=slice(0, lat_full_chunks * lat_chunk_size),
                  lon=slice(0, lon_full_chunks * lon_chunk_size))
    return ds

def get_tmax_template_coords(coord_dump_path=COORD_DUMP_PATH):
    if not os.path.exists(coord_dump_path):
        raise FileNotFoundError(f"Template coords file not found at {coord_dump_path}. Please run the script to generate it.")
        ds = xr.open_zarr(TMAX_RECHUNKED_FILE, consolidated=True)
        # make a template for various other scripts to regrid/rechunk to
        template = ds['Ta_TMAX'].isel(time=0).drop_vars('time')
        coords = template.coords
        # pickle the coords

        with open(coord_dump_path, 'wb') as f:
            pickle.dump(coords, f)

    with open(coord_dump_path, 'rb') as f:
        loaded_coords = pickle.load(f)

    da = xr.DataArray(np.zeros((len(loaded_coords['lat']), len(loaded_coords['lon']))),
                      coords={'lat': loaded_coords['lat'], 'lon': loaded_coords['lon']},
                      dims=('lat', 'lon'))
    return da

def _check_coords_match(das=[], warn_if_not_exact=True):
    """for a list of dataarrays, check that lat/lon coords match. Raises ValueError if not."""
    if len(das) < 2:
        return ValueError("At least two dataarrays are required to check coordinates.")
    combinations = itertools.combinations(das, 2)
    for da1, da2 in combinations:
        for dim in ['lat', 'lon']:
            if not np.allclose(da1[dim].values, da2[dim].values, atol=1e-6):
                raise ValueError(f"{dim} coordinates of {da1.name} and {da2.name} do not match.")
            if not np.array_equal(da1[dim].values, da2[dim].values) and warn_if_not_exact:
                warnings.warn(f"{dim} coordinates of {da1.name} and {da2.name} match but are not exactly equal")
