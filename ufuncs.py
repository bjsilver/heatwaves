import zarr
from pathlib import Path
import json
import shutil
import xarray as xr
import numpy as np

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


def ensure_match_coords(ds1, ds2, round_decimals=2):
    for ds in [ds1, ds2]:
        ds['lat'] = np.round(ds['lat'], round_decimals)
        ds['lon'] = np.round(ds['lon'], round_decimals)
    lat_coords = np.intersect1d(ds1.lat.values, ds2.lat.values)
    lon_coords = np.intersect1d(ds1.lon.values, ds2.lon.values)
    ds1 = ds1.sel(lat=lat_coords, lon=lon_coords)
    ds2 = ds2.sel(lat=lat_coords, lon=lon_coords)
    return ds1, ds2