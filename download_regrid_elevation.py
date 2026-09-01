#%%
import os
from pathlib import Path
import numpy as np
import xarray as xr
from ufuncs import load_heat
from constants import STORAGE_DIR
import xesmf as xe


# NOAA GLOBE tile bounds (lon_min, lon_max, lat_min, lat_max).
_TILE_BOUNDS = {
    "A": (-180, -90,  50,  90),
    "B": ( -90,   0,  50,  90),
    "C": (   0,  90,  50,  90),
    "D": (  90, 180,  50,  90),

    "E": (-180, -90,   0,  50),
    "F": ( -90,   0,   0,  50),
    "G": (   0,  90,   0,  50),
    "H": (  90, 180,   0,  50),

    "I": (-180, -90, -50,   0),
    "J": ( -90,   0, -50,   0),
    "K": (   0,  90, -50,   0),
    "L": (  90, 180, -50,   0),

    "M": (-180, -90, -90, -50),
    "N": ( -90,   0, -90, -50),
    "O": (   0,  90, -90, -50),
    "P": (  90, 180, -90, -50),
}

def open_globe_tile(
    file_path,
    tile_letter=None,
    cols=10800,
    dtype="<i2",
    missing_below=-500,
    name="elevation",
):
    """Open a NOAA GLOBE binary tile as a lat/lon xarray.DataArray.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to a tile file such as 'a10g' or 'f10g'.
    tile_letter : str, optional
        Tile letter A-P. If omitted, inferred from the first character of file name.
    cols : int, default 10800
        Number of columns in the tile.
    dtype : str or numpy.dtype, default '<i2'
        Binary dtype for the input tile values.
    missing_below : int or float, default -500
        Values below or equal to this threshold are converted to NaN.
    name : str, default 'elevation'
        Name of the returned DataArray.

    Returns
    -------
    xarray.DataArray
        Elevation tile with dimensions ('lat', 'lon') and pixel-center coordinates.
    """
    path = Path(file_path)

    if tile_letter is None:
        if not path.name:
            raise ValueError("Could not infer tile letter from an empty file name.")
        tile_letter = path.name[0]

    tile_letter = str(tile_letter).upper()
    if tile_letter not in _TILE_BOUNDS:
        valid = ", ".join(sorted(_TILE_BOUNDS))
        raise ValueError(f"Unknown tile letter '{tile_letter}'. Expected one of: {valid}")

    lon_min, lon_max, lat_min, lat_max = _TILE_BOUNDS[tile_letter]

    values = np.fromfile(path, dtype=dtype)
    if values.size % cols != 0:
        raise ValueError(
            f"Tile has {values.size} values, which cannot be reshaped with cols={cols}."
        )

    rows = values.size // cols
    data = values.reshape(rows, cols).astype(np.float32)
    data[data <= missing_below] = np.nan

    # Build pixel-center coordinates from bounds and shape.
    lon_step = (lon_max - lon_min) / cols
    lat_step = (lat_max - lat_min) / rows
    lons = lon_min + (np.arange(cols) + 0.5) * lon_step
    lats = lat_max - (np.arange(rows) + 0.5) * lat_step

    da = xr.DataArray(
        data,
        dims=("lat", "lon"),
        coords={"lat": lats, "lon": lons},
        name=name,
        attrs={
            "units": "m",
            "long_name": "Global Land One-km Base Elevation",
            "source": "NOAA GLOBE",
            "tile": tile_letter,
            "lon_min": lon_min,
            "lon_max": lon_max,
            "lat_min": lat_min,
            "lat_max": lat_max,
        },
    )

    da = da.sortby("lat", ascending=True)  # ensure lat is ascending for consistency

    return da


#%%

da1 = open_globe_tile(Path(STORAGE_DIR) / "../elevation/f10g")
da2 = open_globe_tile(Path(STORAGE_DIR) / "../elevation/j10g")
da = xr.concat([da2, da1], dim="lat")

template = load_heat('heatwaves')

# slice da by template limits
da = da.sel(lat=slice(template.lat.min(), template.lat.max()),
             lon=slice(template.lon.min(), template.lon.max()))

# create and save regridder weights
regridder = xe.Regridder(da, template, "bilinear", reuse_weights=False)
regridder.to_netcdf(STORAGE_DIR+"../elevation/elevation_regrid_weights.nc")

#%%
if os.path.exists(STORAGE_DIR+"../elevation/elevation_regrid_weights.nc"):
    regridder = xe.Regridder(da, template, "bilinear", weights=STORAGE_DIR+"../elevation/elevation_regrid_weights.nc")

da_regridded = regridder(da)
da_regridded.name = "elevation"
da_regridded.attrs["source"] = "NOAA GLOBE (regridded to match heatwave template)"
da_regridded.to_netcdf(Path(STORAGE_DIR) / "elevation_regridded.nc")


#%%
# fig = plt.figure(figsize=(12, 12))
# map_ax = fig.add_subplot(projection=ccrs.PlateCarree())
# map_ax.add_feature(cf.COASTLINE, linewidth=10)
# da1[1000:2000, 1000:2000].plot(ax=map_ax, cmap="terrain", cbar_kwargs={"label": "Elevation (m)"},
#          vmax=1000, vmin=-10)