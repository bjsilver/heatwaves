#%%

import xarray as xr
import numpy as np
import os
import dask.array as da
from constants import HANSEN_DIR, MWA_DAILY_DIR, STORAGE_DIR, LAT_CHUNK_SIZE, LON_CHUNK_SIZE
import gc
from ufuncs import local_cluster
from calc_era5_percentiles import load_era5_t2m


COARSEN_FACTORS = [None, 2, 5, 10, 25]
SPATH_SUFFIX = None
USE_DASK_DASHBOARD = False
SEASON_MAP = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11]}

def _load_single_ind_var(ind_var_name):

    def __calc_tc_loss():
            tc = ds['tree_cover']
            ind_var = -(tc - tc[0])
            ind_var = ind_var.loc[{'time': slice('2003', '2020')}]
            ind_var = ind_var.rename({'time': 'year'})
            return ind_var
    if ind_var_name == 'cc_loss':
        ds = xr.open_zarr(os.path.join(HANSEN_DIR, 'hansen_tc_2000_2024_rechunked.zarr'))
        ind_var = __calc_tc_loss()
        ind_var.name = ind_var_name
        return ind_var
    
    if ind_var_name == 'cc_loss_era5grid':
        ds = xr.open_zarr(os.path.join(HANSEN_DIR, 'hansen_regridded.zarr'))
        ind_var = __calc_tc_loss()
        ind_var.name = ind_var_name
        return ind_var

    if ind_var_name == 'forest_loss_age':
        ds = xr.open_zarr(os.path.join(HANSEN_DIR, 'hansen_tc_2000_2024_loss_age.zarr'))
        var_name = 'forest_loss_age' if 'forest_loss_age' in ds.data_vars else list(ds.data_vars)[0]
        ind_var = ds[var_name]
        ind_var = ind_var.loc[{'time': slice('2003', '2020')}]
        ind_var = ind_var.rename({'time': 'year'})
        ind_var.name = ind_var_name
        return ind_var

    raise ValueError(f"Unsupported independent variable name: {ind_var_name}")


def load_ind_dep_data(ind_var_name, dep_var_name, driver_ind_var_name=None):

    if dep_var_name == 'tmax':
        dep_var = xr.open_zarr(os.path.join(STORAGE_DIR, 'rechunked', 'zhang_zhou_regridded_rechunked.zarr'))['Ta_TMAX']
        dep_var = dep_var.astype(np.float32).where(dep_var != -9999.0) / 10.0  # convert to °C
        dep_var.name = dep_var_name
    elif dep_var_name == 'heatwaves':
        dep_var = xr.open_zarr(os.path.join(STORAGE_DIR, 'heatdays', 'heatwaves_annual_sum_zz_2003-2020.zarr'))['heatwave_days']
        dep_var.name = dep_var_name
    elif dep_var_name == 'heatwaves_smoothed':
        dep_var = xr.open_zarr(os.path.join(STORAGE_DIR, 'heatdays', 'heatwaves_annual_sum_zz_2003-2020_smooth.zarr'))['heatwave_days']
        dep_var.name = dep_var_name
    elif dep_var_name == 'heatwaves_gapfilled':
        dep_var = xr.open_zarr(os.path.join(STORAGE_DIR, 'heatdays', 'heatwaves_annual_sum_zz_2003-2020_gapfilled.zarr'))['heatwave_days']
        dep_var.name = dep_var_name
    elif dep_var_name == 'heatwaves_era5-p95':
        dep_var = xr.open_zarr(os.path.join(STORAGE_DIR, 'heatdays', 'heatwaves_annual_sum_era5_2003-2020.zarr'))['heatwave_days']
        dep_var.name = dep_var_name

    elif dep_var_name == 'tmax_era5':
        dep_var = load_era5_t2m()
        dep_var.name = dep_var_name
    else:
        raise ValueError(f"Unsupported dependent variable name: {dep_var_name}")

    ind_var = _load_single_ind_var(ind_var_name).assign_coords(lat=dep_var.lat, lon=dep_var.lon)

    if driver_ind_var_name is None:
        driver_ind_var_name = ind_var_name
        driver_ind_var = ind_var
    elif driver_ind_var_name == ind_var_name:
        driver_ind_var = ind_var
    else:
        driver_ind_var = _load_single_ind_var(driver_ind_var_name).assign_coords(
            lat=dep_var.lat,
            lon=dep_var.lon,
        )

    return ind_var, driver_ind_var, dep_var



def _mwa_kernel(block, window_radius):
    """
    block: (2, Y, H+2wr, W+2wr)
      channel 0 = driver independent variable used for neighbor selection
      channel 1 = flattened (lat, lon) linear index grid
    Returns (2, Y, H, W): neighbor_index, valid_mask
    """

    driver_ind = block[0]
    linear_index_grid = block[1]
    wr = window_radius

    y_size = driver_ind.shape[0]
    h_size = driver_ind.shape[1] - 2 * wr
    w_size = driver_ind.shape[2] - 2 * wr

    out = np.empty((2, y_size, h_size, w_size), dtype=np.float32)
    out.fill(np.nan)

    center_driver = driver_ind[:, wr:wr+h_size, wr:wr+w_size]

    max_driver_diff = np.empty((y_size, h_size, w_size), dtype=np.float32)
    max_driver_diff.fill(-np.inf)
    neighbor_linear_selected = np.empty((y_size, h_size, w_size), dtype=np.float32)
    neighbor_linear_selected.fill(-1.0)
    has_lower = np.zeros((y_size, h_size, w_size), dtype=np.bool_)

    for di in range(-wr, wr + 1):
        for dj in range(-wr, wr + 1):
            if di == 0 and dj == 0:
                continue
            ri, rj = wr + di, wr + dj

            n_driver = driver_ind[:, ri:ri+h_size, rj:rj+w_size]
            n_linear = linear_index_grid[:, ri:ri+h_size, rj:rj+w_size]

            driver_diff = center_driver - n_driver
            lower = n_driver < center_driver
            candidate = lower & ~np.isnan(driver_diff)

            better = candidate & (driver_diff > max_driver_diff)
            max_driver_diff = np.where(better, driver_diff, max_driver_diff)
            neighbor_linear_selected = np.where(better, n_linear, neighbor_linear_selected)

            has_lower = has_lower | lower

    valid = (center_driver >= 0) & has_lower

    out[0] = np.where(valid, neighbor_linear_selected, -1.0)
    out[1] = valid.astype(np.float32)

    return out


def calculate_mwa_pairs(ind_var, driver_ind_var, window_radius=1, coarsen_factor=None):
    """Stage 1: calculate annual moving-window neighbour pairs from independent variables."""
    if "year" not in ind_var.dims or "year" not in driver_ind_var.dims:
        raise ValueError("ind_var and driver_ind_var must have dimensions (year, lat, lon)")

    ind_var = ind_var.astype(np.float32).chunk({"lat": LAT_CHUNK_SIZE, "lon": LON_CHUNK_SIZE})
    driver_ind_var = driver_ind_var.astype(np.float32).chunk({"lat": LAT_CHUNK_SIZE, "lon": LON_CHUNK_SIZE})

    if ind_var.sizes["lat"] != driver_ind_var.sizes["lat"] or ind_var.sizes["lon"] != driver_ind_var.sizes["lon"]:
        raise ValueError("ind_var and driver_ind_var must have identical spatial dimensions")

    n_lon = ind_var.sizes["lon"]
    linear_idx_2d = xr.DataArray(
        np.arange(ind_var.sizes["lat"] * n_lon, dtype=np.int32).reshape(ind_var.sizes["lat"], n_lon),
        dims=("lat", "lon"),
        coords={"lat": ind_var.coords["lat"], "lon": ind_var.coords["lon"]},
    )
    linear_idx_grid = linear_idx_2d.broadcast_like(ind_var).astype(np.float32)

    stacked = da.stack([
        driver_ind_var.data,
        linear_idx_grid.data,
    ])
    stacked = stacked.rechunk({
        0: 2,
        1: -1,
        2: LAT_CHUNK_SIZE,
        3: LON_CHUNK_SIZE,
    })

    result = da.map_overlap(
        lambda block: _mwa_kernel(block, window_radius),
        stacked,
        depth={0: 0, 1: 0, 2: window_radius, 3: window_radius},
        boundary=np.nan,
        trim=False,
        dtype=np.float32,
        chunks=((2,),) + stacked.chunks[1:],
    )

    valid = result[1] > 0.5
    neighbour_index = da.where(valid, result[0], -1.0).astype(np.int32)

    mwa_pairs = xr.Dataset(
        data_vars={
            "neighbour_index": (("year", "lat", "lon"), neighbour_index),
            "valid": (("year", "lat", "lon"), valid),
        },
        coords={
            "year": ind_var.coords["year"],
            "lat": ind_var.coords["lat"],
            "lon": ind_var.coords["lon"],
        },
        attrs={
            "window_radius": window_radius,
            "coarsen_factor": "" if coarsen_factor is None else coarsen_factor,
        },
    )

    return mwa_pairs


def _year_positions(mwa_pairs, dep_var):
    """Return lazy integer positions into mwa_pairs.year for each dep timestep."""
    pair_years = np.asarray(mwa_pairs.year.values)
    if "time" in dep_var.dims:
        dep_years = np.asarray(dep_var.time.dt.year.values)
    elif "year" in dep_var.dims:
        dep_years = np.asarray(dep_var.year.values)
    else:
        raise ValueError("dep_var must have either a 'time' or 'year' dimension")

    pos = np.searchsorted(pair_years, dep_years)
    in_bounds = (pos >= 0) & (pos < len(pair_years))
    exact = np.zeros_like(in_bounds, dtype=bool)
    exact[in_bounds] = pair_years[pos[in_bounds]] == dep_years[in_bounds]
    if not np.all(exact):
        missing_years = np.unique(dep_years[~exact])
        raise ValueError(f"Dependent variable contains years not present in mwa_pairs: {missing_years}")

    chunk_0 = dep_var.data.chunks[0]
    return da.from_array(pos.astype(np.int16), chunks=chunk_0)


def _gather_from_pairs(mwa_pairs, var):
    """Gather center/neighbor values from var using annual pair map, mapped per var timestep."""
    if "year" not in mwa_pairs.dims:
        raise ValueError("mwa_pairs must have dimensions (year, lat, lon)")

    dep_dim = "time" if "time" in var.dims else "year"
    if var.sizes["lat"] != mwa_pairs.sizes["lat"] or var.sizes["lon"] != mwa_pairs.sizes["lon"]:
        raise ValueError("mwa_pairs and var must have identical spatial dimensions")

    year_pos = _year_positions(mwa_pairs, var)

    var_arr = var.astype(np.float32).data
    center = da.take(var_arr, year_pos, axis=0)

    neighbor_index = da.take(mwa_pairs["neighbour_index"].data, year_pos, axis=0)
    valid = da.take(mwa_pairs["valid"].data, year_pos, axis=0)

    center_flat = center.reshape((center.shape[0], center.shape[1] * center.shape[2]))
    center_flat = center_flat.rechunk({
        0: center.chunks[0],
        1: -1,
    })

    safe_neighbor_index = da.where(neighbor_index < 0, 0, neighbor_index).astype(np.int64)

    def _gather_numpy(center_block, index_block):
        """Perform neighbour lookup inside each Dask chunk."""
        t = center_block.shape[0]
        lat = index_block.shape[1]
        lon = index_block.shape[2]

        out = np.empty((t, lat, lon), dtype=np.float32)
        for ti in range(t):
            out[ti] = center_block[ti, index_block[ti]]
        return out

    neighbor_raw = da.map_blocks(
        _gather_numpy,
        center_flat,
        safe_neighbor_index,
        dtype=np.float32,
        chunks=safe_neighbor_index.chunks,
    )

    gather_valid = valid & ~da.isnan(center)
    center_masked = da.where(gather_valid, center, np.nan).astype(np.float32)
    neighbor_masked = da.where(gather_valid, neighbor_raw, np.nan).astype(np.float32)

    return dep_dim, center_masked, neighbor_masked


def apply_mwa_pairs(mwa_pairs, dep_var):
    """Stage 2: apply annual neighbour pairs to dependent-variable data."""
    dep_var = dep_var.astype(np.float32).chunk({"lat": LAT_CHUNK_SIZE, "lon": LON_CHUNK_SIZE})
    dep_dim, dep_center, dep_neighbor = _gather_from_pairs(mwa_pairs, dep_var)

    return xr.Dataset(
        data_vars={
            "dep_center": ((dep_dim, "lat", "lon"), dep_center),
            "dep_neighbor": ((dep_dim, "lat", "lon"), dep_neighbor),
        },
        coords={
            dep_dim: dep_var.coords[dep_dim],
            "lat": dep_var.coords["lat"],
            "lon": dep_var.coords["lon"],
        },
    )

def moving_window_analysis(ds, window_radius, ind_var_name, dep_var_name,
                           driver_ind_var_name=None, dep_var_agg=None,
                           time_scale='annual', year_index=None):
    """Compatibility wrapper around two-stage MWA workflow."""
    if driver_ind_var_name is None:
        driver_ind_var_name = ind_var_name

    if ind_var_name not in ds:
        raise ValueError(f"Independent variable '{ind_var_name}' is missing from dataset")
    if driver_ind_var_name not in ds:
        raise ValueError(f"Driver variable '{driver_ind_var_name}' is missing from dataset")

    mwa_pairs = calculate_mwa_pairs(
        ind_var=ds[ind_var_name],
        driver_ind_var=ds[driver_ind_var_name],
        window_radius=window_radius,
    )


    dep_pairs = apply_mwa_pairs(mwa_pairs, ds[dep_var_name])
    _, ind_center_raw, ind_neighbor_raw = _gather_from_pairs(mwa_pairs, ds[ind_var_name])

    time_dim = "time" if "time" in ds[dep_var_name].dims else "year"
    dep_valid = ~da.isnan(dep_pairs["dep_center"].data)
    data_vars = {
        f'{ind_var_name}_center':   ((time_dim, "lat", "lon"), da.where(dep_valid, ind_center_raw, np.nan).astype(np.float32)),
        f'{ind_var_name}_neighbor': ((time_dim, "lat", "lon"), da.where(dep_valid, ind_neighbor_raw, np.nan).astype(np.float32)),
        f'{dep_var_name}_center':   ((time_dim, "lat", "lon"), dep_pairs["dep_center"].data),
        f'{dep_var_name}_neighbor': ((time_dim, "lat", "lon"), dep_pairs["dep_neighbor"].data),
    }

    result_ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            time_dim: ds[dep_var_name].coords[time_dim],
            "lat": ds.coords["lat"],
            "lon": ds.coords["lon"],
        },
        attrs={
            "driver_ind_var_name": driver_ind_var_name,
            "ind_var_name": ind_var_name,
            "window_radius": window_radius,
            "dep_var_agg": dep_var_agg if dep_var_agg is not None else "",
        },
    )

    return result_ds


def _generate_coarsened_versions(ind_var, driver_ind_var, coarsen_factor):
    """Generate coarsened independent-variable pairs, with offset versions if coarsen_factor >= 10.

    Returns a list of (ind_var_coarse, driver_ind_var_coarse) tuples.
    For coarsen_factor >= 10, creates offset versions with 5-pixel intervals.
    """
    dss = []

    # First version: standard coarsening
    ind_coarse = ind_var.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary='trim').mean()
    driver_ind_coarse = driver_ind_var.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary='trim').mean()
    # rechunk for the coarsened versions
    ind_coarse = ind_coarse.chunk({'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})
    driver_ind_coarse = driver_ind_coarse.chunk({'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})
    dss.append((ind_coarse, driver_ind_coarse))

    # Add offset versions if coarsen_factor >= 10
    if coarsen_factor >= 10:
        num_offsets = (coarsen_factor - 10) // 5 + 1
        for offset_idx in range(num_offsets):
            offset = (offset_idx + 1) * 5

            # Create shifted versions with offset slicing
            ind_shifted = ind_var.loc[
                {
                    'lat': slice(ind_var.lat.values[offset], ind_var.lat.values[-offset]),
                    'lon': slice(ind_var.lon.values[offset], ind_var.lon.values[-offset]),
                }
            ]
            driver_ind_shifted = driver_ind_var.loc[
                {
                    'lat': slice(driver_ind_var.lat.values[offset], driver_ind_var.lat.values[-offset]),
                    'lon': slice(driver_ind_var.lon.values[offset], driver_ind_var.lon.values[-offset]),
                }
            ]
            # Coarsen the shifted versions
            ind_shifted_coarse = ind_shifted.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary='trim').mean()
            driver_ind_shifted_coarse = driver_ind_shifted.coarsen(
                lat=coarsen_factor,
                lon=coarsen_factor,
                boundary='trim',
            ).mean()

            # rechunk for the coarsened versions
            ind_shifted_coarse = ind_shifted_coarse.chunk({'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})
            driver_ind_shifted_coarse = driver_ind_shifted_coarse.chunk({'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})

            dss.append((ind_shifted_coarse, driver_ind_shifted_coarse))

    return dss


def calc_moving_window(ind_var, driver_ind_var, dep_var, client=None, dep_var_agg='annual_mean',
                       coarsen_factor=None, window_radius=1, sname=None,
                       driver_ind_var_name=None, time_scale='annual'):

    ind_var_name = ind_var.name
    dep_var_name = dep_var.name

    if driver_ind_var_name is None:
        driver_ind_var_name = ind_var_name

    if len(dep_var.lat) == 5000 and len(dep_var.lon) == 4750:    
        elev = xr.open_dataset(os.path.join(STORAGE_DIR, '..', 'elevation', 'elevation_regridded.nc'))['elevation']
        ref_ind_var = ind_var
        elev = elev.assign_coords(lat=ref_ind_var.lat, lon=ref_ind_var.lon)
        ind_var = ind_var.where(elev < 1000)
        driver_ind_var = driver_ind_var.where(elev < 1000)
        dep_var = dep_var.where(elev < 1000)

    # Generate list of variable triplets (with offsets if applicable)
    if coarsen_factor:
        dss = _generate_coarsened_versions(ind_var, driver_ind_var, coarsen_factor)
    else:
        dss = [(ind_var, driver_ind_var)]

    os.makedirs(MWA_DAILY_DIR, exist_ok=True)

    # Process each version
    dep_var_agg_all = aggregate_dependent_variable(dep_var, dep_var_agg, time_scale=time_scale)

    for version_idx, (ind_var_version, driver_ind_var_version) in enumerate(dss):
        # Add version suffix to output paths if there are multiple versions
        if len(dss) > 1:
            out_path = os.path.join(MWA_DAILY_DIR, f'{sname}_mwa_results_{version_idx}.zarr')
        else:
            out_path = os.path.join(MWA_DAILY_DIR, f'{sname}_mwa_results.zarr')

        ind_var_version = ind_var_version.astype(np.float32).chunk({'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})
        driver_ind_var_version = driver_ind_var_version.astype(np.float32).chunk({'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})

        # Keep dependent-variable values at native resolution; only select/interpolate to pair grid.
        dep_var_version = dep_var_agg_all.astype(np.float32).interp(
            lat=ind_var_version.lat,
            lon=ind_var_version.lon,
            method='nearest',
        ).chunk({'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})

        pair_base = f'{ind_var_name}_driver_{driver_ind_var_name}'
        pair_base += f"_lat{ind_var_version.sizes['lat']}_lon{ind_var_version.sizes['lon']}"
        if coarsen_factor:
            pair_base += f'_coarsened{coarsen_factor}'
        pair_base += f'_wr{window_radius}'
        if SPATH_SUFFIX:
            pair_base += f'_{SPATH_SUFFIX}'

        if len(dss) > 1:
            pair_out_path = os.path.join(MWA_DAILY_DIR, f'{pair_base}_mwa_pairs_{version_idx}.zarr')
        else:
            pair_out_path = os.path.join(MWA_DAILY_DIR, f'{pair_base}_mwa_pairs.zarr')

        if os.path.exists(pair_out_path):
            mwa_pairs = xr.open_zarr(pair_out_path, consolidated=True)
            print(f"Reusing neighbour pairs from {pair_out_path}")
        else:
            mwa_pairs = calculate_mwa_pairs(
                ind_var=ind_var_version,
                driver_ind_var=driver_ind_var_version,
                window_radius=window_radius,
                coarsen_factor=coarsen_factor,
            )
            mwa_pairs = mwa_pairs.chunk({'year': 1, 'lat': LAT_CHUNK_SIZE, 'lon': LON_CHUNK_SIZE})
            mwa_pairs.to_zarr(pair_out_path, mode='w', consolidated=True)
            print(f"Neighbour pairs written to {pair_out_path}")

        dep_pairs = apply_mwa_pairs(mwa_pairs, dep_var_version)
        _, ind_center_raw, ind_neighbor_raw = _gather_from_pairs(mwa_pairs, ind_var_version)

        dep_dim = "time" if "time" in dep_var_version.dims else "year"
        dep_valid = ~da.isnan(dep_pairs["dep_center"].data)
        result_ds = xr.Dataset(
            data_vars={
                f'{ind_var_name}_center': ((dep_dim, "lat", "lon"), da.where(dep_valid, ind_center_raw, np.nan).astype(np.float32)),
                f'{ind_var_name}_neighbor': ((dep_dim, "lat", "lon"), da.where(dep_valid, ind_neighbor_raw, np.nan).astype(np.float32)),
                f'{dep_var_name}_center': ((dep_dim, "lat", "lon"), dep_pairs["dep_center"].data),
                f'{dep_var_name}_neighbor': ((dep_dim, "lat", "lon"), dep_pairs["dep_neighbor"].data),
            },
            coords={
                dep_dim: dep_var_version.coords[dep_dim],
                "lat": dep_var_version.coords["lat"],
                "lon": dep_var_version.coords["lon"],
            },
            attrs={
                "driver_ind_var_name": driver_ind_var_name,
                "ind_var_name": ind_var_name,
                "window_radius": window_radius,
                "dep_var_agg": dep_var_agg if dep_var_agg is not None else "",
            },
        )

        result_ds = result_ds.chunk({
            dep_dim: 1,
            "lat": LAT_CHUNK_SIZE,
            "lon": LON_CHUNK_SIZE,
        })

        result_ds.to_zarr(out_path, mode='w', consolidated=True)

        if client is not None:
            client.cancel(result_ds)
        gc.collect()
        print(f"All-year results written to {out_path}")




def aggregate_dependent_variable(dep_var, aggregation, time_scale='annual'):
    if time_scale == 'daily':
        return dep_var

    if 'annual' in aggregation and 'year' in dep_var.dims:
        return dep_var
    match aggregation:
        case "annual_mean":
            annual = dep_var.groupby('time.year').mean()
        case "annual_sum":
            annual = dep_var.groupby('time.year').sum()
        case "annual_95pct":
            annual = dep_var.groupby('time.year').quantile(0.95)
        case s if s in ["DJF_mean", "MAM_mean", "JJA_mean", "SON_mean"]:
            season = s.split("_")[0]
            months = SEASON_MAP[season]
            annual = dep_var.where(dep_var['time.month'].isin(months)).groupby('time.year').mean()
        case _:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
    


    return annual



def _check_zarrs_exist(sname):
    """Check if output zarrs exist (either indexed or non-indexed versions)."""
    # Check for indexed versions (from offset resampling)
    indexed_out = os.path.join(MWA_DAILY_DIR, f'{sname}_mwa_results_0.zarr')
    if os.path.exists(indexed_out):
        return True

    # Check for non-indexed versions
    out_path = os.path.join(MWA_DAILY_DIR, f'{sname}_mwa_results.zarr')
    return os.path.exists(out_path)


def run_mwa(coarsen_factor=None, window_radius=1, dep_var_agg='annual_mean', client=None,
            ind_var_name='cc_loss', driver_ind_var_name=None, dep_var_name='heatwaves',
            time_scale='annual'):

    if driver_ind_var_name is None:
        driver_ind_var_name = ind_var_name

    ind_var, driver_ind_var, dep_var = load_ind_dep_data(
        ind_var_name=ind_var_name,
        dep_var_name=dep_var_name,
        driver_ind_var_name=driver_ind_var_name,
    )

    # check if output zarr already exists
    sname = f'{dep_var.name}_vs_{ind_var_name}_driver_{driver_ind_var_name}_{dep_var_agg}'
    if coarsen_factor:
        sname += f'_coarsened{coarsen_factor}'
    if time_scale != "annual":
        sname += f"_{time_scale}"
    if SPATH_SUFFIX:
        sname += f'_{SPATH_SUFFIX}'

    if not _check_zarrs_exist(sname):
        print(f"Output zarrs do not exist for {sname}. Proceeding with moving window analysis.")
        calc_moving_window(ind_var=ind_var, driver_ind_var=driver_ind_var, dep_var=dep_var, client=client, window_radius=window_radius,
                           coarsen_factor=coarsen_factor, sname=sname, dep_var_agg=dep_var_agg,
                           driver_ind_var_name=driver_ind_var_name, time_scale=time_scale)
        print(
            f"Moving window analysis completed for {dep_var.name} vs {ind_var_name} "
            f"(driver={driver_ind_var_name}) with coarsen factor {coarsen_factor}."
        )
    else:
        print(f"Output zarrs already exist for {sname}. Skipping moving window analysis.")


def run_mwa_coarsen_loop(ind_var_name, driver_ind_var_name,
                         dep_var_name, dep_var_agg='annual_mean', time_scale='annual',
                         client=None):
    for coarsen_factor in COARSEN_FACTORS:
        run_mwa(coarsen_factor=coarsen_factor, window_radius=1, dep_var_agg=dep_var_agg, client=client,
                ind_var_name=ind_var_name, driver_ind_var_name=driver_ind_var_name,
                dep_var_name=dep_var_name, time_scale=time_scale)
            
#%%
if __name__ == "__main__":
    
    with local_cluster(n_workers=40, threads_per_worker=1, memory_limit='12GB',
                       suppress_dask_warnings=False, use_dask_dashboard=USE_DASK_DASHBOARD) as client:
        # run_mwa_coarsen_loop(ind_var_name='cc_loss', driver_ind_var_name='cc_loss',
        #                      dep_var_name='tmax', dep_var_agg='annual_mean', client=client)
        # run_mwa_coarsen_loop(ind_var_name='forest_loss_age', driver_ind_var_name='forest_loss_age',
        #                     dep_var_name='tmax', dep_var_agg='annual_mean', client=client)
        # run_mwa_coarsen_loop(ind_var_name='cc_loss', driver_ind_var_name='cc_loss',
        #                      dep_var_name='heatwaves_gapfilled', dep_var_agg='annual_sum', client=client)

        run_mwa_coarsen_loop(ind_var_name='cc_loss', driver_ind_var_name='cc_loss',
                                    dep_var_name='tmax', time_scale='daily', client=client)