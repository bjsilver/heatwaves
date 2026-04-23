#%%
import glob
import json
import math
import os
import re
import shutil
import time

import dask
import dask.array as da
import xarray as xr
from dask.distributed import Client, LocalCluster
from dask.utils import format_bytes

# This must be set before Dask/Distributed are imported.
os.environ["MALLOC_TRIM_THRESHOLD_"] = "65536"

dask.config.set({
    # Let workers retain more in-memory chunks before spilling.
    "distributed.worker.memory.target": 0.75,
    "distributed.worker.memory.pause": 0.90,
    "distributed.worker.memory.spill": 0.85,
})


# ==========================================
# 1. CONFIGURATION
# ==========================================
NC_SOURCE_PATTERN = "/uolstore/Research/b/b0256/Users/earjba/ZhangandZhou/tropical_regions/*TMAX_daily_tropics30N30S.nc"

WORK_DIR = "/uolstore/Research/b/b0256/Users/earbsi/heatwaves/rechunked/"
ANNUAL_STORE = os.path.join(WORK_DIR, "tmax_annual_chunks.zarr")
PROGRESS_JSON = os.path.join(WORK_DIR, "tmax_annual_chunks_progress.json")

START_DATE = "2003-01-01"

# This rewrite only creates annual time chunks.
TIME_CHUNK = 365
# 256 is a safer default for downstream percentile tiling, which loads
# year*doy*lat*lon tiles eagerly and can prefetch multiple tiles.
SPATIAL_CHUNK = 256

# Keep int16 output for fastest writes and smallest network I/O.
CAST_TO_FLOAT32 = False

# Resume behavior.
RESET_OUTPUT = True

# Validate the assumption that each source file is one 365-day year.
STRICT_365_DAYS = True

# Dask cluster sizing for I/O-heavy NFS writes.
# Increase concurrency to keep more chunk I/O in flight.
N_WORKERS = 10
THREADS_PER_WORKER = 1
WORKER_MEMORY_LIMIT = "12GB"


def extract_year(path):
    """Extract a year from the filename for logging and sorting checks."""
    match = re.search(r"(19|20)\d{2}", os.path.basename(path))
    return int(match.group(0)) if match else None


def preprocess_standardize(ds):
    """Normalize each yearly file to a common schema."""
    ds = ds.drop_vars(["spatial_ref"], errors="ignore")
    if "DOY" in ds.dims:
        ds = ds.rename({"DOY": "time"})
    return ds


def open_year(path):
    """Open one yearly file with aligned chunks."""
    ds = xr.open_dataset(
        path,
        decode_cf=False,
        mask_and_scale=False,
        chunks={"DOY": TIME_CHUNK, "lat": SPATIAL_CHUNK, "lon": SPATIAL_CHUNK},
    )
    ds = preprocess_standardize(ds)
    if CAST_TO_FLOAT32:
        ds = ds.astype("float32")

    # NetCDF inputs can carry CF packing fields in both attrs and encoding
    # (e.g. add_offset/scale_factor), which causes to_zarr encoding conflicts.
    # Keep attrs, but clear auto-propagated serialization encoding.
    for name in ds.variables:
        ds[name].encoding = {}

    # Remove serialization attrs from data variables to avoid clashes with the
    # backend encoder during incremental region writes.
    for name in ds.data_vars:
        for key in ("add_offset", "scale_factor", "_FillValue"):
            ds[name].attrs.pop(key, None)
        # spatial_ref is dropped above; avoid dangling references in attrs.
        ds[name].attrs.pop("grid_mapping", None)

    return ds


def discover_files():
    """Return sorted yearly source files."""
    files = sorted(glob.glob(NC_SOURCE_PATTERN))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {NC_SOURCE_PATTERN}")
    return files


def inspect_files(files):
    """Collect per-file timing metadata needed for region writes."""
    records = []
    for path in files:
        with xr.open_dataset(path, decode_cf=False, mask_and_scale=False) as ds:
            n_time = int(ds.sizes["DOY"] if "DOY" in ds.sizes else ds.sizes["time"])
        records.append({
            "path": path,
            "name": os.path.basename(path),
            "year": extract_year(path),
            "n_time": n_time,
        })
    return records


def validate_records(records):
    """Fail fast if files are not laid out as expected."""
    years = [r["year"] for r in records if r["year"] is not None]
    if years and years != sorted(years):
        raise ValueError("Input files are not in chronological order after sorting.")

    if STRICT_365_DAYS:
        bad = [r for r in records if r["n_time"] != TIME_CHUNK]
        if bad:
            details = ", ".join(f"{r['name']} ({r['n_time']} days)" for r in bad)
            raise ValueError(
                "STRICT_365_DAYS=True but some files are not 365 days: " + details
            )


def estimate_total_bytes(example_ds, total_time):
    """Estimate logical output size for throughput expectations."""
    total_bytes = 0
    for data_array in example_ds.data_vars.values():
        if "time" not in data_array.dims:
            continue
        shape = [total_time if dim == "time" else data_array.sizes[dim] for dim in data_array.dims]
        total_bytes += math.prod(shape) * data_array.dtype.itemsize
    return total_bytes


def print_runtime_estimate(total_bytes):
    """Print rough elapsed time expectations for direct 2x I/O path."""
    total_io_bytes = total_bytes * 2
    print(f"Logical output size: {format_bytes(total_bytes)}")
    print(f"Estimated total I/O (read+write): {format_bytes(total_io_bytes)}")
    for mib_s in (50, 100, 200, 400):
        bps = mib_s * 1024 * 1024
        hours = (total_io_bytes / bps) / 3600
        print(f"At about {mib_s} MiB/s sustained: about {hours:.1f} hours")


def print_analysis_chunk_feasibility(total_time, spatial_chunk):
    """Report memory needed for one full-time spatial chunk in downstream analysis."""
    n_items = total_time * spatial_chunk * spatial_chunk
    bytes_int16 = n_items * 2
    bytes_float32 = n_items * 4
    bytes_bool = n_items * 1

    print("Downstream full-time chunk memory feasibility")
    print(f"  Time length: {total_time} days")
    print(f"  Spatial chunk: {spatial_chunk}x{spatial_chunk}")
    print(f"  One int16 data chunk:   {format_bytes(bytes_int16)}")
    print(f"  One float32 data chunk: {format_bytes(bytes_float32)}")
    print(f"  One bool mask chunk:    {format_bytes(bytes_bool)}")

    # Typical working-set examples for analysis code.
    # 1) one float32 array + one bool mask + one float32 output
    working_set_basic = bytes_float32 + bytes_bool + bytes_float32
    # 2) two float32 arrays + one bool mask + one float32 output
    working_set_heavier = bytes_float32 + bytes_float32 + bytes_bool + bytes_float32
    print(f"  Typical basic working set (~2x f32 + mask): {format_bytes(working_set_basic)}")
    print(f"  Typical heavier working set (~3x f32 + mask): {format_bytes(working_set_heavier)}")

    # calculate_tmax_percentiles.py uses ThreadPool prefetch with defaults:
    # reader_workers=3, prefetch_depth=9. Each payload contains one full
    # int16 tile across all time steps, so queued payloads dominate RAM.
    prefetched_tiles = 9
    percentile_prefetch_bytes = prefetched_tiles * bytes_int16
    percentile_compute_bytes = bytes_int16 + (365 * spatial_chunk * spatial_chunk * 8)
    print("  Percentile-step estimate (from calculate_tmax_percentiles.py defaults):")
    print(f"    Prefetch queue (~{prefetched_tiles} int16 tiles): {format_bytes(percentile_prefetch_bytes)}")
    print(f"    One active tile compute (int16 + float64 out): {format_bytes(percentile_compute_bytes)}")


def build_time_axis(total_time):
    """Create one synthetic no-leap daily time axis."""
    return xr.date_range(
        start=START_DATE,
        periods=total_time,
        freq="D",
        calendar="noleap",
        use_cftime=True,
    )


def build_skeleton(example_ds, total_time, time_axis):
    """Create an empty output dataset with final shape/chunk metadata."""
    data_vars = {}
    write_var_names = []

    for name, data_array in example_ds.data_vars.items():
        if "time" not in data_array.dims:
            continue

        shape = []
        chunks = []
        for dim in data_array.dims:
            if dim == "time":
                shape.append(total_time)
                chunks.append(TIME_CHUNK)
            elif dim in ("lat", "lon"):
                shape.append(data_array.sizes[dim])
                chunks.append(SPATIAL_CHUNK)
            else:
                shape.append(data_array.sizes[dim])
                chunks.append(data_array.sizes[dim])

        empty_data = da.zeros(tuple(shape), chunks=tuple(chunks), dtype=data_array.dtype)
        data_vars[name] = xr.DataArray(
            empty_data,
            dims=data_array.dims,
            attrs=data_array.attrs,
        )
        write_var_names.append(name)

    if not write_var_names:
        raise ValueError("No time-varying data variables found to write.")

    coords = {
        "time": time_axis,
    }
    if "lat" in example_ds.coords:
        coords["lat"] = example_ds["lat"]
    if "lon" in example_ds.coords:
        coords["lon"] = example_ds["lon"]

    out_ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=example_ds.attrs)
    return out_ds, write_var_names


def build_offsets(records):
    """Build cumulative time slices for each file's write region."""
    offsets = []
    cursor = 0
    for record in records:
        start = cursor
        end = cursor + record["n_time"]
        offsets.append((start, end))
        cursor = end
    return offsets


def reset_outputs():
    """Remove old output and progress files for a clean run."""
    if os.path.isdir(ANNUAL_STORE):
        print(f"Removing old store: {ANNUAL_STORE}")
        shutil.rmtree(ANNUAL_STORE)
    elif os.path.exists(ANNUAL_STORE):
        os.remove(ANNUAL_STORE)

    if os.path.exists(PROGRESS_JSON):
        print(f"Removing old progress file: {PROGRESS_JSON}")
        os.remove(PROGRESS_JSON)


def load_progress():
    """Load completed-file state for resume support."""
    if not os.path.exists(PROGRESS_JSON):
        return {"completed": []}
    with open(PROGRESS_JSON, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_progress(progress):
    """Persist completed-file state after each successful year write."""
    with open(PROGRESS_JSON, "w", encoding="utf-8") as handle:
        json.dump(progress, handle, indent=2)


def finalize_metadata():
    """Consolidate Zarr metadata once at the end for fast reads."""
    ds = xr.open_zarr(ANNUAL_STORE, consolidated=False)
    ds.to_zarr(ANNUAL_STORE, mode="a", consolidated=True)


def main():
    files = discover_files()
    records = inspect_files(files)
    validate_records(records)

    print(f"Found {len(records)} files")
    print(f"First file: {records[0]['name']}")
    print(f"Last file:  {records[-1]['name']}")

    total_time = sum(r["n_time"] for r in records)
    offsets = build_offsets(records)

    print_analysis_chunk_feasibility(total_time, SPATIAL_CHUNK)

    os.makedirs(WORK_DIR, exist_ok=True)
    if RESET_OUTPUT:
        reset_outputs()

    with open_year(records[0]["path"]) as example_ds:

        time_axis = build_time_axis(total_time)
        total_bytes = estimate_total_bytes(example_ds, total_time)
        print_runtime_estimate(total_bytes)

        if not os.path.isdir(ANNUAL_STORE):
            skeleton, write_var_names = build_skeleton(example_ds, total_time, time_axis)
            print("Creating Zarr skeleton")
            # Metadata is written immediately; we intentionally do not compute
            # the zero-valued skeleton data graph.
            skeleton.to_zarr(ANNUAL_STORE, mode="w", consolidated=False, compute=False)
        else:
            write_var_names = [
                name for name, data_array in example_ds.data_vars.items() if "time" in data_array.dims
            ]

    progress = load_progress()
    completed = set(progress.get("completed", []))

    with LocalCluster(
        n_workers=N_WORKERS,
        threads_per_worker=THREADS_PER_WORKER,
        memory_limit=WORKER_MEMORY_LIMIT,
        dashboard_address=":8787",
    ) as cluster, Client(cluster) as client:
        print(f"Dask dashboard: {client.dashboard_link}")

        for record, (start, end) in zip(records, offsets):
            if record["name"] in completed:
                print(f"Skipping completed file: {record['name']}")
                continue

            t0 = time.time()
            print(
                f"Writing {record['name']} -> time[{start}:{end}] "
                f"({record['n_time']} steps)"
            )

            with open_year(record["path"]) as ds_year:
                ds_year = ds_year[write_var_names]
                ds_year = ds_year.assign_coords(time=build_time_axis(total_time)[start:end])

                write_graph = ds_year.to_zarr(
                    ANNUAL_STORE,
                    mode="r+",
                    region={
                        "time": slice(start, end),
                        "lat": slice(0, ds_year.sizes["lat"]),
                        "lon": slice(0, ds_year.sizes["lon"]),
                    },
                    consolidated=False,
                    compute=False,
                )
                client.compute(write_graph).result()

            elapsed = max(time.time() - t0, 1e-6)
            logical_bytes = 0
            for data_array in ds_year.data_vars.values():
                logical_bytes += math.prod(data_array.sizes.values()) * data_array.dtype.itemsize
            mib_s = logical_bytes / elapsed / (1024 * 1024)
            print(f"Completed in {elapsed:.1f}s at about {mib_s:.1f} MiB/s logical")

            completed.add(record["name"])
            save_progress({"completed": sorted(completed)})

    finalize_metadata()
    ds_final = xr.open_zarr(ANNUAL_STORE, consolidated=True)
    print(f"Done. Annual chunked store: {ANNUAL_STORE}")
    print(f"Total time steps: {ds_final.sizes['time']} across {len(records)} files")


if __name__ == "__main__":
    main()
