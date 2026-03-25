#%%
import xarray as xr
import os
# This must be set before Dask/Distributed are imported
os.environ["MALLOC_TRIM_THRESHOLD_"] = "65536"
import math
import shutil
import glob
from rechunker import rechunk
from dask.distributed import Client, LocalCluster
import dask
from dask.utils import format_bytes, parse_bytes

dask.config.set({
    # Start moving data to disk when worker is 50% full
    "distributed.worker.memory.target": 0.50,
    # Stop the worker from taking new tasks at 60%
    "distributed.worker.memory.pause": 0.60,
    # If unmanaged memory is the issue, this forces a spill
    "distributed.worker.memory.spill": 0.55,
})


# ==========================================
# 1. CONFIGURATION (Update these paths!)
# ==========================================
# Path to your 16 NetCDF files
NC_SOURCE_PATTERN = '/uolstore/Research/b/b0256/Users/earjba/ZhangandZhou/tropical_regions/*TMAX_daily_tropics30N30S.nc'

# Where to save the output (needs ~1TB of space)
WORK_DIR = '/uolstore/Research/b/b0256/Users/earbsi/heatwaves/rechunked/' 
TARGET_STORE = os.path.join(WORK_DIR, 'tmax_rechunked.zarr')
TEMP_STORE = os.path.join(WORK_DIR, 'rechunk_temp.zarr')

# Set your coordinate start date
START_DATE = "2006-01-01"

# Keep the cluster budget comfortably below host RAM. The old defaults
# requested 8 x 40 GB = 320 GB, which can get the process OOM-killed on a
# ~190 GB machine even though rechunker itself is memory-aware.
N_WORKERS = 4
THREADS_PER_WORKER = 1
WORKER_MEMORY_LIMIT = '20GB'

# These source files are effectively contiguous on disk, so preserving the
# source layout would expose one very large chunk per yearly file. Force a
# modest Dask chunking at read time so rechunker has bounded-size source blocks.
SOURCE_CHUNKS = {'time': 30, 'lat': 512, 'lon': 512}

# This must be at least as large as the biggest source or target chunk.
MAX_MEM = '2GB'

def preprocess_drop(ds):
    """Clean up files so they stack perfectly by index."""
    return ds.drop_vars(['DOY', 'spatial_ref'], errors='ignore')


def get_system_memory_bytes():
    """Return total host RAM so we can reject unsafe cluster settings early."""
    return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')


def get_largest_source_chunk_bytes(ds):
    """Estimate the largest currently exposed chunk across all data variables."""
    largest_chunk = 0
    for data_array in ds.data_vars.values():
        if data_array.chunks is None:
            continue
        chunk_shape = [max(axis_chunks) for axis_chunks in data_array.chunks]
        chunk_bytes = math.prod(chunk_shape) * data_array.dtype.itemsize
        largest_chunk = max(largest_chunk, chunk_bytes)
    return largest_chunk


def get_largest_target_chunk_bytes(ds, target_chunks):
    """Estimate the largest output chunk implied by the rechunk target."""
    largest_chunk = 0
    for data_array in ds.data_vars.values():
        chunk_shape = []
        for dim in data_array.dims:
            target = target_chunks.get(dim, data_array.sizes[dim])
            chunk_shape.append(data_array.sizes[dim] if target == -1 else target)
        chunk_bytes = math.prod(chunk_shape) * data_array.dtype.itemsize
        largest_chunk = max(largest_chunk, chunk_bytes)
    return largest_chunk


def validate_memory_budget(ds, target_chunks, max_mem):
    """Fail fast when the cluster or rechunk plan is larger than expected."""
    largest_source_chunk = get_largest_source_chunk_bytes(ds)
    largest_target_chunk = get_largest_target_chunk_bytes(ds, target_chunks)
    requested_max_mem = parse_bytes(max_mem)

    print(f"📚 Largest source chunk: {format_bytes(largest_source_chunk)}")
    print(f"🎯 Largest target chunk: {format_bytes(largest_target_chunk)}")
    print(f"🧠 Rechunk max_mem: {format_bytes(requested_max_mem)}")

    largest_required_chunk = max(largest_source_chunk, largest_target_chunk)
    if requested_max_mem < largest_required_chunk:
        raise ValueError(
            "max_mem must be at least as large as the largest source/target chunk: "
            f"need {format_bytes(largest_required_chunk)}, got {max_mem}"
        )

    total_worker_budget = N_WORKERS * parse_bytes(WORKER_MEMORY_LIMIT)
    system_memory = get_system_memory_bytes()
    print(f"🖥️  Host RAM: {format_bytes(system_memory)}")
    print(f"👷 Cluster memory budget: {format_bytes(total_worker_budget)}")

    if total_worker_budget > int(system_memory * 0.8):
        raise ValueError(
            "Cluster memory budget is too close to host RAM. Lower N_WORKERS or "
            f"WORKER_MEMORY_LIMIT. Requested {format_bytes(total_worker_budget)} "
            f"on a host with {format_bytes(system_memory)}."
        )


def estimate_total_data_bytes(ds):
    """Estimate total logical size across all data variables."""
    total_bytes = 0
    for data_array in ds.data_vars.values():
        n_items = math.prod(data_array.sizes.values())
        total_bytes += n_items * data_array.dtype.itemsize
    return total_bytes


def print_runtime_estimate(ds):
    """Print coarse runtime estimates from expected total I/O volume."""
    data_bytes = estimate_total_data_bytes(ds)
    # Typical rechunker workflow is roughly:
    # read source + write temp + read temp + write target ~= 4x logical size.
    io_multiplier = 4
    total_io_bytes = data_bytes * io_multiplier
    print(f"📦 Logical data size: {format_bytes(data_bytes)}")
    print(f"💽 Estimated total I/O (x{io_multiplier}): {format_bytes(total_io_bytes)}")

    for mib_s in (50, 100, 200, 400):
        bytes_per_second = mib_s * 1024 * 1024
        seconds = total_io_bytes / bytes_per_second
        hours = seconds / 3600
        print(f"⏱️  At ~{mib_s} MiB/s sustained: ~{hours:.1f} hours")
#%%
# ==========================================
# 2. EXECUTION ENGINE
# ==========================================
if __name__ == "__main__":
    # --- Step A: Prepare File List ---
    # CRITICAL: We sort to ensure 2007 is index 0, 2008 is index 365, etc.
    files = sorted(glob.glob(NC_SOURCE_PATTERN))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {NC_SOURCE_PATTERN}")
    print(f"📦 Found {len(files)} files. First: {os.path.basename(files[0])}")

    os.makedirs(WORK_DIR, exist_ok=True)

    # Start Dask Cluster
    # Forward port 8787 in VS Code to see the dashboard
    with LocalCluster(
        n_workers=N_WORKERS,
        threads_per_worker=THREADS_PER_WORKER,
        memory_limit=WORKER_MEMORY_LIMIT,
        dashboard_address=':8787'
    ) as cluster, Client(cluster) as client:
        print(f"🚀 Dask Dashboard: {client.dashboard_link}")

        # --- Step B: Open Dataset ---
        ds = xr.open_mfdataset(
            files,
            preprocess=preprocess_drop,
            concat_dim='time',
            combine='nested',
            chunks=SOURCE_CHUNKS,
            decode_cf=False,
            mask_and_scale=False
        )

        # --- Step C: Define Rechunk Plan ---
        # We want ALL time (5840) for every 100x100 pixel block
        target_chunks = {'time': -1, 'lat': 100, 'lon': 100}

        validate_memory_budget(ds, target_chunks, MAX_MEM)
        print_runtime_estimate(ds)

#%%

        # Clean up previous failed runs
        for p in [TARGET_STORE, TEMP_STORE]:
            if os.path.isdir(p):
                print(f"🧹 Removing old data at {p}")
                shutil.rmtree(p)
            elif os.path.exists(p):
                os.remove(p)

        # Initialize Rechunker
        array_plan = rechunk(
            ds,
            target_chunks=target_chunks,
            max_mem=MAX_MEM,
            target_store=TARGET_STORE,
            temp_store=TEMP_STORE
        )

        print("🛠️  Executing Rechunker Plan (Phase 1: Read/Temp, Phase 2: Final)...")
        array_plan.execute()

        # --- Step D: Finalize Metadata (The 'Year' Tracker) ---
        print("📝 Stamping Zarr with proper time coordinates...")

        # Open the finished Zarr (lazy)
        ds_final = xr.open_zarr(TARGET_STORE)

        # Create the 16-year time axis (No-Leap)
        time_axis = xr.date_range(
            start=START_DATE,
            periods=len(ds_final.time),
            freq="D",
            calendar="noleap",
            use_cftime=True
        )

        # Assign and save metadata only
        ds_final = ds_final.assign_coords(time=time_axis)
        ds_final.to_zarr(TARGET_STORE, mode='a', consolidated=True)

        # --- Step E: Cleanup ---
        if os.path.isdir(TEMP_STORE):
            shutil.rmtree(TEMP_STORE)

        print(f"✅ Success! Final Zarr ready at: {TARGET_STORE}")
        print(f"Total time steps: {len(ds_final.time)} ({len(files)} years)")
