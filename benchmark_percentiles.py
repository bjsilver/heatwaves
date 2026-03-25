#!/usr/bin/env python3
"""Benchmark Dask cluster configs for climatological percentile workflow.

Usage example:
python benchmark_percentiles.py \
  --configs 4,1,12GB 6,1,12GB 8,1,10GB \
  --lat-min -5 --lat-max 0 --lon-min -70 --lon-max -65 \
  --output-csv benchmark_results.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile
import time
from dataclasses import dataclass

import xarray as xr
from dask.distributed import Client, LocalCluster

from calculate_tmax_percentiles import PAD, compute_percentiles, load_tmax_da


@dataclass
class ClusterConfig:
    n_workers: int
    threads_per_worker: int
    memory_limit: str


@dataclass
class BenchmarkResult:
    config: ClusterConfig
    mode: str
    seconds: float
    status: str
    message: str


def parse_config(s: str) -> ClusterConfig:
    """Parse config string in the form 'workers,threads,memory'"""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(
            f"Invalid config '{s}'. Expected format workers,threads,memory (e.g. 4,1,12GB)."
        )
    return ClusterConfig(
        n_workers=int(parts[0]),
        threads_per_worker=int(parts[1]),
        memory_limit=parts[2],
    )


def select_subset(da: xr.DataArray, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> xr.DataArray:
    """Subset with robust handling of descending or ascending latitude coordinates."""
    lat0 = float(da.lat.values[0])
    lat1 = float(da.lat.values[-1])
    if lat0 > lat1:
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)
    lon_slice = slice(lon_min, lon_max)
    return da.sel(lat=lat_slice, lon=lon_slice)


def run_single_benchmark(
    cfg: ClusterConfig,
    da_subset: xr.DataArray,
    write_test: bool,
    output_dir: str,
) -> BenchmarkResult:
    mode = "write" if write_test else "compute"
    t0 = time.perf_counter()
    try:
        with LocalCluster(
            n_workers=cfg.n_workers,
            threads_per_worker=cfg.threads_per_worker,
            memory_limit=cfg.memory_limit,
            dashboard_address=":8787",
        ) as cluster, Client(cluster) as client:
            print(
                f"Running config workers={cfg.n_workers}, threads={cfg.threads_per_worker}, "
                f"memory={cfg.memory_limit} ({mode} mode)"
            )
            print(f"Dashboard: {client.dashboard_link}")

            pct = compute_percentiles(da_subset, PAD)

            if write_test:
                zarr_path = os.path.join(
                    output_dir,
                    f"benchmark_w{cfg.n_workers}_t{cfg.threads_per_worker}_m{cfg.memory_limit}.zarr",
                )
                if os.path.isdir(zarr_path):
                    shutil.rmtree(zarr_path)
                pct.to_zarr(zarr_path, mode="w", consolidated=True)
            else:
                # Force full graph execution while minimizing write overhead.
                _ = pct.mean().compute()

        return BenchmarkResult(cfg, mode, time.perf_counter() - t0, "ok", "")
    except Exception as exc:
        return BenchmarkResult(
            cfg,
            mode,
            time.perf_counter() - t0,
            "error",
            f"{type(exc).__name__}: {exc}",
        )


def write_results_csv(path: str, rows: list[BenchmarkResult]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "workers",
            "threads_per_worker",
            "memory_limit",
            "mode",
            "seconds",
            "status",
            "message",
        ])
        for r in rows:
            writer.writerow([
                r.config.n_workers,
                r.config.threads_per_worker,
                r.config.memory_limit,
                r.mode,
                f"{r.seconds:.3f}",
                r.status,
                r.message,
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Dask configs for percentile script.")
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="List of workers,threads,memory values, e.g. 4,1,12GB 6,1,12GB",
    )
    parser.add_argument("--lat-min", type=float, required=True, help="Subset minimum latitude")
    parser.add_argument("--lat-max", type=float, required=True, help="Subset maximum latitude")
    parser.add_argument("--lon-min", type=float, required=True, help="Subset minimum longitude")
    parser.add_argument("--lon-max", type=float, required=True, help="Subset maximum longitude")
    parser.add_argument(
        "--write-test",
        action="store_true",
        help="Write benchmark output to temporary zarr (end-to-end test).",
    )
    parser.add_argument(
        "--output-csv",
        default="benchmark_percentiles_results.csv",
        help="Path for benchmark result CSV.",
    )
    args = parser.parse_args()

    configs = [parse_config(s) for s in args.configs]

    print("Loading source data lazily...")
    da = load_tmax_da()
    da_subset = select_subset(da, args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    print(f"Subset shape: {da_subset.sizes}")

    work_dir = tempfile.mkdtemp(prefix="pct_bench_")
    print(f"Temporary benchmark directory: {work_dir}")

    results: list[BenchmarkResult] = []
    try:
        for cfg in configs:
            result = run_single_benchmark(
                cfg=cfg,
                da_subset=da_subset,
                write_test=args.write_test,
                output_dir=work_dir,
            )
            results.append(result)
            print(
                f"Result: workers={cfg.n_workers}, threads={cfg.threads_per_worker}, "
                f"mem={cfg.memory_limit}, mode={result.mode}, seconds={result.seconds:.2f}, "
                f"status={result.status}"
            )
            if result.message:
                print(f"  details: {result.message}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    write_results_csv(args.output_csv, results)
    print(f"Wrote benchmark summary to {args.output_csv}")

    ok = [r for r in results if r.status == "ok"]
    if ok:
        best = min(ok, key=lambda r: r.seconds)
        print(
            "Best successful config: "
            f"workers={best.config.n_workers}, threads={best.config.threads_per_worker}, "
            f"mem={best.config.memory_limit}, mode={best.mode}, seconds={best.seconds:.2f}"
        )


if __name__ == "__main__":
    main()
