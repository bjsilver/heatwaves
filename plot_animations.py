#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:01:05 2024

@author: ben
"""
#%%
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from constants import FIGDIR
from calculate_heatdays_heatwaves import load_data as load_zz_era_pct
import gc
from pathlib import Path
from tqdm import tqdm
import shutil
from datetime import datetime


def _log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


#%%

def _render_frame_batch(args):
    da, frame_indices, frames_dir, dpi, cmap, vmin, vmax = args

    first_t = frame_indices[0]
    first_frame = da.isel(time=first_t).astype("float32").compute()

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(first_frame, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    title = ax.set_title(str(np.array(da.time.isel(time=first_t).values)))
    plt.colorbar(im, ax=ax, shrink=0.8)

    frame_path = Path(frames_dir) / f"frame_{first_t:06d}.png"
    fig.savefig(frame_path, dpi=dpi, bbox_inches="tight")

    for t in frame_indices[1:]:
        frame = da.isel(time=t).astype("float32").compute()
        im.set_data(frame)
        title.set_text(str(np.array(da.time.isel(time=t).values)))
        frame_path = Path(frames_dir) / f"frame_{t:06d}.png"
        fig.savefig(frame_path, dpi=dpi, bbox_inches="tight")
        del frame
        gc.collect()

    plt.close(fig)
    return len(frame_indices)


def write_animation_parallel_frames(da, out_mp4, workers=4, fps=12, dpi=120, cmap="magma",
                                    vmin=None, vmax=None, crf=17, preset="slow"):
    _log("Preparing animation job")

    tsize = da.sizes["time"]
    batch_size = max(1, tsize // (workers * 4))
    _log(f"Timesteps: {tsize}, workers: {workers}, batch size: {batch_size}")

    frames_dir = Path(out_mp4).with_suffix("") / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Writing temporary frames to: {frames_dir}")

    jobs = []
    for start in range(0, tsize, batch_size):
        stop = min(start + batch_size, tsize)
        frame_indices = list(range(start, stop))
        jobs.append((da, frame_indices, str(frames_dir), dpi, cmap, vmin, vmax))

    with mp.Pool(processes=workers) as pool:
        with tqdm(total=tsize, desc="Rendering frames", unit="frame") as pbar:
            for completed in pool.imap_unordered(_render_frame_batch, jobs):
                pbar.update(completed)

    _log("Frame rendering complete, stitching mp4 with ffmpeg")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "frame_%06d.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            out_mp4,
        ],
        check=True,
    )

    shutil.rmtree(frames_dir, ignore_errors=True)
    _log(f"Done: {out_mp4}")

if __name__ == "__main__":
    mp.freeze_support()
    _log("Loading percentile datasets")
    _, pct = load_zz_era_pct()
    _log("Datasets loaded")
    write_animation_parallel_frames(
        pct["zz_2003-2020"],
        FIGDIR + "percentiles/zz_2003-2020.mp4",
        workers=10,
        fps=10,
        dpi=180,
        vmin=28,
        vmax=40,
    )

    write_animation_parallel_frames(
        pct["era_2003-2020"],
        FIGDIR + "percentiles/era_2003-2020.mp4",
        workers=10,
        fps=10,
        dpi=180,
        vmin=28,
        vmax=40,
    )

    write_animation_parallel_frames(
        pct["era_1970-1999"],
        FIGDIR + "percentiles/era_1970-1999.mp4",
        workers=10,
        fps=10,
        dpi=180,
        vmin=22,
        vmax=36,
    )

