#%%
import ee
from constants import STORAGE_DIR, AREA_BBOX
import time
import os
import subprocess


RCLONE_REMOTE = "gdrive:"
REMOTE_DIR = "GEE_TMAX"
LOCAL_DIR = STORAGE_DIR+"gee_download_t2m/"
TASK_POLL_INTERVAL_S = 60

ee.Authenticate(auth_mode='notebook')
ee.Initialize()


bbox = ee.Geometry.Rectangle([
    AREA_BBOX['westlimit'],
    AREA_BBOX['southlimit'],
    AREA_BBOX['eastlimit'],
    AREA_BBOX['northlimit'],
])

DATASET = 'projects/sat-io/open-datasets/global-daily-air-temp/latin_america'
EXPORT_FOLDER = 'GEE_TMAX'

collection = ee.ImageCollection(DATASET)

# Native projection
projection = ee.Image(collection.first()).projection()
scale = projection.nominalScale().getInfo()

print(f"Native scale: {scale:.3f} m")

os.makedirs(LOCAL_DIR, exist_ok=True)

#%%
def wait_for_task(task):
    """Block until a GEE export task reaches a terminal state and return that state."""
    elapsed = 0
    while True:
        state = task.status()['state']
        if state in ('COMPLETED', 'FAILED', 'CANCELLED', 'CANCEL_REQUESTED'):
            return state
        print(f"    task state: {state} (elapsed {elapsed // 60} min)")
        time.sleep(TASK_POLL_INTERVAL_S)
        elapsed += TASK_POLL_INTERVAL_S


def fetch_and_cleanup(filename):
    remote_path = f"{RCLONE_REMOTE}{REMOTE_DIR}/{filename}"
    print(f"  downloading {filename} to {LOCAL_DIR}")
    subprocess.run(["rclone", "copyto", remote_path, os.path.join(LOCAL_DIR, filename)], check=True)
    print(f"  deleting {filename} from Drive")
    subprocess.run(["rclone", "deletefile", remote_path], check=True)


#%%
def export_year(year):

    tmax = (
        collection
        .filterDate(f'{year}-01-01', f'{year + 1}-01-01')
        .filter(ee.Filter.stringContains('system:index', 'TMAX'))
        .sort('system:time_start')
    )

    n = tmax.size().getInfo()
    print(f"{year}: {n} images")

    if n != 365:
        raise ValueError(f"Expected 365 images, found {n}")

    # Single-band annual mean (missing days masked out) instead of a 365-band daily
    # stack - much smaller, so it should no longer hit GEE's per-export shard limit
    # that was producing multiple tifs for one year (not a hard guarantee for very
    # large ROIs, but expected to resolve it here).
    yearly_image = (
        tmax
        .map(lambda img: img.updateMask(img.neq(-9999)))
        .select('b1')
        .mean()
        .rename(f'TMAX_mean_{year}')
        .clip(bbox)
    )
    filename = f'TMAX_mean_{year}'

    task = ee.batch.Export.image.toDrive(
        image=yearly_image,
        description=filename,
        folder=EXPORT_FOLDER,
        fileNamePrefix=filename,
        region=bbox,
        scale=scale,
        crs='EPSG:4326',
        fileFormat='GeoTIFF',
        formatOptions={
            'cloudOptimized': True,
        },
        maxPixels=1e13,
    )

    task.start()
    print(f"{year}: submitted export task, waiting for completion...")

    state = wait_for_task(task)

    if state == 'COMPLETED':
        print(f"{year}: task completed")
        fetch_and_cleanup(f'{filename}.tif')
        print(f"{year}: done")
    else:
        print(f"{year}: task ended with state {state}, skipping transfer")

#%%
if __name__ == "__main__":
    for year in range(2003, 2025):
        print(f"=== Year {year} ===")
        export_year(year)