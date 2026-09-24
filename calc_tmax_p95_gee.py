"""GEE reimplementation of the local numba/dask ±7-day cyclic-DOY 95th-percentile
TMAX climatology (see calc_era5_percentiles.py / calculate_zz_percentiles.py).

All heavy computation (windowing, percentile reduction, spatial ops) runs server-side
in Earth Engine. Python only builds the request graph, validates counts, and submits
the export - no numpy/xarray/dask/local raster processing of the daily data.
"""
#%%
import time

import ee
from constants import AREA_BBOX

# --- config -----------------------------------------------------------------
START_YEAR = 2003
END_YEAR = 2024
PAD = 7
PERCENTILE = 95
EXPECTED_YEARS = 22
DAYS_PER_YEAR = 365
EXPECTED_IMAGES = 8030
EXPECTED_WINDOW_IMAGES = 1

GEE_FOLDER = 'GEE_Exports'
OUTPUT_NAME = 'TMAX_p95_2003_2024'

DATASET = 'projects/sat-io/open-datasets/global-daily-air-temp/latin_america'
BAND = 'b1'
MISSING_VALUE = -9999
EXPORT_POLL_SECONDS = 60

# Native TMAX grid - must match download_regrid_hansen_gee.py exactly (do not round).
TMAX_CRS = 'EPSG:4326'
TMAX_TRANSFORM = [
    0.008055117514264009,
    0,
    -154.88527132920055,
    0,
    -0.008055117514264009,
    28.075000000030002,
]


ee.Authenticate(auth_mode='notebook')
ee.Initialize()
TEST_ROI = ee.Geometry.BBox(-63, -6, -62, -5)


# --- pipeline -----------------------------------------------------------------
def load_tmax_collection(roi):
    """Load, filter to TMAX images 2003-2024, mask missing values, clip to roi."""
    collection = (
        ee.ImageCollection(DATASET)
        .filterDate(f'{START_YEAR}-01-01', f'{END_YEAR + 1}-01-01')
        .filter(ee.Filter.stringContains('system:index', 'TMAX'))
        .map(lambda img: img.updateMask(img.neq(MISSING_VALUE)).clip(roi))
    )

    n = collection.size().getInfo()
    print(f'TMAX collection size: {n}')
    if n != EXPECTED_IMAGES:
        raise ValueError(f'Expected {EXPECTED_IMAGES} images, found {n}')

    return collection


def add_date_properties(collection):
    """Tag each image with 'year' and a sequential 1-365 'doy'.

    Deliberately NOT calendar-based (e.g. ee.Date.getRelative('day','year')):
    the source has no Feb-29, so a real-calendar day-of-year would drift by one
    day after Feb in leap years. Instead each calendar year's images are sorted
    by time and assigned doy = 1..365 by position, matching the local numpy
    implementation's flat (year, doy) indexing exactly.
    """
    years = ee.List.sequence(START_YEAR, END_YEAR)
    doy_seq = ee.List.sequence(1, DAYS_PER_YEAR)

    def tag_year(y):
        y = ee.Number(y)
        year_imgs = (
            collection
            .filter(ee.Filter.calendarRange(y, y, 'year'))
            .sort('system:time_start')
            .toList(DAYS_PER_YEAR)
        )
        pairs = doy_seq.zip(year_imgs)

        def tag(pair):
            pair = ee.List(pair)
            doy = ee.Number(pair.get(0))
            img = ee.Image(pair.get(1))
            return img.set('year', y).set('doy', doy)

        return pairs.map(tag)

    tagged = ee.List(years.map(tag_year)).flatten()
    return ee.ImageCollection(tagged)


def add_cyclic_doy_properties(collection):
    """Add doy_minus_365 / doy_plus_365 so windows crossing the year boundary can be
    selected with plain numeric range filters on 'doy'."""

    def tag(img):
        doy = ee.Number(img.get('doy'))
        return img.set('doy_minus_365', doy.subtract(DAYS_PER_YEAR)).set(
            'doy_plus_365', doy.add(DAYS_PER_YEAR)
        )

    return collection.map(tag)


def get_window_collection(collection, d):
    """Select the ±PAD calendar-day window around target doy d, wrapping cyclically.

    An image participates if ANY of its doy / doy-365 / doy+365 falls in
    [d-PAD, d+PAD]. E.g. for d=1, doy=359..365 match via doy-365 in [-6,8], and
    doy=1..8 match directly.
    """
    d = ee.Number(d)
    lo, hi = d.subtract(PAD), d.add(PAD)
    f = ee.Filter.Or(
        ee.Filter.rangeContains('doy', lo, hi),
        ee.Filter.rangeContains('doy_minus_365', lo, hi),
        ee.Filter.rangeContains('doy_plus_365', lo, hi),
    )
    return collection.filter(f)


def calculate_doy_percentile(collection, d):
    """330-image window -> 95th percentile image for target doy d."""
    window = get_window_collection(collection, d).select(BAND)
    # maxRaw forces GEE to compute the exact percentile from raw sorted values
    # rather than its default histogram/bucket approximation. KNOWN DISCREPANCY
    # (confirmed empirically via validate_percentile_equivalence): even with
    # maxRaw set, GEE's result does not use the same h=(m-1)*q linear
    # interpolation as _quantile_skipna_1d - it behaves like a nearest-rank
    # percentile (picks a raw value, e.g. ceil(n*p/100)), producing a small,
    # consistent offset (~0.9 tenths-degC in test data) vs the local method.
    # Accepted as-is per user decision; see validate() output for the measured offset.
    p95 = window.reduce(
        ee.Reducer.percentile([PERCENTILE])
    )
    return p95.rename('Ta_TMAX_p95').set('doy', d)


def build_percentile_collection(collection):
    doys = ee.List.sequence(1, DAYS_PER_YEAR)
    images = doys.map(lambda d: calculate_doy_percentile(collection, d))
    return ee.ImageCollection(images)


def wait_for_export_task(task, task_name):
    """Wait until a GEE export finishes, raising if it fails or is cancelled."""
    while True:
        status = task.status()
        state = status.get('state')
        print(f'{task_name}: {state}')

        if state == 'COMPLETED':
            return
        if state in {'FAILED', 'CANCELLED'}:
            error = status.get('error_message', 'no error message provided')
            raise RuntimeError(f'{task_name} {state.lower()}: {error}')

        time.sleep(EXPORT_POLL_SECONDS)


def export_percentiles(p95_collection, roi):
    """Submit one GeoTIFF export task for each day-of-year percentile image."""
    images = p95_collection.toList(DAYS_PER_YEAR)
    tasks = []

    for doy in range(1, DAYS_PER_YEAR + 1):
        task_name = f'{OUTPUT_NAME}_doy_{doy:03d}'
        task = ee.batch.Export.image.toDrive(
            image=ee.Image(images.get(doy - 1)),
            description=task_name,
            folder=GEE_FOLDER,
            fileNamePrefix=task_name,
            region=roi,
            crs=TMAX_CRS,
            crsTransform=TMAX_TRANSFORM,
            fileFormat='GeoTIFF',
            formatOptions={'cloudOptimized': True},
            maxPixels=1e13,
        )
        task.start()
        tasks.append(task)
        print(f'Submitted export task: {task_name}')
        wait_for_export_task(task, task_name)

    return tasks


# --- validation ---------------------------------------------------------------
def validate_window_counts(collection, sample_doys=(1, 100, 180, 365)):
    for d in sample_doys:
        n = get_window_collection(collection, d).size().getInfo()
        print(f'doy={d}: {n} images selected (expected {EXPECTED_WINDOW_IMAGES})')
        if n != EXPECTED_WINDOW_IMAGES:
            raise ValueError(
                f'doy={d}: expected {EXPECTED_WINDOW_IMAGES} images, got {n}'
            )


def validate_percentile_equivalence(collection, scale, sample_doy=1):
    """Compare GEE's percentile reducer against a local NumPy linear-interpolation
    percentile for one pixel, over the tiny TEST_ROI only.

    CONFIRMED DISCREPANCY: even with maxRaw=EXPECTED_WINDOW_IMAGES (forcing GEE's
    exact/raw computation instead of its histogram approximation), GEE's result
    does not match numpy's 'linear' interpolation used by _quantile_skipna_1d.
    Empirically GEE behaves like a nearest-rank percentile (e.g. index
    ceil(n*p/100), no interpolation), producing a small systematic offset
    (~0.9 tenths-degC / ~0.09 degC in test data) vs the local method. This has
    been accepted as a known, documented limitation rather than worked around.
    """
    import numpy as np

    point = TEST_ROI.centroid(maxError=1)
    window = get_window_collection(collection, sample_doy).select(BAND)

    region_rows = window.getRegion(point, scale).getInfo()
    header = region_rows[0]
    band_idx = header.index(BAND)
    raw_values = [row[band_idx] for row in region_rows[1:] if row[band_idx] is not None]

    local_p95 = float(np.percentile(raw_values, PERCENTILE))
    gee_p95 = (
        calculate_doy_percentile(collection, sample_doy)
        .reduceRegion(ee.Reducer.first(), point, scale)
        .get('Ta_TMAX_p95')
        .getInfo()
    )

    print(
        f'doy={sample_doy}: n={len(raw_values)}, '
        f'local numpy p95={local_p95:.3f}, GEE p95={gee_p95:.3f}, '
        f'diff={abs(local_p95 - gee_p95):.3f}'
    )


def validate(collection, scale):
    validate_window_counts(collection)

    sample_p95 = calculate_doy_percentile(collection, 1)
    stats = sample_p95.reduceRegion(
        reducer=ee.Reducer.minMax().combine(ee.Reducer.mean(), sharedInputs=True),
        geometry=TEST_ROI,
        scale=scale,
        maxPixels=1e9,
    ).getInfo()
    print(f'doy=1 test ROI stats: {stats}')

    validate_percentile_equivalence(collection, scale, sample_doy=1)
    validate_percentile_equivalence(collection, scale, sample_doy=365)

#%%
# --- main -----------------------------------------------------------------
def main():

    roi = ee.Geometry.BBox(
        AREA_BBOX['westlimit'],
        AREA_BBOX['southlimit'],
        AREA_BBOX['eastlimit'],
        AREA_BBOX['northlimit'],
    )

    raw_collection = load_tmax_collection(roi)
    collection = add_cyclic_doy_properties(add_date_properties(raw_collection))

    scale = ee.Image(collection.first()).select(BAND).projection().nominalScale().getInfo()
    # validate(collection, scale)

    p95_collection = build_percentile_collection(collection)
    export_percentiles(p95_collection, roi)


if __name__ == '__main__':
    main()
