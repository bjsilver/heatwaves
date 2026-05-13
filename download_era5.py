"""Download ERA5 daily-mean skin temperature for the Amazon region via CDS API.

This script requests the CDS dataset
`derived-era5-single-levels-daily-statistics` using `cdsapi` and saves one
NetCDF file per year into `ERA5_PATH`.

Notes
-----
- The CDS daily-statistics form currently requires `product_type`, `variable`,
  `year`, `month`, `day`, `daily_statistic`, `time_zone`, `frequency`, and
  `area`.
- CDS request field names have changed over time. This script first tries the
  newer `data_format` / `download_format` keys and falls back to the older
  `format` key if needed.
- `AMAZON_BBOX` in `constants.py` is stored as xarray-style slices; CDS expects
  `[north, west, south, east]`.
"""
#%%
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from constants import AMAZON_BBOX, ERA5_PATH

try:
    import cdsapi  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "cdsapi is required for this downloader. Install it in your environment first."
    ) from exc


DATASET = "derived-era5-single-levels-daily-statistics"
VARIABLE = "skin_temperature"
PRODUCT_TYPE = "reanalysis"
DAILY_STATISTIC = "daily_maximum"
TIME_ZONE = "utc+00:00"
FREQUENCY = "1_hourly"

# Adjust these if you want a different period.
START_YEAR = 2001
END_YEAR = 2020

OUTPUT_DIR = Path('/uolstore/Research/b/b0256/Users/earbsi/heatwaves/era5/')
OUTPUT_TEMPLATE = "era5_land_skinT_dailymax_{year}.nc"

MONTHS = [f"{month:02d}" for month in range(1, 13)]
DAYS = [f"{day:02d}" for day in range(1, 32)]


def bbox_to_cds_area(bbox: dict[str, slice]) -> list[float]:
    """Convert an xarray-style bbox dict to CDS area order.

    Parameters
    ----------
    bbox
        Dictionary like `{'latitude': slice(-30, 20), 'longitude': slice(-82, -34.5)}`.

    Returns
    -------
    list[float]
        CDS area list in `[north, west, south, east]` order.
    """
    lat_slice = bbox["latitude"]
    lon_slice = bbox["longitude"]

    south = min(lat_slice.start, lat_slice.stop)
    north = max(lat_slice.start, lat_slice.stop)
    west = min(lon_slice.start, lon_slice.stop)
    east = max(lon_slice.start, lon_slice.stop)
    return [north, west, south, east]


def build_base_request(year: int) -> dict[str, object]:
    return {
        "product_type": PRODUCT_TYPE,
        "variable": VARIABLE,
        "year": str(year),
        "month": MONTHS,
        "day": DAYS,
        "daily_statistic": DAILY_STATISTIC,
        "time_zone": TIME_ZONE,
        "frequency": FREQUENCY,
        "area": bbox_to_cds_area(AMAZON_BBOX),
    }


def target_path(year: int) -> Path:
    return OUTPUT_DIR / OUTPUT_TEMPLATE.format(year=year)


def iter_years(start_year: int = START_YEAR, end_year: int = END_YEAR) -> Iterable[int]:
    return range(start_year, end_year + 1)


def retrieve_year(client: cdsapi.Client, year: int, overwrite: bool = False) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = target_path(year)

    if output_path.exists() and not overwrite:
        print(f"Skipping existing file: {output_path}")
        return output_path

    base_request = build_base_request(year)
    new_style_request = {
        **base_request,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    old_style_request = {
        **base_request,
        "format": "netcdf",
    }

    print(f"Requesting {year} -> {output_path.name}")
    try:
        client.retrieve(DATASET, new_style_request, str(output_path))
    except Exception as exc:
        print(
            "New-style CDS request keys failed; retrying with legacy `format` key. "
            f"Original error: {exc}"
        )
        client.retrieve(DATASET, old_style_request, str(output_path))

    return output_path


def main(start_year: int = START_YEAR, end_year: int = END_YEAR, overwrite: bool = False) -> None:
    print(f"Dataset: {DATASET}")
    print(f"Variable: {VARIABLE}")
    print(f"Area: {bbox_to_cds_area(AMAZON_BBOX)}")
    print(f"Output directory: {OUTPUT_DIR}")

    client = cdsapi.Client()
    for year in iter_years(start_year, end_year):
        retrieve_year(client, year, overwrite=overwrite)

    print("Download requests complete.")


if __name__ == "__main__":
    main()
