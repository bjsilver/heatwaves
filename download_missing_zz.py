"""Download Zhang & Zhou Figshare LST files for selected years.

The collection page contains one article per year/time-of-day. This script uses the
Figshare API to discover the correct yearly article, then downloads each TIFF file
from the article's file list.
"""
#%%
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import hashlib
import re

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


COLLECTION_ID = 5078492
COLLECTION_API_URL = f"https://api.figshare.com/v2/collections/{COLLECTION_ID}"
COLLECTION_ARTICLES_API_URL = f"{COLLECTION_API_URL}/articles"

# Years to download.
#YEARS = [2003, 2004, 2005, 2018, 2019, 2020]
YEARS = range(2006, 2018)

# Daytime files are the ~71 GB per-year products requested here.
TIME_OF_DAY = "daytime"

OUTPUT_DIR = Path("/uolstore/Research/b/b0256/Users/earbsi/heatwaves/zhang_zhou_tifs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 1024 * 1024 * 8  # 8 MB
REQUEST_TIMEOUT = (30, 300)


def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "heatwaves-figshare-downloader/1.0"})
    return session


def get_json(session: requests.Session, url: str, **params):
    response = session.get(url, params=params or None, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def iter_collection_articles(session: requests.Session) -> Iterable[dict]:
    page = 1
    page_size = 100

    while True:
        articles = get_json(
            session,
            COLLECTION_ARTICLES_API_URL,
            page=page,
            page_size=page_size,
        )
        if not articles:
            break

        yield from articles

        if len(articles) < page_size:
            break
        page += 1


def find_article_for_year(articles: Iterable[dict], year: int, time_of_day: str) -> dict:
    for article in articles:
        title = article["title"].lower()
        if str(year) in title and time_of_day in title:
            return article

    raise ValueError(f"No {time_of_day} article found for year {year}")


def list_article_files(session: requests.Session, article_id: int) -> list[dict]:
    article = get_json(session, f"https://api.figshare.com/v2/articles/{article_id}")
    files = article.get("files", [])
    if not files:
        raise ValueError(f"Article {article_id} does not contain any files")
    return sorted(files, key=lambda item: item["name"])


def md5sum(file_path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    digest = hashlib.md5()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid_existing_file(file_path: Path, metadata: dict) -> bool:
    if not file_path.exists():
        return False

    expected_size = metadata.get("size")
    if expected_size is not None and file_path.stat().st_size != expected_size:
        return False

    expected_md5 = metadata.get("computed_md5") or metadata.get("supplied_md5")
    if expected_md5:
        return md5sum(file_path) == expected_md5

    return True


def safe_year_dir(base_dir: Path, year: int) -> Path:
    year_dir = base_dir / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)
    return year_dir


def extract_year_from_title(title: str) -> int:
    match = re.search(r"\b(20\d{2})\b", title)
    if not match:
        raise ValueError(f"Could not extract year from title: {title}")
    return int(match.group(1))


def download_file(session: requests.Session, file_metadata: dict, destination: Path) -> None:
    temp_path = destination.with_suffix(destination.suffix + ".part")
    url = file_metadata["download_url"]
    expected_size = file_metadata.get("size")

    with session.get(url, stream=True, timeout=REQUEST_TIMEOUT) as response:
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type.lower():
            raise RuntimeError(
                f"Unexpected HTML response for {destination.name} from {response.url}"
            )

        with temp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    handle.write(chunk)

    actual_size = temp_path.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded size mismatch for {destination.name}: "
            f"got {actual_size}, expected {expected_size}"
        )

    expected_md5 = file_metadata.get("computed_md5") or file_metadata.get("supplied_md5")
    if expected_md5 and md5sum(temp_path) != expected_md5:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"MD5 mismatch for {destination.name}")

    temp_path.replace(destination)


def main() -> None:
    session = build_session()

    print(f"Loading collection metadata from {COLLECTION_API_URL}")
    collection = get_json(session, COLLECTION_API_URL)
    print(f"Collection: {collection['title']}")

    articles = list(iter_collection_articles(session))
    print(f"Found {len(articles)} articles in collection {COLLECTION_ID}")

    for year in YEARS:
        article = find_article_for_year(articles, year, TIME_OF_DAY)
        title = article["title"]
        article_year = extract_year_from_title(title)
        year_dir = safe_year_dir(OUTPUT_DIR, article_year)
        print(f"\nYear {year}: {title} (article {article['id']})")

        files = list_article_files(session, article["id"])
        print(f"  {len(files)} files listed")

        for index, file_metadata in enumerate(files, start=1):
            destination = year_dir / file_metadata["name"]
            size_gb = file_metadata["size"] / 1024**3

            if is_valid_existing_file(destination, file_metadata):
                print(
                    f"  [{index:03d}/{len(files):03d}] Skipping existing {destination.name} "
                    f"({size_gb:.2f} GB)"
                )
                continue

            print(
                f"  [{index:03d}/{len(files):03d}] Downloading {destination.name} "
                f"({size_gb:.2f} GB)"
            )
            download_file(session, file_metadata, destination)

    print("\nDownload complete.")


if __name__ == "__main__":
    main()
# %%
