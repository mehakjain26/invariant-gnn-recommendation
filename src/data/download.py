"""Download and cache recommendation datasets."""

import os
import zipfile
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

DATASET_URLS = {
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "amazon-books": None,
    "yelp2018": None,
}


def download_ml1m(data_dir: Path = DATA_DIR) -> Path:
    """Download and extract MovieLens-1M."""
    dest = data_dir / "ml-1m"
    if dest.exists() and (dest / "ratings.dat").exists():
        print("ML-1M already downloaded.")
        return dest

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "ml-1m.zip"
    url = DATASET_URLS["ml-1m"]
    print(f"Downloading ML-1M from {url} ...")
    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    zip_path.unlink()
    print(f"ML-1M extracted to {dest}")
    return dest


def download_amazon_books(data_dir: Path = DATA_DIR) -> Path:
    """Download Amazon-Books dataset (5-core ratings)."""
    dest = data_dir / "amazon-books"
    rating_file = dest / "Books.csv"
    if dest.exists() and rating_file.exists():
        print("Amazon-Books already downloaded.")
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    url = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Books.csv"
    print(f"Downloading Amazon-Books from {url} ...")
    urllib.request.urlretrieve(url, rating_file)
    print(f"Amazon-Books saved to {rating_file}")
    return dest


def download_yelp2018(data_dir: Path = DATA_DIR) -> Path:
    """Yelp-2018 requires manual download or a preprocessed benchmark copy."""
    dest = data_dir / "yelp2018"
    if dest.exists():
        print("Yelp-2018 directory exists. Ensure data files are present.")
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    print(
        "Yelp-2018 requires manual download.\n"
        "Please download from https://www.yelp.com/dataset\n"
        f"and place files in {dest}"
    )
    return dest


def download_dataset(name: str, data_dir: Path = DATA_DIR) -> Path:
    """Download a dataset by name."""
    downloaders = {
        "ml-1m": download_ml1m,
        "amazon-books": download_amazon_books,
        "yelp2018": download_yelp2018,
    }
    if name not in downloaders:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(downloaders.keys())}")
    return downloaders[name](data_dir)


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "ml-1m"
    download_dataset(name)
