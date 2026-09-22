"""Download the DeepWeeds dataset (images + labels.csv) into data/raw/.

Source: https://github.com/AlexOlsen/DeepWeeds
Images are distributed as a single zip (~470 MB); labels.csv is in the repo.
"""
from pathlib import Path
import urllib.request, zipfile
import gdown

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
LABELS_URL = "https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels/labels.csv"
# TODO: confirm current images.zip mirror (originally hosted on Google Drive / Zenodo)
IMAGES_URL = "https://drive.google.com/file/d/1xnK3B6K6KekDI55vwJ0vnc2IGoDga9cj"


def download_labels() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / "labels.csv"
    if not dest.exists():
        urllib.request.urlretrieve(LABELS_URL, dest)
    return dest


def download_images() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "images.zip"
    extract_dir = RAW_DIR / "images"

    if not extract_dir.exists():
        if not zip_path.exists():
            file_id = IMAGES_URL.split("/d/")[1].split("/")[0]
            gdown.download(id=file_id, output=str(zip_path), quiet=False)

        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)

    return extract_dir


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    download_labels()
    download_images()
