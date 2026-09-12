"""Download the DeepWeeds dataset (images + labels.csv) into data/raw/.

Source: https://github.com/AlexOlsen/DeepWeeds
Images are distributed as a single zip (~470 MB); labels.csv is in the repo.
"""
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
LABELS_URL = "https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels/labels.csv"
# TODO: confirm current images.zip mirror (originally hosted on Google Drive / Zenodo)
IMAGES_URL = ""


def download_labels() -> Path:
    raise NotImplementedError


def download_images() -> Path:
    raise NotImplementedError


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    download_labels()
    download_images()
