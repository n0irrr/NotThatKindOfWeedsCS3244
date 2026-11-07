"""Download the DeepWeeds images into data/images/.

Images are distributed as a single zip (~470 MB).
Needs gdown:  pip install -U gdown

    python -m src.data.download

Result: data/images/*.jpg
"""
import shutil
import zipfile
from pathlib import Path

import gdown

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
ZIP_PATH = DATA_DIR / "images.zip"
FOLDER_TMP = DATA_DIR / "_gdrive_tmp"

DRIVE_ID = "1xnK3B6K6KekDI55vwJ0vnc2IGoDga9cj"

MANUAL_HINT = (
    "Download images.zip manually from the DeepWeeds README "
    f"(https://github.com/AlexOlsen/DeepWeeds), save it as {ZIP_PATH}, and re-run."
)


def _fetch_zip() -> None:
    """Get images.zip into ZIP_PATH. Tries the ID as a file, then as a folder."""
    part = ZIP_PATH.with_name(ZIP_PATH.name + ".part")

    # 1) the ID is a single file
    try:
        out = gdown.download(url=f"https://drive.google.com/uc?id={DRIVE_ID}",
                             output=str(part), quiet=False)
    except Exception as e:
        print(f"single-file download failed ({type(e).__name__}); trying it as a folder")
        out = None
    if out:
        part.replace(ZIP_PATH)  # rename only once the download fully finished
        return

    # 2) the ID is a folder: download it and pull out images.zip
    try:
        gdown.download_folder(id=DRIVE_ID, output=str(FOLDER_TMP), quiet=False)
    except Exception as e:
        raise RuntimeError(f"Folder download failed ({type(e).__name__}: {e}). {MANUAL_HINT}")
    found = next(FOLDER_TMP.rglob("images.zip"), None)
    if found is None:
        raise RuntimeError(f"images.zip not found in the Drive folder. {MANUAL_HINT}")
    shutil.move(str(found), ZIP_PATH)
    shutil.rmtree(FOLDER_TMP, ignore_errors=True)


def download_images() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if IMAGES_DIR.exists() and any(IMAGES_DIR.glob("*.jpg")):
        return IMAGES_DIR

    if not ZIP_PATH.exists():
        print("downloading images.zip (~470 MB)")
        _fetch_zip()

    print("extracting")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        for member in z.namelist():
            if member.lower().endswith(".jpg"):
                # drop any folder prefix so files land directly in data/images/
                with z.open(member) as src, open(IMAGES_DIR / Path(member).name, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    print(f"extracted {len(list(IMAGES_DIR.glob('*.jpg')))} images to {IMAGES_DIR}")
    ZIP_PATH.unlink()  # remove the zip to save space; delete this line to keep it
    return IMAGES_DIR


if __name__ == "__main__":
    download_images()