"""Copy one fold of the DeepWeeds split shipped in the Kaggle zip into data/.

The Kaggle mirror (kaggle.com/datasets/imsparsh/deepweeds) has the authors' five
folds in labels/{train,val,test}_subset{0-4}.csv (60/20/20, columns Filename,
Label; identical to github.com/AlexOlsen/DeepWeeds/tree/master/labels). This
keeps each file's rows and order and adds the Species column the loaders use,
taken from the official labels.csv.

Labels come from labels.csv. The fold files disagree with it on one image:
20170714-110407-3.jpg is Label 0 (Chinee apple) in the fold files but Lantana
in labels.csv, which every other experiment uses. Mismatches are printed.

    python -m src.data.kaggle_split <unzipped Kaggle labels/ folder> --fold 0
    # -> data/splits_kaggle0/{train,val,test}.csv
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def convert_fold(kaggle_labels_dir, fold: int, labels_csv, out_dir) -> dict[str, pd.DataFrame]:
    species = pd.read_csv(labels_csv).set_index("Filename")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {}
    for name in ["train", "val", "test"]:
        df = pd.read_csv(Path(kaggle_labels_dir) / f"{name}_subset{fold}.csv")
        official = species.loc[df["Filename"]]
        for f, theirs, ours, sp in zip(df["Filename"], df["Label"], official["Label"], official["Species"]):
            if theirs != ours:
                print(f"  {name}: {f} is Label {theirs} in the fold file, {ours} ({sp}) in labels.csv; using labels.csv")
        df["Label"] = official["Label"].to_numpy()
        df["Species"] = official["Species"].to_numpy()
        df.to_csv(out_dir / f"{name}.csv", index=False)
        splits[name] = df
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kaggle_labels_dir", help="the labels/ folder from the Kaggle zip")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    out_dir = ROOT / "data" / f"splits_kaggle{args.fold}"
    splits = convert_fold(args.kaggle_labels_dir, args.fold, ROOT / "data" / "labels.csv", out_dir)
    for name, df in splits.items():
        print(f"{name:5s} {len(df):5d} images -> {out_dir / f'{name}.csv'}")


if __name__ == "__main__":
    main()
