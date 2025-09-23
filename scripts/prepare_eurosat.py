import argparse
from pathlib import Path
import shutil
import random

from torchvision.datasets import EuroSAT

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data", help="data root")
    ap.add_argument("--out", type=str, default="data/eurosat_custom", help="output folder")
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=41)
    args = ap.parse_args()

    random.seed(args.seed)
    root = Path(args.root)
    out = Path(args.out)

    # 1) Download EuroSAT (RGB)
    ds = EuroSAT(root=root, download=True)
    src = Path(ds.root) / "eurosat" / "2750"  # torchvision’s EuroSAT layout (class folders)

    # 2) Collect class folders
    classes = sorted([p.name for p in src.iterdir() if p.is_dir()])
    print("Classes:", classes)

    # 3) Create split folders
    for split in ["train", "val", "test"]:
        for c in classes:
            (out / split / c).mkdir(parents=True, exist_ok=True)

    # 4) Stratified split inside each class folder
    for c in classes:
        imgs = sorted((src / c).glob("*.jpg"))
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * args.train)
        n_val   = int(n * args.val)
        n_test  = n - n_train - n_val
        split_map = (
            (imgs[:n_train], "train"),
            (imgs[n_train:n_train+n_val], "val"),
            (imgs[n_train+n_val:], "test"),
        )
        for subset, split in split_map:
            for p in subset:
                shutil.copy2(p, out / split / c / p.name)

    print("Done. Output:", out.resolve())

if __name__ == "__main__":
    main()