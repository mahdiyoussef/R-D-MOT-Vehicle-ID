#!/usr/bin/env python3
"""
scripts/convert_visdrone_to_yolo.py
─────────────────────────────────────
Convert VisDrone2019-DET annotations to YOLO format.

VisDrone annotation format (per line in .txt):
  <x_min>, <y_min>, <w>, <h>, <score>, <category>, <truncation>, <occlusion>

  Categories:
    0: ignored       1: pedestrian    2: people        3: bicycle
    4: car           5: van           6: truck         7: tricycle
    8: awning-tricycle 9: bus         10: motor        11: others

Vehicle-only mapping → unified classes:
    4  (car)    → 0
    5  (van)    → 0  (merge into car)
    6  (truck)  → 1
    9  (bus)    → 2
    10 (motor)  → 3
    3  (bicycle)→ 4

Usage
-----
    python scripts/convert_visdrone_to_yolo.py \\
        --src data/raw/VisDrone2019-DET \\
        --out data/processed \\
        --split train

    python scripts/convert_visdrone_to_yolo.py \\
        --src data/raw/VisDrone2019-DET \\
        --out data/processed \\
        --split val
"""

from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image

# VisDrone category → unified vehicle class (None = skip)
CATEGORY_MAP: dict[int, int | None] = {
    0:  None,   # ignored region
    1:  None,   # pedestrian
    2:  None,   # people
    3:  4,      # bicycle
    4:  0,      # car
    5:  0,      # van → car
    6:  1,      # truck
    7:  None,   # tricycle
    8:  None,   # awning-tricycle
    9:  2,      # bus
    10: 3,      # motor → motorcycle
    11: None,   # others
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VisDrone → YOLO format converter")
    p.add_argument("--src",   default="data/raw/VisDrone2019-DET",
                   help="Root dir of VisDrone (contains VisDrone2019-DET-train/ etc.)")
    p.add_argument("--out",   default="data/processed", help="Output root")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    return p.parse_args()


def convert_split(src_root: Path, out_root: Path, split: str) -> None:
    # VisDrone split folder names
    split_map = {
        "train": "VisDrone2019-DET-train",
        "val":   "VisDrone2019-DET-val",
        "test":  "VisDrone2019-DET-test-dev",
    }
    split_dir = src_root / split_map[split]

    if not split_dir.exists():
        print(f"[skip] Split directory not found: {split_dir}")
        return

    img_src  = split_dir / "images"
    ann_src  = split_dir / "annotations"

    out_img = out_root / split / "images"
    out_lbl = out_root / split / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    img_files = sorted(img_src.glob("*.jpg")) + sorted(img_src.glob("*.png"))
    print(f"Processing {split}: {len(img_files)} images in {split_dir.name}")

    converted = skipped = 0

    for img_path in img_files:
        ann_path = ann_src / (img_path.stem + ".txt")
        if not ann_path.exists():
            skipped += 1
            continue

        # Get image dimensions without loading full image into RAM
        with Image.open(img_path) as img:
            img_w, img_h = img.size

        yolo_lines: list[str] = []
        for raw_line in ann_path.read_text().strip().splitlines():
            parts = raw_line.strip().split(",")
            if len(parts) < 6:
                continue

            x, y, w, h = [int(v) for v in parts[:4]]
            score       = int(parts[4])
            category    = int(parts[5])

            # Skip ignored (score==0) and non-vehicle categories
            if score == 0:
                continue
            cls_id = CATEGORY_MAP.get(category)
            if cls_id is None:
                continue

            # Skip degenerate boxes
            if w <= 0 or h <= 0:
                continue

            # Convert to YOLO (cx, cy, nw, nh) — all normalised
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h

            # Clamp to [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        # Write label (empty file = background-only image, still valid for YOLO)
        lbl_out = out_lbl / (img_path.stem + ".txt")
        lbl_out.write_text("\n".join(yolo_lines))

        # Symlink image
        img_out = out_img / img_path.name
        if not img_out.exists():
            img_out.symlink_to(img_path.resolve())

        converted += 1

    print(f"  Done. Converted: {converted}  Skipped (no annotation): {skipped}")
    print(f"  Labels → {out_lbl}")
    print(f"  Images → {out_img}")


def main() -> None:
    args = parse_args()
    convert_split(Path(args.src), Path(args.out), args.split)


if __name__ == "__main__":
    main()
