#!/usr/bin/env python3
"""
scripts/convert_bdd100k_to_yolo.py
─────────────────────────────────────
Convert BDD100K detection JSON annotations to YOLO format.

Usage
-----
    python scripts/convert_bdd100k_to_yolo.py \\
        --json-dir  data/raw/BDD100K/labels/det_20/det_train.json \\
        --image-dir data/raw/BDD100K/images/100k/train \\
        --out-dir   data/processed

BDD100K → unified class mapping (vehicles only):
    car           → 0 (car)
    truck         → 1 (truck)
    bus           → 2 (bus)
    motor         → 3 (motorcycle)
    bike          → 4 (bicycle)
    (all others skipped)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BDD_CLASS_MAP: dict[str, int] = {
    "car":   0,
    "truck": 1,
    "bus":   2,
    "motor": 3,
    "bike":  4,
}

IMG_W, IMG_H = 1280, 720  # BDD100K standard resolution


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert BDD100K JSON → YOLO TXT")
    p.add_argument("--json-file",  required=True, help="Path to BDD100K JSON annotation file")
    p.add_argument("--image-dir",  required=True, help="Dir with BDD100K images")
    p.add_argument("--out-dir",    required=True, help="Output root (data/processed)")
    p.add_argument("--split",      default="train", choices=["train", "val", "test"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    json_file = Path(args.json_file)
    image_dir = Path(args.image_dir)
    out_dir   = Path(args.out_dir)

    out_img = out_dir / args.split / "images"
    out_lbl = out_dir / args.split / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    with open(json_file) as f:
        annotations = json.load(f)

    print(f"Processing {len(annotations)} BDD100K frames …")
    skipped = 0
    total   = 0

    for entry in annotations:
        filename = entry["name"]            # e.g. "b1c9c847-3bda4659.jpg"
        stem     = Path(filename).stem

        # Skip frames without detection labels
        labels = entry.get("labels") or []
        vehicle_labels = [
            lb for lb in labels
            if lb.get("category") in BDD_CLASS_MAP
        ]

        if not vehicle_labels:
            skipped += 1
            continue

        src_img = image_dir / filename
        if not src_img.exists():
            skipped += 1
            continue

        yolo_lines: list[str] = []
        for lb in vehicle_labels:
            cls_id = BDD_CLASS_MAP[lb["category"]]
            box2d  = lb.get("box2d", {})
            x1 = float(box2d.get("x1", 0))
            y1 = float(box2d.get("y1", 0))
            x2 = float(box2d.get("x2", 1))
            y2 = float(box2d.get("y2", 1))

            cx = ((x1 + x2) / 2) / IMG_W
            cy = ((y1 + y2) / 2) / IMG_H
            nw = (x2 - x1) / IMG_W
            nh = (y2 - y1) / IMG_H

            yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        (out_lbl / f"{stem}.txt").write_text("\n".join(yolo_lines))

        dst_img = out_img / filename
        if not dst_img.exists():
            dst_img.symlink_to(src_img.resolve())

        total += 1

    print(f"Done.  Converted: {total}  Skipped: {skipped}")


if __name__ == "__main__":
    main()
