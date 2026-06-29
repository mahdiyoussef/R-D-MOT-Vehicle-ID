#!/usr/bin/env python3
"""
scripts/convert_detrac_to_yolo.py
───────────────────────────────────
Convert UA-DETRAC XML annotations to YOLO format (.txt label files).

Usage
-----
    python scripts/convert_detrac_to_yolo.py \\
        --xml-dir   data/raw/UA-DETRAC/DETRAC-Train-Annotations-XML \\
        --image-dir data/raw/UA-DETRAC/Insight-MVT_Annotation_Train \\
        --out-dir   data/processed

Output structure:
    data/processed/
    ├── train/images/<seq>/<frame>.jpg   (symlinks or copies)
    └── train/labels/<seq>/<frame>.txt
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

# UA-DETRAC class → unified class index
DETRAC_CLASS_MAP = {
    "car":   0,
    "van":   0,   # merge van into car
    "bus":   2,
    "others": -1, # skip
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert UA-DETRAC XML → YOLO TXT")
    p.add_argument("--xml-dir",   required=True, help="Dir with DETRAC XML files")
    p.add_argument("--image-dir", required=True, help="Dir with DETRAC image sequences")
    p.add_argument("--out-dir",   required=True, help="Output root (data/processed)")
    p.add_argument("--split",     default="train", choices=["train", "val", "test"])
    return p.parse_args()


def bbox_to_yolo(
    x: float, y: float, w: float, h: float,
    img_w: int, img_h: int,
) -> str:
    """Convert absolute DETRAC box (x, y, w, h) to YOLO relative format."""
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return f"{cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def convert_sequence(
    xml_path:  Path,
    image_dir: Path,
    out_img:   Path,
    out_lbl:   Path,
) -> int:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    seq_name = xml_path.stem
    seq_img_dir = image_dir / seq_name
    if not seq_img_dir.exists():
        print(f"  [skip] image dir not found: {seq_img_dir}")
        return 0

    # Image size from first frame attribute (or default 960x540)
    img_w, img_h = 960, 540
    size_el = root.find(".//ignored_region/..")  # parent element may have attrs
    # Try reading from sequence element
    seq_el = root.find("sequence") if root.tag != "sequence" else root
    if seq_el is not None:
        img_w = int(seq_el.get("sWidth", img_w))
        img_h = int(seq_el.get("sHeight", img_h))

    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    n_frames = 0
    for frame_el in root.iter("frame"):
        frame_num = int(frame_el.get("num", 0))
        frame_name = f"img{frame_num:05d}"

        # Find source image (jpg or jpeg)
        src_img = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = seq_img_dir / f"{frame_name}{ext}"
            if candidate.exists():
                src_img = candidate
                break

        if src_img is None:
            continue

        yolo_lines: list[str] = []
        for target in frame_el.findall(".//target"):
            cls_name = target.find("attribute").get("vehicle_type", "car") \
                if target.find("attribute") is not None else "car"
            cls_id = DETRAC_CLASS_MAP.get(cls_name.lower(), -1)
            if cls_id < 0:
                continue

            box = target.find("box")
            if box is None:
                continue

            bx = float(box.get("left",   0))
            by = float(box.get("top",    0))
            bw = float(box.get("width",  1))
            bh = float(box.get("height", 1))

            yolo_str = bbox_to_yolo(bx, by, bw, bh, img_w, img_h)
            yolo_lines.append(f"{cls_id} {yolo_str}")

        # Write label file (even if empty — YOLO expects it)
        lbl_file = out_lbl / f"{seq_name}_{frame_name}.txt"
        lbl_file.write_text("\n".join(yolo_lines))

        # Symlink or copy image
        dst_img = out_img / f"{seq_name}_{frame_name}{src_img.suffix}"
        if not dst_img.exists():
            dst_img.symlink_to(src_img.resolve())

        n_frames += 1

    return n_frames


def main() -> None:
    args = parse_args()

    xml_dir   = Path(args.xml_dir)
    image_dir = Path(args.image_dir)
    out_dir   = Path(args.out_dir)

    out_img = out_dir / args.split / "images"
    out_lbl = out_dir / args.split / "labels"

    xml_files = sorted(xml_dir.glob("*.xml"))
    print(f"Found {len(xml_files)} XML files in '{xml_dir}'")

    total = 0
    for xml_path in xml_files:
        n = convert_sequence(xml_path, image_dir, out_img, out_lbl)
        print(f"  {xml_path.stem:<30} → {n:5d} frames")
        total += n

    print(f"\nDone. Total frames: {total}")
    print(f"Labels: {out_lbl}")
    print(f"Images: {out_img}")


if __name__ == "__main__":
    main()
