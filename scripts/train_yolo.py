#!/usr/bin/env python3
"""
scripts/train_yolo.py
──────────────────────
Fine-tune YOLOv11m on the processed vehicle dataset.

Usage
-----
    python scripts/train_yolo.py
    python scripts/train_yolo.py --model yolo11n.pt --epochs 30 --batch 32
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune YOLO for vehicle detection")
    p.add_argument("--model",        default="yolo11m.pt",            help="Base weights")
    p.add_argument("--data",         default="configs/yolo_finetune.yaml")
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--imgsz",        type=int,   default=640)
    p.add_argument("--batch",        type=int,   default=16)
    p.add_argument("--lr0",          type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0005)
    p.add_argument("--device",       default="0",                      help="'0' for GPU, 'cpu'")
    p.add_argument("--project",      default="runs/detect")
    p.add_argument("--name",         default="vehicle_v1")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data         = args.data,
        epochs       = args.epochs,
        imgsz        = args.imgsz,
        batch        = args.batch,
        lr0          = args.lr0,
        weight_decay = args.weight_decay,
        device       = args.device,
        project      = args.project,
        name         = args.name,
        # Augmentations (from spec)
        mosaic       = True,
        hsv_h        = 0.015,
        hsv_s        = 0.7,
        hsv_v        = 0.4,
        flipud       = 0.0,
        fliplr       = 0.5,
        degrees      = 5.0,
        translate    = 0.1,
        scale        = 0.5,
        perspective  = 0.001,
    )

    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    
    if Path("/kaggle/working").exists():
        out_pt = Path("/kaggle/working/models/yolo/yolo11m_vehicle.pt")
    else:
        out_pt = Path("models/yolo/yolo11m_vehicle.pt")
        
    out_pt.parent.mkdir(parents=True, exist_ok=True)

    if best_pt.exists():
        import shutil
        shutil.copy(best_pt, out_pt)
        print(f"\nBest weights copied to: {out_pt}")
    else:
        print(f"\nTraining complete. Weights at: {best_pt}")


if __name__ == "__main__":
    main()
