#!/usr/bin/env python3
"""
scripts/download_datasets.py
─────────────────────────────
Automated dataset downloader for the Vehicle Persistent ReID System using Kaggle API.

AUTOMATIC (via Kaggle CLI):
   VisDrone2019-DET  (Detection)
   UA-DETRAC         (Detection/Tracking)
   BDD100K           (Detection)
   VeRi-776          (Re-Identification)

Usage
-----
    # Ensure you have your kaggle.json in ~/.kaggle/kaggle.json
    
    # Download everything:
    python scripts/download_datasets.py --all

    # Individual datasets:
    python scripts/download_datasets.py --visdrone
    python scripts/download_datasets.py --detrac
    python scripts/download_datasets.py --veri
    python scripts/download_datasets.py --bdd100k
    python scripts/download_datasets.py --yolo-weights
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# ── Colour helpers ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"{GREEN}    {msg}{RESET}")
def warn(msg): print(f"{YELLOW}     {msg}{RESET}")
def err(msg):  print(f"{RED}    {msg}{RESET}")
def info(msg): print(f"{CYAN}     {msg}{RESET}")
def hdr(msg):  print(f"\n{BOLD}{msg}{RESET}\n" + "─" * 60)

# ── Kaggle Slugs ───────────────────────────────────────────────────────────────
KAGGLE_DATASETS = {
    "visdrone": {
        "slug": "banuprasadb/visdrone-dataset",
        "dir": "VisDrone2019-DET"
    },
    "detrac": {
        "slug": "bratjay/ua-detrac-orig",
        "dir": "UA-DETRAC"
    },
    "veri": {
        "slug": "abhyudaya12/veri-vehicle-re-identification-dataset",
        "dir": "VeRi-776"
    },
    "bdd100k": {
        "slug": "awsaf49/bdd100k-dataset",
        "dir": "BDD100K"
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight Check
# ─────────────────────────────────────────────────────────────────────────────
def check_kaggle_auth():
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        err("Kaggle API token not found!")
        print(f"""
  Please set up your Kaggle API credentials:
  1. Go to https://www.kaggle.com/settings
  2. Click 'Create New Token' to download kaggle.json
  3. Move it to ~/.kaggle/kaggle.json and set permissions:
     mkdir -p ~/.kaggle
     mv ~/Downloads/kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
""")
        sys.exit(1)
        
    try:
        import kaggle
    except ImportError:
        warn("kaggle package not found — installing …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])

# ─────────────────────────────────────────────────────────────────────────────
# Kaggle Downloader
# ─────────────────────────────────────────────────────────────────────────────
def download_kaggle_dataset(name: str, raw_dir: Path) -> None:
    ds_info = KAGGLE_DATASETS[name]
    slug = ds_info["slug"]
    out_dir = raw_dir / ds_info["dir"]
    
    hdr(f"Downloading {name.upper()} Dataset via Kaggle")
    info(f"Kaggle Slug: {slug}")
    
    if out_dir.exists() and any(out_dir.iterdir()):
        ok(f"{ds_info['dir']} already exists and is not empty — skipping.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    
    info(f"Downloading dataset (this may take a while)...")
    try:
        import kaggle
        # Download and unzip directly
        kaggle.api.dataset_download_files(slug, path=str(out_dir), unzip=True)
        ok(f"{name.upper()} dataset downloaded and extracted to: {out_dir}")
    except Exception as e:
        err(f"Download failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# YOLO base weights (automatic via ultralytics)
# ─────────────────────────────────────────────────────────────────────────────

def download_yolo_weights() -> None:
    hdr("YOLO Base Weights")
    try:
        from ultralytics import YOLO
    except ImportError:
        warn("ultralytics not found — skipping YOLO weights.")
        return

    weights_dir = Path("models/yolo")
    weights_dir.mkdir(parents=True, exist_ok=True)

    for variant in ["yolo11m.pt", "yolo11n.pt"]:
        dest = weights_dir / variant
        if dest.exists():
            ok(f"{variant} already present — skipping.")
            continue
        info(f"Downloading {variant} …")
        model = YOLO(variant)   # auto-downloads to ~/.ultralytics/assets/
        cache_path = Path.home() / ".ultralytics" / "assets" / variant
        if cache_path.exists():
            shutil.copy(cache_path, dest)
            ok(f"{variant} → {dest}")
        else:
            ok(f"{variant} downloaded (cached by ultralytics)")

# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def print_status(raw_dir: Path) -> None:
    hdr("Dataset Status")
    checks = [
        ("VisDrone2019-DET",           raw_dir / "VisDrone2019-DET"),
        ("UA-DETRAC",                  raw_dir / "UA-DETRAC"),
        ("BDD100K",                    raw_dir / "BDD100K"),
        ("VeRi-776",                   raw_dir / "VeRi-776"),
        ("YOLO yolo11m.pt",            Path("models/yolo/yolo11m.pt")),
        ("YOLO yolo11n.pt",            Path("models/yolo/yolo11n.pt")),
    ]
    for label, path in checks:
        if path.exists() and any(path.iterdir()):
            ok(f"{label:<35}  {path}")
        else:
            warn(f"{label:<35}  NOT FOUND")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dataset downloader for Vehicle Persistent ReID System using Kaggle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--all",           action="store_true", help="Download all datasets")
    p.add_argument("--visdrone",      action="store_true", help="Download VisDrone2019-DET")
    p.add_argument("--detrac",        action="store_true", help="Download UA-DETRAC")
    p.add_argument("--veri",          action="store_true", help="Download VeRi-776")
    p.add_argument("--bdd100k",       action="store_true", help="Download BDD100K")
    p.add_argument("--yolo-weights",  action="store_true", help="Download YOLO base weights")
    p.add_argument("--status",        action="store_true", help="Check which datasets are present")
    p.add_argument("--raw-dir",       default="data/raw",  help="Root directory for raw datasets")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not any([
        args.all, args.visdrone, args.detrac, args.veri, 
        args.bdd100k, args.yolo_weights, args.status
    ]):
        args.status = True

    if args.all or args.visdrone or args.detrac or args.veri or args.bdd100k:
        check_kaggle_auth()

    if args.all or args.yolo_weights:
        download_yolo_weights()

    if args.all or args.visdrone:
        download_kaggle_dataset("visdrone", raw_dir)

    if args.all or args.detrac:
        download_kaggle_dataset("detrac", raw_dir)

    if args.all or args.veri:
        download_kaggle_dataset("veri", raw_dir)
        
    if args.all or args.bdd100k:
        download_kaggle_dataset("bdd100k", raw_dir)

    if args.status or args.all:
        print_status(raw_dir)

if __name__ == "__main__":
    main()
