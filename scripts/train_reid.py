#!/usr/bin/env python3
"""
scripts/train_reid.py
──────────────────────
Fine-tune OSNet on VeRi-776 for vehicle re-identification.

Usage
-----
    python scripts/train_reid.py
    python scripts/train_reid.py --config configs/reid_train.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train OSNet ReID on VeRi-776")
    p.add_argument("--config", default="configs/reid_train.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    try:
        import torchreid
    except ImportError:
        raise SystemExit("torchreid is not installed. Run: pip install torchreid")

    model_cfg = cfg["model"]
    data_cfg  = cfg["data"]
    opt_cfg   = cfg["optimizer"]
    sch_cfg   = cfg["scheduler"]
    train_cfg = cfg["training"]

    # ── Data ─────────────────────────────────────────────────────────────────
    datamanager = torchreid.data.ImageDataManager(
        root             = data_cfg["root"],
        sources          = data_cfg["sources"],
        targets          = data_cfg["targets"],
        height           = data_cfg["height"],
        width            = data_cfg["width"],
        batch_size_train = data_cfg["batch_size_train"],
        batch_size_test  = data_cfg["batch_size_test"],
        transforms       = data_cfg["transforms"],
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = torchreid.models.build_model(
        name        = model_cfg["name"],
        num_classes = datamanager.num_train_pids,
        loss        = model_cfg["loss"],
        pretrained  = model_cfg["pretrained"],
    )

    # ── Optimiser + Scheduler ─────────────────────────────────────────────────
    optimizer = torchreid.optim.build_optimizer(
        model,
        optim        = opt_cfg["name"],
        lr           = opt_cfg["lr"],
        weight_decay = opt_cfg.get("weight_decay", 0.0005),
    )
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler = sch_cfg["name"],
        stepsize     = sch_cfg["stepsize"],
    )

    # ── Engine ────────────────────────────────────────────────────────────────
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer    = optimizer,
        scheduler    = scheduler,
        label_smooth = train_cfg.get("label_smooth", True),
    )

    if Path("/kaggle/working").exists():
        save_dir = Path("/kaggle/working/models/reid")
    else:
        save_dir = Path(train_cfg["save_dir"])
        
    save_dir.mkdir(parents=True, exist_ok=True)

    engine.run(
        save_dir   = str(save_dir),
        max_epoch  = train_cfg["max_epoch"],
        eval_freq  = train_cfg["eval_freq"],
        print_freq = train_cfg["print_freq"],
        test_only  = False,
    )

    print(f"\nReID training complete. Checkpoints saved in: {save_dir}")
    print(f"Rename best checkpoint to '{save_dir}/osnet_veri776.pth' before running the pipeline.")


if __name__ == "__main__":
    main()
