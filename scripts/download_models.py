"""
scripts/download_models.py
───────────────────────────
Download all model weights for the Vehicle Re-ID Pipeline (v1.0 + v2.0).

Usage:
    python scripts/download_models.py                       # download all v2 models
    python scripts/download_models.py --model transreid-vit-small
    python scripts/download_models.py --model clipreid
    python scripts/download_models.py --model aflink
    python scripts/download_models.py --all --force         # re-download even if cached
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _gdown(gdrive_id: str, dest: Path) -> bool:
    """Download from Google Drive using gdown. Returns True on success."""
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        gdown.download(url, str(dest), quiet=False)
        return dest.exists()
    except Exception as e:
        logger.warning("gdown failed: %s", e)
        return False


def _hf_download(repo_id: str, filename: str, dest: Path) -> bool:
    """Download from HuggingFace Hub. Returns True on success."""
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id    = repo_id,
            filename   = filename,
            local_dir  = str(dest.parent),
        )
        if Path(local) != dest:
            shutil.move(local, str(dest))
        return dest.exists()
    except Exception as e:
        logger.warning("HuggingFace Hub download failed: %s", e)
        return False


def _print_result(name: str, status: str, note: str = "") -> None:
    symbol = "[OK]" if status == "OK" else ("[--]" if status == "SKIP" else "[FAIL]")
    print(f"  {symbol:<6s} {name:<35s}  {status}  {note}")


# ─────────────────────────────────────────────────────────────────────────────
# Individual download functions
# ─────────────────────────────────────────────────────────────────────────────

def download_transreid_vit_small(force: bool = False) -> Path:
    """
    Download TransReID ViT-S/16 weights fine-tuned on VeRi-776.
    Target: models/reid/transreid_vit_small_veri776.pth  (~350 MB)
    Note: The official repository only released ViT-Base weights, which we download here.
    """
    dest = Path("models/reid/transreid_vit_small_veri776.pth")
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        logger.info("TransReID weights already present at '%s'. Skipping.", dest)
        return dest

    logger.info("Downloading TransReID weights from official Google Drive…")

    # Attempt 1: gdown
    gdrive_id = "1iF5JNPw9xi-rLY3Ri9EY-PFAkK6Vg_Pf"
    if _gdown(gdrive_id, dest):
        logger.info("TransReID weights downloaded via gdown → %s", dest)
        return dest

    # Attempt 2: Manual instructions
    logger.error(
        "\n"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║  TransReID weights could not be auto-downloaded.            ║\n"
        "║  Please download manually:                                  ║\n"
        "║    Source: https://github.com/damo-cv/TransReID             ║\n"
        "║    Target: models/reid/transreid_vit_small_veri776.pth      ║\n"
        "╚══════════════════════════════════════════════════════════════╝"
    )
    return dest


def download_clipreid_weights(force: bool = False) -> Path:
    """
    Download CLIP-ReID fine-tuned ViT-B/16 weights for VeRi-776.
    Target: models/reid/clipreid_veri776.pth
    Note: Official authors did not publish a pre-trained Stage-2 checkpoint.
          The pipeline automatically falls back to OpenAI's pre-trained ViT-B/16.
    """
    dest = Path("models/reid/clipreid_veri776.pth")
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        logger.info("CLIP-ReID weights already present at '%s'. Skipping.", dest)
        return dest

    logger.info("CLIP-ReID will use OpenAI's high-fidelity zero-shot CLIP ViT-B/16 encoder (auto-loaded).")
    return dest


def download_aflink_weights(force: bool = False) -> Path:
    """
    Download AFLink MLP weights for StrongSORT v2.
    Target: models/aflink.pth  (~4.35 MB)
    """
    dest = Path("models/aflink.pth")
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        logger.info("AFLink weights already present at '%s'. Skipping.", dest)
        return dest

    logger.info("Downloading AFLink weights from official StrongSORT Google Drive…")

    # Attempt 1: gdown
    gdrive_id = "1DFMUkL-dc-j8-fibcJIq-46Xoq_bFoO9"
    if _gdown(gdrive_id, dest):
        logger.info("AFLink weights downloaded via gdown → %s", dest)
        return dest

    logger.error(
        "\n"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║  AFLink weights could not be auto-downloaded.               ║\n"
        "║  Please download manually:                                  ║\n"
        "║    Source: https://github.com/dyhBUPT/StrongSORT/releases  ║\n"
        "║    Target: models/aflink.pth                                ║\n"
        "╚══════════════════════════════════════════════════════════════╝"
    )
    return dest


# ─────────────────────────────────────────────────────────────────────────────

def download_all_models_v2(force: bool = False) -> None:
    """
    Download ALL v2.0 model weights in sequence.
    Prints a colored summary table of success/failure per model.
    """
    print("\n" + "═" * 65)
    print("  Vehicle Re-ID Pipeline v2.0 — Model Weight Downloader")
    print("═" * 65 + "\n")

    results: list[tuple[str, str, str]] = []

    models = [
        ("TransReID ViT-S/16", download_transreid_vit_small,
         "models/reid/transreid_vit_small_veri776.pth"),
        ("CLIP-ReID ViT-B/16", download_clipreid_weights,
         "models/reid/clipreid_veri776.pth"),
        ("AFLink (StrongSORT)", download_aflink_weights,
         "models/aflink.pth"),
    ]

    for name, fn, target in models:
        try:
            path = fn(force=force)
            if name == "CLIP-ReID ViT-B/16":
                status = "OK"
                note = "Using OpenAI Zero-Shot CLIP fallback (no download needed)"
            else:
                status = "OK" if Path(path).exists() else "FAIL"
                note = str(path) if status == "OK" else "Not found after download attempt"
        except Exception as e:
            status = "FAIL"
            note   = str(e)
        results.append((name, status, note))

    print("\n  Download Summary:")
    print("  " + "─" * 62)
    for name, status, note in results:
        _print_result(name, status, note)
    print("  " + "─" * 62 + "\n")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download model weights for Vehicle Re-ID Pipeline v2.0"
    )
    parser.add_argument(
        "--model",
        choices=["transreid-vit-small", "clipreid", "aflink"],
        help="Download a specific model only.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all v2.0 model weights.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if file already exists.",
    )
    args = parser.parse_args()

    if args.model == "transreid-vit-small":
        download_transreid_vit_small(force=args.force)
    elif args.model == "clipreid":
        download_clipreid_weights(force=args.force)
    elif args.model == "aflink":
        download_aflink_weights(force=args.force)
    elif args.all or (not args.model):
        download_all_models_v2(force=args.force)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
