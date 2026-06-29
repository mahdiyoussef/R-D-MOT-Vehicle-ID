import os
import shutil
from pathlib import Path

src = Path('outputs/dino_lora/dinov2_lora_veri776.pth')
dst_dir = Path('models/reid')
dst = dst_dir / 'dinov2_lora_veri776.pth'

dst_dir.mkdir(parents=True, exist_ok=True)
if src.exists():
    shutil.move(str(src), str(dst))
    print(f"Successfully moved {src} to {dst}")
else:
    print(f"Source file {src} does not exist. Already moved?")
