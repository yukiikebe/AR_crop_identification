#!/usr/bin/env python3
# copy_cdl.py

from pathlib import Path
import re
import os
import shutil

src_root = Path("/mnt/data/mzarvani/new_out/out/cdl")
dst_root = Path("/media/khoavo/Seagate Expansion Drive/2022_all_final")

PATTERN = re.compile(r"roi_mask_(\d+)_(\d+)\.tiff$")

for src in src_root.glob("roi_mask_*_*.tiff"):
    m = PATTERN.match(src.name)
    if not m:
        print(f"⚠️  skipped {src.name} (name doesn’t match)")
        continue

    x, y = m.groups()                  # coordinate strings
    dst_dir = dst_root / f"{x}_{y}"
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst = dst_dir / "cdl.tiff"
    os.remove(dst)
    dst = dst_dir / f"cdl.tif"  # rename to avoid overwriting
    shutil.copy2(src, dst)             # metadata-preserving copy
    print(f"✓ copied {src.name} → {dst}")
