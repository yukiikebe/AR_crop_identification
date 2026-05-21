#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds


RESULT_RE = re.compile(r"^(?P<step>\d+)_(?P<idx>\d+)_sr\.(?:png|jpg|jpeg|tif|tiff)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FastDiffSR infer.py on georeferenced Sentinel patches and convert outputs back to GeoTIFF."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--infer-script", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--reconstruction-out", type=Path, required=True)
    parser.add_argument("--gpu-ids", default=None, help="Passed to infer.py --gpu_ids, e.g. 0 or 0,1.")
    parser.add_argument("--skip-infer", action="store_true", help="Only convert existing infer.py results.")
    return parser.parse_args()


def load_kept_manifest(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("status") == "kept" and payload.get("sample_id"):
                records.append(payload)
    records.sort(key=lambda item: str(item["sample_id"]))
    return records


def write_infer_config(template_path: Path, *, dataset_root: Path, results_dir: Path) -> Path:
    config = json.loads(template_path.read_text(encoding="utf-8"))
    config["phase"] = "val"
    config.setdefault("path", {})
    config["path"]["results"] = str(results_dir.resolve())
    config.setdefault("datasets", {})
    config.setdefault("datasets", {}).setdefault("val", {})
    config["datasets"]["val"]["dataroot"] = str(dataset_root.resolve())
    config["datasets"]["val"]["datatype"] = "img"
    config["datasets"]["val"]["l_resolution"] = int(config["datasets"]["val"].get("l_resolution", 64))
    config["datasets"]["val"]["r_resolution"] = int(config["datasets"]["val"].get("r_resolution", 256))
    config["datasets"]["val"]["data_len"] = -1
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", prefix="fastdiffsr_infer_", delete=False)
    with tmp:
        json.dump(config, tmp, indent=4)
    return Path(tmp.name)


def run_infer(infer_script: Path, config_path: Path, gpu_ids: str | None) -> None:
    infer_script = infer_script.resolve()
    fastdiffsr_root = infer_script.parent
    cmd = [os.environ.get("PYTHON", sys.executable), str(infer_script), "--config", str(config_path), "--phase", "val"]
    if gpu_ids:
        cmd.extend(["--gpu_ids", gpu_ids])
    subprocess.run(cmd, cwd=str(fastdiffsr_root), check=True)


def collect_results(results_dir: Path) -> dict[int, Path]:
    result_paths: dict[int, Path] = {}
    for path in results_dir.iterdir():
        if not path.is_file():
            continue
        match = RESULT_RE.match(path.name)
        if not match:
            continue
        idx = int(match.group("idx"))
        result_paths[idx] = path
    return result_paths


def save_reconstruction_geotiff(path: Path, img_uint8: np.ndarray, *, crs: str | None, bounds: list[float]) -> None:
    transform = from_bounds(*[float(v) for v in bounds], img_uint8.shape[1], img_uint8.shape[0])
    profile = {
        "driver": "GTiff",
        "height": int(img_uint8.shape[0]),
        "width": int(img_uint8.shape[1]),
        "count": 3,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "compress": "DEFLATE",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.moveaxis(np.ascontiguousarray(img_uint8), -1, 0))


def convert_results(records: list[dict], *, results_dir: Path, reconstruction_out: Path) -> int:
    results = collect_results(results_dir)
    if len(results) < len(records):
        raise RuntimeError(f"Expected at least {len(records)} result images, found {len(results)} in {results_dir}")

    manifest_out = reconstruction_out.parent / "reconstruction_manifest.jsonl"
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with manifest_out.open("w", encoding="utf-8") as manifest_fp:
        for data_index, record in enumerate(records, start=1):
            result_path = results.get(data_index)
            if result_path is None:
                raise RuntimeError(f"Missing FastDiffSR result for dataset index {data_index}: expected *_{data_index}_sr.png")
            sample_id = str(record["sample_id"])
            img = Image.open(result_path).convert("RGB")
            img_uint8 = np.asarray(img, dtype=np.uint8)
            out_path = reconstruction_out / f"{sample_id}.tif"
            save_reconstruction_geotiff(
                out_path,
                img_uint8,
                crs=record.get("crs"),
                bounds=record["bounds"],
            )
            payload = dict(record)
            payload.update(
                {
                    "kind": "reconstruction",
                    "status": "kept",
                    "infer_result_path": str(result_path),
                    "reconstruction_path": str(out_path),
                    "reconstruction_width": int(img_uint8.shape[1]),
                    "reconstruction_height": int(img_uint8.shape[0]),
                }
            )
            manifest_fp.write(json.dumps(payload, sort_keys=True) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    records = load_kept_manifest(args.manifest)
    if not records:
        raise RuntimeError(f"No kept patches found in {args.manifest}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_infer:
        if any(args.results_dir.iterdir()):
            shutil.rmtree(args.results_dir)
            args.results_dir.mkdir(parents=True, exist_ok=True)
        config_path = write_infer_config(
            args.config,
            dataset_root=args.dataset_root,
            results_dir=args.results_dir,
        )
        try:
            run_infer(args.infer_script, config_path, args.gpu_ids)
        finally:
            try:
                config_path.unlink()
            except FileNotFoundError:
                pass

    converted = convert_results(records, results_dir=args.results_dir, reconstruction_out=args.reconstruction_out)
    print(f"[done] converted {converted} reconstructions to {args.reconstruction_out}")


if __name__ == "__main__":
    main()
