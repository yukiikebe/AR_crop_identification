#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shlex
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from build_fastdiffsr_planet_ortho_dataset import (
    PLANET_DEFAULT_ROOT,
    collect_planet_scenes,
    describe_planet_scene,
    format_bounds,
    serialize_date,
    transform_bounds_safe,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WINDOWS_TSV = Path(__file__).with_name("sentinel_nearby_windows_pm5d.tsv")
DEFAULT_RAW_OUT_ROOT = Path(__file__).resolve().parent / "raw_sentinel"
STAGE_DAY_RADIUS = {
    "pm5": 5,
    "pm10": 10,
}


@dataclass(frozen=True)
class PlanetSceneRecord:
    scene_id: str
    source_side: str
    planet_date: date
    year_month: str
    planet_path: Path
    bbox_epsg4326: tuple[float, float, float, float]
    width: int
    height: int
    crs: str


@dataclass(frozen=True)
class WindowRecord:
    year_month: str
    source_sides: str
    window_days: int
    merged_window_index: int
    start_day: str
    end_day: str
    covered_planet_dates: tuple[str, ...]


@dataclass(frozen=True)
class SceneWindowTarget:
    scene: PlanetSceneRecord
    stage: str
    start_day: str
    end_day: str
    window_source: str
    window_days: int
    merged_window_index: int


@dataclass(frozen=True)
class PlannedRequest:
    stage: str
    request_mode: str
    request_id: str
    start_day: str
    end_day: str
    bbox_epsg4326: tuple[float, float, float, float]
    window_source: str
    window_days: int
    merged_window_index: int
    scenes: tuple[PlanetSceneRecord, ...]
    data_root: Path


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan Sentinel-2 downloads for Model 1 "
            "(Sentinel-2 RGB 16x16 -> Planet RGB 64x64). "
            "Default mode groups nearby Planet scenes that share the same date window "
            "to reduce redundant Sentinel downloads."
        )
    )
    parser.add_argument("--planet-root", type=Path, default=PLANET_DEFAULT_ROOT)
    parser.add_argument("--windows-tsv", type=Path, default=DEFAULT_WINDOWS_TSV)
    parser.add_argument("--project", default="satelite-430703")
    parser.add_argument("--raw-out-root", type=Path, default=DEFAULT_RAW_OUT_ROOT)
    parser.add_argument("--stage", choices=("pm5", "pm10", "month"), default="pm5")
    parser.add_argument("--retry-from", type=Path, default=None)
    parser.add_argument("--scene-manifest-tsv", type=Path, default=None)
    parser.add_argument("--plan-tsv", type=Path, default=None)
    parser.add_argument("--commands-sh", type=Path, default=None)
    parser.add_argument("--cloud-thresh", type=float, default=100.0)
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument(
        "--request-mode",
        choices=("grouped", "scene"),
        default="grouped",
        help=(
            "How to emit download requests. "
            "'grouped' merges nearby Planet scenes that share a Sentinel time window. "
            "'scene' preserves the old one-scene-per-request behavior."
        ),
    )
    parser.add_argument(
        "--max-group-width-deg",
        type=float,
        default=1.50,
        help="Maximum grouped request width in EPSG:4326 degrees before splitting.",
    )
    parser.add_argument(
        "--max-group-height-deg",
        type=float,
        default=1.50,
        help="Maximum grouped request height in EPSG:4326 degrees before splitting.",
    )
    parser.add_argument(
        "--merge-gap-deg",
        type=float,
        default=0.01,
        help="Treat bboxes within this gap as connected when grouping nearby scenes.",
    )
    parser.add_argument(
        "--max-scenes-per-request",
        type=int,
        default=64,
        help="Hard cap on the number of Planet scenes assigned to one grouped request.",
    )
    parser.add_argument(
        "--limit-scenes",
        type=int,
        default=None,
        help="Only plan the first N Planet scenes after sorting. Useful for dry runs.",
    )
    return parser.parse_args()


def default_scene_manifest_path() -> Path:
    return Path(__file__).with_name("planet_scene_manifest.tsv")


def default_plan_path(stage: str) -> Path:
    return Path(__file__).with_name(f"sentinel_download_plan_{stage}.tsv")


def default_commands_path(stage: str) -> Path:
    return Path(__file__).with_name(f"run_sentinel_downloads_{stage}.sh")


def detect_source_side(planet_root: Path, scene_path: Path) -> str:
    try:
        rel_parts = [part.lower() for part in scene_path.relative_to(planet_root).parts]
    except ValueError:
        rel_parts = []
    full_parts = [part.lower() for part in scene_path.parts]
    for parts in (rel_parts, full_parts):
        if "lower" in parts:
            return "lower"
        if "upper" in parts:
            return "upper"
    if rel_parts:
        return rel_parts[0]
    return "unknown"


def build_scene_id(source_side: str, scene_stem: str) -> str:
    prefix = source_side.lower().strip().replace(" ", "_") or "unknown"
    return f"{prefix}__{scene_stem}"


def bbox_to_csv(bounds: tuple[float, float, float, float]) -> str:
    return ",".join(f"{value:.6f}" for value in bounds)


def month_bounds_for_day(day_value: date) -> tuple[str, str]:
    month_start = day_value.replace(day=1)
    if month_start.month == 12:
        next_month = month_start.replace(year=month_start.year + 1, month=1)
    else:
        next_month = month_start.replace(month=month_start.month + 1)
    month_end = next_month - timedelta(days=1)
    return month_start.isoformat(), month_end.isoformat()


def load_retry_filter(path: Path) -> tuple[set[str], set[str]]:
    scene_ids: set[str] = set()
    planet_paths: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Retry file is missing a TSV header: {path}")
        for row in reader:
            scene_id = (row.get("scene_id") or "").strip()
            planet_path = (row.get("planet_path") or "").strip()
            if scene_id:
                scene_ids.add(scene_id)
            if planet_path:
                planet_paths.add(planet_path)
    if not scene_ids and not planet_paths:
        raise ValueError(f"No scene_id or planet_path rows found in retry TSV: {path}")
    return scene_ids, planet_paths


def load_window_records(path: Path) -> list[WindowRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Window TSV not found: {path}")

    rows: list[WindowRecord] = []
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        required = {
            "year_month",
            "source_sides",
            "window_days",
            "merged_window_index",
            "start_day",
            "end_day",
            "covered_planet_dates",
        }
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Window TSV must contain columns {sorted(required)}. Found: {reader.fieldnames}"
            )
        for row in reader:
            covered = tuple(
                item.strip()
                for item in (row["covered_planet_dates"] or "").split(",")
                if item.strip()
            )
            rows.append(
                WindowRecord(
                    year_month=row["year_month"].strip(),
                    source_sides=row["source_sides"].strip(),
                    window_days=int(row["window_days"]),
                    merged_window_index=int(row["merged_window_index"]),
                    start_day=row["start_day"].strip(),
                    end_day=row["end_day"].strip(),
                    covered_planet_dates=covered,
                )
            )
    return rows


def collect_planet_scene_records(planet_root: Path) -> list[PlanetSceneRecord]:
    scene_paths = collect_planet_scenes(planet_root)
    records: list[PlanetSceneRecord] = []
    skipped = 0
    for scene_path in scene_paths:
        described = describe_planet_scene(scene_path)
        if isinstance(described, tuple):
            skipped += 1
            continue
        bbox_epsg4326 = transform_bounds_safe(described.bounds, described.crs, "EPSG:4326")
        if bbox_epsg4326 is None:
            skipped += 1
            continue
        source_side = detect_source_side(planet_root, scene_path)
        records.append(
            PlanetSceneRecord(
                scene_id=build_scene_id(source_side, described.stem),
                source_side=source_side,
                planet_date=described.date,
                year_month=described.date.strftime("%Y-%m"),
                planet_path=scene_path,
                bbox_epsg4326=bbox_epsg4326,
                width=described.width,
                height=described.height,
                crs=str(described.crs),
            )
        )
    records.sort(key=lambda item: (item.planet_date, item.scene_id, str(item.planet_path)))
    print(f"[info] discovered usable_planet_scenes={len(records)} skipped={skipped}")
    return records


def source_sides_match(source_side: str, source_sides: str) -> bool:
    if source_sides == "both":
        return source_side in {"lower", "upper"}
    if source_side == "lower":
        return source_sides == "lower_only"
    if source_side == "upper":
        return source_sides == "upper_only"
    return False


def resolve_windows_for_scene(
    scene: PlanetSceneRecord,
    *,
    stage: str,
    window_records: list[WindowRecord],
) -> list[dict[str, str | int]]:
    if stage == "month":
        start_day, end_day = month_bounds_for_day(scene.planet_date)
        return [
            {
                "stage": stage,
                "start_day": start_day,
                "end_day": end_day,
                "window_source": "scene_month",
                "window_days": -1,
                "merged_window_index": 1,
            }
        ]

    if stage == "pm10":
        radius = STAGE_DAY_RADIUS[stage]
        return [
            {
                "stage": stage,
                "start_day": (scene.planet_date - timedelta(days=radius)).isoformat(),
                "end_day": (scene.planet_date + timedelta(days=radius)).isoformat(),
                "window_source": "scene_radius",
                "window_days": radius,
                "merged_window_index": 1,
            }
        ]

    matches: list[dict[str, str | int]] = []
    for row in window_records:
        if row.year_month != scene.year_month:
            continue
        if not source_sides_match(scene.source_side, row.source_sides):
            continue
        if scene.planet_date.isoformat() not in row.covered_planet_dates:
            continue
        matches.append(
            {
                "stage": stage,
                "start_day": row.start_day,
                "end_day": row.end_day,
                "window_source": "pm5_tsv",
                "window_days": row.window_days,
                "merged_window_index": row.merged_window_index,
            }
        )
    if matches:
        return matches

    radius = STAGE_DAY_RADIUS[stage]
    return [
        {
            "stage": stage,
            "start_day": (scene.planet_date - timedelta(days=radius)).isoformat(),
            "end_day": (scene.planet_date + timedelta(days=radius)).isoformat(),
            "window_source": "fallback_scene_radius",
            "window_days": radius,
            "merged_window_index": 1,
        }
    ]


def write_scene_manifest(path: Path, scenes: list[PlanetSceneRecord]) -> None:
    fieldnames = [
        "scene_id",
        "source_side",
        "planet_date",
        "year_month",
        "planet_path",
        "bbox_epsg4326",
        "bbox_epsg4326_pretty",
        "width",
        "height",
        "crs",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for scene in scenes:
            writer.writerow(
                {
                    "scene_id": scene.scene_id,
                    "source_side": scene.source_side,
                    "planet_date": serialize_date(scene.planet_date),
                    "year_month": scene.year_month,
                    "planet_path": str(scene.planet_path),
                    "bbox_epsg4326": bbox_to_csv(scene.bbox_epsg4326),
                    "bbox_epsg4326_pretty": format_bounds(scene.bbox_epsg4326),
                    "width": int(scene.width),
                    "height": int(scene.height),
                    "crs": scene.crs,
                }
            )


def build_download_command(
    *,
    project: str,
    start_day: str,
    end_day: str,
    bbox_epsg4326: tuple[float, float, float, float],
    data_root: Path,
    cloud_thresh: float,
    download_workers: int,
) -> str:
    parts = [
        "python",
        "ar_deploy.py",
        "download",
        "--project",
        project,
        "--start-day",
        start_day,
        "--end-day",
        end_day,
        f"--roi-bbox={bbox_to_csv(bbox_epsg4326)}",
        "--data-root",
        str(data_root),
        "--cloud-thresh",
        f"{float(cloud_thresh):g}",
        "--band-preset",
        "rgb_scl",
        "--download-workers",
        str(int(download_workers)),
    ]
    return " ".join(shlex.quote(part) for part in parts)


def build_scene_targets(
    scenes: list[PlanetSceneRecord],
    *,
    stage: str,
    window_records: list[WindowRecord],
) -> list[SceneWindowTarget]:
    targets: list[SceneWindowTarget] = []
    for scene in scenes:
        scene_windows = resolve_windows_for_scene(
            scene,
            stage=stage,
            window_records=window_records,
        )
        for window_payload in scene_windows:
            targets.append(
                SceneWindowTarget(
                    scene=scene,
                    stage=str(window_payload["stage"]),
                    start_day=str(window_payload["start_day"]),
                    end_day=str(window_payload["end_day"]),
                    window_source=str(window_payload["window_source"]),
                    window_days=int(window_payload["window_days"]),
                    merged_window_index=int(window_payload["merged_window_index"]),
                )
            )
    return targets


def union_bounds(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return (
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    )


def bounds_size(bounds: tuple[float, float, float, float]) -> tuple[float, float]:
    return bounds[2] - bounds[0], bounds[3] - bounds[1]


def bounds_intersect_with_gap(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    *,
    gap_deg: float,
) -> bool:
    return not (
        left[2] + gap_deg < right[0]
        or right[2] + gap_deg < left[0]
        or left[3] + gap_deg < right[1]
        or right[3] + gap_deg < left[1]
    )


def centroid_sort_key(target: SceneWindowTarget) -> tuple[float, float, str]:
    minx, miny, maxx, maxy = target.scene.bbox_epsg4326
    return ((minx + maxx) * 0.5, (miny + maxy) * 0.5, target.scene.scene_id)


def split_group_if_needed(
    targets: list[SceneWindowTarget],
    *,
    max_group_width_deg: float,
    max_group_height_deg: float,
    max_scenes_per_request: int,
) -> list[list[SceneWindowTarget]]:
    ordered = sorted(targets, key=centroid_sort_key)
    groups: list[list[SceneWindowTarget]] = []
    current: list[SceneWindowTarget] = []
    current_bounds: tuple[float, float, float, float] | None = None

    for target in ordered:
        target_bounds = target.scene.bbox_epsg4326
        if not current:
            current = [target]
            current_bounds = target_bounds
            continue

        assert current_bounds is not None
        merged_bounds = union_bounds(current_bounds, target_bounds)
        width_deg, height_deg = bounds_size(merged_bounds)
        exceeds_count = len(current) >= max_scenes_per_request
        exceeds_width = width_deg > max_group_width_deg
        exceeds_height = height_deg > max_group_height_deg
        if exceeds_count or exceeds_width or exceeds_height:
            groups.append(current)
            current = [target]
            current_bounds = target_bounds
            continue

        current.append(target)
        current_bounds = merged_bounds

    if current:
        groups.append(current)
    return groups


def connected_components_by_overlap(
    targets: list[SceneWindowTarget],
    *,
    gap_deg: float,
) -> list[list[SceneWindowTarget]]:
    if not targets:
        return []

    indexed = sorted(
        enumerate(targets),
        key=lambda item: (
            item[1].scene.bbox_epsg4326[0],
            item[1].scene.bbox_epsg4326[1],
            item[1].scene.scene_id,
        ),
    )
    uf = UnionFind(len(targets))
    active: list[tuple[int, tuple[float, float, float, float]]] = []

    for current_index, target in indexed:
        current_bounds = target.scene.bbox_epsg4326
        current_minx = current_bounds[0]
        active = [
            (other_index, other_bounds)
            for other_index, other_bounds in active
            if other_bounds[2] + gap_deg >= current_minx
        ]
        for other_index, other_bounds in active:
            if bounds_intersect_with_gap(current_bounds, other_bounds, gap_deg=gap_deg):
                uf.union(current_index, other_index)
        active.append((current_index, current_bounds))

    grouped: dict[int, list[SceneWindowTarget]] = {}
    for index, target in enumerate(targets):
        grouped.setdefault(uf.find(index), []).append(target)
    return list(grouped.values())


def build_request_id(
    *,
    stage: str,
    start_day: str,
    end_day: str,
    ordinal: int,
) -> str:
    start_token = start_day.replace("-", "")
    end_token = end_day.replace("-", "")
    return f"{stage}__{start_token}_{end_token}__{ordinal:04d}"


def plan_requests(
    targets: list[SceneWindowTarget],
    *,
    raw_out_root: Path,
    request_mode: str,
    max_group_width_deg: float,
    max_group_height_deg: float,
    merge_gap_deg: float,
    max_scenes_per_request: int,
) -> list[PlannedRequest]:
    if request_mode == "scene":
        planned: list[PlannedRequest] = []
        for ordinal, target in enumerate(targets, start=1):
            request_id = build_request_id(
                stage=target.stage,
                start_day=target.start_day,
                end_day=target.end_day,
                ordinal=ordinal,
            )
            planned.append(
                PlannedRequest(
                    stage=target.stage,
                    request_mode=request_mode,
                    request_id=request_id,
                    start_day=target.start_day,
                    end_day=target.end_day,
                    bbox_epsg4326=target.scene.bbox_epsg4326,
                    window_source=target.window_source,
                    window_days=target.window_days,
                    merged_window_index=target.merged_window_index,
                    scenes=(target.scene,),
                    data_root=raw_out_root / target.stage / target.scene.scene_id,
                )
            )
        return planned

    grouped_targets: dict[tuple[str, str, str, str, int, int], list[SceneWindowTarget]] = {}
    for target in targets:
        bucket_key = (
            target.stage,
            target.start_day,
            target.end_day,
            target.window_source,
            target.window_days,
            target.merged_window_index,
        )
        grouped_targets.setdefault(bucket_key, []).append(target)

    planned_requests: list[PlannedRequest] = []
    ordinal = 1
    for bucket_key in sorted(grouped_targets):
        stage, start_day, end_day, window_source, window_days, merged_window_index = bucket_key
        components = connected_components_by_overlap(
            grouped_targets[bucket_key],
            gap_deg=merge_gap_deg,
        )
        for component in components:
            for request_targets in split_group_if_needed(
                component,
                max_group_width_deg=max_group_width_deg,
                max_group_height_deg=max_group_height_deg,
                max_scenes_per_request=max_scenes_per_request,
            ):
                request_scenes = tuple(
                    target.scene
                    for target in sorted(
                        request_targets,
                        key=lambda item: (
                            item.scene.planet_date,
                            item.scene.scene_id,
                            str(item.scene.planet_path),
                        ),
                    )
                )
                request_bounds = request_scenes[0].bbox_epsg4326
                for scene in request_scenes[1:]:
                    request_bounds = union_bounds(request_bounds, scene.bbox_epsg4326)
                request_id = build_request_id(
                    stage=stage,
                    start_day=start_day,
                    end_day=end_day,
                    ordinal=ordinal,
                )
                planned_requests.append(
                    PlannedRequest(
                        stage=stage,
                        request_mode=request_mode,
                        request_id=request_id,
                        start_day=start_day,
                        end_day=end_day,
                        bbox_epsg4326=request_bounds,
                        window_source=window_source,
                        window_days=window_days,
                        merged_window_index=merged_window_index,
                        scenes=request_scenes,
                        data_root=raw_out_root / stage / request_id,
                    )
                )
                ordinal += 1
    return planned_requests


def preview_scene_ids(scenes: tuple[PlanetSceneRecord, ...], *, limit: int = 3) -> str:
    preview = [scene.scene_id for scene in scenes[:limit]]
    suffix = ",..." if len(scenes) > limit else ""
    return ",".join(preview) + suffix


def main() -> None:
    args = parse_args()

    if not args.planet_root.exists():
        raise FileNotFoundError(f"Planet root not found: {args.planet_root}")
    if int(args.download_workers) <= 0:
        raise ValueError("--download-workers must be positive.")
    if args.limit_scenes is not None and int(args.limit_scenes) <= 0:
        raise ValueError("--limit-scenes must be positive when provided.")
    if float(args.max_group_width_deg) <= 0.0:
        raise ValueError("--max-group-width-deg must be positive.")
    if float(args.max_group_height_deg) <= 0.0:
        raise ValueError("--max-group-height-deg must be positive.")
    if float(args.merge_gap_deg) < 0.0:
        raise ValueError("--merge-gap-deg must be non-negative.")
    if int(args.max_scenes_per_request) <= 0:
        raise ValueError("--max-scenes-per-request must be positive.")

    scene_manifest_tsv = (args.scene_manifest_tsv or default_scene_manifest_path()).resolve()
    plan_tsv = (args.plan_tsv or default_plan_path(args.stage)).resolve()
    commands_sh = (args.commands_sh or default_commands_path(args.stage)).resolve()
    raw_out_root = args.raw_out_root.resolve()

    window_records = load_window_records(args.windows_tsv)
    scenes = collect_planet_scene_records(args.planet_root.resolve())

    if args.retry_from is not None:
        retry_scene_ids, retry_planet_paths = load_retry_filter(args.retry_from.resolve())
        scenes = [
            scene
            for scene in scenes
            if scene.scene_id in retry_scene_ids or str(scene.planet_path) in retry_planet_paths
        ]
        print(f"[info] filtered retry_scenes={len(scenes)} from={args.retry_from}")

    if args.limit_scenes is not None:
        scenes = scenes[: int(args.limit_scenes)]
        print(f"[info] limited scenes to first {len(scenes)} records")

    if not scenes:
        raise RuntimeError("No Planet scenes matched the current planner filters.")

    write_scene_manifest(scene_manifest_tsv, scenes)

    scene_targets = build_scene_targets(
        scenes,
        stage=args.stage,
        window_records=window_records,
    )
    planned_requests = plan_requests(
        scene_targets,
        raw_out_root=raw_out_root,
        request_mode=args.request_mode,
        max_group_width_deg=float(args.max_group_width_deg),
        max_group_height_deg=float(args.max_group_height_deg),
        merge_gap_deg=float(args.merge_gap_deg),
        max_scenes_per_request=int(args.max_scenes_per_request),
    )

    plan_rows: list[dict[str, str | int | float]] = []
    for request in planned_requests:
        command = build_download_command(
            project=args.project,
            start_day=request.start_day,
            end_day=request.end_day,
            bbox_epsg4326=request.bbox_epsg4326,
            data_root=request.data_root,
            cloud_thresh=float(args.cloud_thresh),
            download_workers=int(args.download_workers),
        )
        plan_rows.append(
            {
                "stage": request.stage,
                "request_mode": request.request_mode,
                "request_id": request.request_id,
                "scene_count": len(request.scenes),
                "source_sides": ",".join(sorted({scene.source_side for scene in request.scenes})),
                "year_months": ",".join(sorted({scene.year_month for scene in request.scenes})),
                "covered_scene_ids": ",".join(scene.scene_id for scene in request.scenes),
                "covered_planet_dates": ",".join(
                    sorted({serialize_date(scene.planet_date) for scene in request.scenes})
                ),
                "bbox_epsg4326": bbox_to_csv(request.bbox_epsg4326),
                "bbox_epsg4326_pretty": format_bounds(request.bbox_epsg4326),
                "window_source": request.window_source,
                "window_days": int(request.window_days),
                "merged_window_index": int(request.merged_window_index),
                "start_day": request.start_day,
                "end_day": request.end_day,
                "data_root": str(request.data_root),
                "command": command,
            }
        )

    fieldnames = [
        "stage",
        "request_mode",
        "request_id",
        "scene_count",
        "source_sides",
        "year_months",
        "covered_scene_ids",
        "covered_planet_dates",
        "bbox_epsg4326",
        "bbox_epsg4326_pretty",
        "window_source",
        "window_days",
        "merged_window_index",
        "start_day",
        "end_day",
        "data_root",
        "command",
    ]
    plan_tsv.parent.mkdir(parents=True, exist_ok=True)
    with plan_tsv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(plan_rows)

    commands_sh.parent.mkdir(parents=True, exist_ok=True)
    with commands_sh.open("w", encoding="utf-8", newline="") as fp:
        fp.write("#!/usr/bin/env bash\n")
        fp.write("set -euo pipefail\n\n")
        fp.write(f"cd {shlex.quote(str(REPO_ROOT))}\n\n")
        for row, request in zip(plan_rows, planned_requests):
            fp.write(
                "# "
                f"{row['request_id']} scenes={row['scene_count']} "
                f"{row['start_day']}..{row['end_day']} "
                f"members={preview_scene_ids(request.scenes)}\n"
            )
            fp.write(f"{row['command']}\n\n")

    unique_windows = {
        (target.stage, target.start_day, target.end_day)
        for target in scene_targets
    }
    print(
        f"[done] scenes={len(scenes)} scene_targets={len(scene_targets)} requests={len(plan_rows)} "
        f"unique_windows={len(unique_windows)} request_mode={args.request_mode} "
        f"scene_manifest={scene_manifest_tsv} plan_tsv={plan_tsv} commands_sh={commands_sh}"
    )


if __name__ == "__main__":
    main()
