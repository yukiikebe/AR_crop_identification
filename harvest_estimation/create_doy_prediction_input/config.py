from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SEEDING_CONFIG: dict[str, dict[str, Any]] = {
    "Corn": {
        "window_start": "03-15",
        "window_end": "04-30",
        "offset_days": 10,
    },
    "Rice": {
        "window_start": "03-20",
        "window_end": "04-30",
        "offset_days": 14,
    },
    "Soybeans": {
        "window_start": "04-01",
        "window_end": "05-20",
        "offset_days": 7,
    },
    "Cotton": {
        "window_start": "04-15",
        "window_end": "05-20",
        "offset_days": 10,
    },
    "Winter Wheat": {
        "window_start": "02-01",
        "window_end": "04-25",
        "offset_days": None,
    },
}


def _normalize_single_seeding_entry(
    crop_name: str, raw_value: Any, base_value: dict[str, Any] | None
) -> dict[str, Any]:
    normalized = dict(base_value or {})

    if isinstance(raw_value, dict):
        window = raw_value.get("window")
        if isinstance(window, (list, tuple)) and len(window) == 2:
            normalized["window_start"] = str(window[0])
            normalized["window_end"] = str(window[1])
        if "window_start" in raw_value:
            normalized["window_start"] = str(raw_value["window_start"])
        if "window_end" in raw_value:
            normalized["window_end"] = str(raw_value["window_end"])
        if "offset_days" in raw_value:
            offset = raw_value["offset_days"]
            normalized["offset_days"] = (
                None if offset in (None, "", "null") else int(offset)
            )
    elif isinstance(raw_value, (list, tuple)):
        if len(raw_value) == 2:
            normalized["window_start"] = str(raw_value[0])
            normalized["window_end"] = str(raw_value[1])
        elif len(raw_value) == 3:
            normalized["window_start"] = str(raw_value[0])
            normalized["window_end"] = str(raw_value[1])
            offset = raw_value[2]
            normalized["offset_days"] = (
                None if offset in (None, "", "null") else int(offset)
            )
        else:
            raise ValueError(f"Bad seeding config list for {crop_name}: {raw_value}")
    else:
        raise ValueError(f"Bad seeding config format for {crop_name}: {raw_value}")

    if "window_start" not in normalized or "window_end" not in normalized:
        raise ValueError(f"Missing seeding window for {crop_name}: {raw_value}")

    offset = normalized.get("offset_days")
    normalized["offset_days"] = None if offset in (None, "", "null") else int(offset)
    return normalized


def load_seeding_config_yaml(path: str | Path | None) -> dict[str, dict[str, Any]]:
    normalized_defaults = {
        crop_name: dict(values) for crop_name, values in DEFAULT_SEEDING_CONFIG.items()
    }

    if path is None:
        return normalized_defaults

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Seeding config YAML not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Seeding config YAML must contain a crop mapping: {path}")

    merged = {
        crop_name: dict(values) for crop_name, values in normalized_defaults.items()
    }
    for crop_name, raw_value in raw.items():
        merged[str(crop_name)] = _normalize_single_seeding_entry(
            str(crop_name),
            raw_value,
            merged.get(str(crop_name)),
        )
    return merged


class Config:
    """
    Configuration holder for the harvest pipeline.
    """

    def __init__(
        self,
        cdl_yaml_path,
        *,
        allowed_crops=None,
        min_points=11,
        savgol_window=11,
        savgol_polyorder=3,
        output_root="./outputs",
        gt_windows=None,
        seeding_config_yaml=None,
    ):
        self.cdl_yaml_path = Path(cdl_yaml_path)
        self.output_root = Path(output_root)
        self.gt_windows = gt_windows or {}
        raw_seeding_config = load_seeding_config_yaml(seeding_config_yaml)

        self._load_cdl_yaml()
        self.seeding_config = {
            crop_name: dict(values)
            for crop_name, values in raw_seeding_config.items()
            if crop_name in set(self.crop_dict.values())
        }

        self.allowed_crops = (
            set(allowed_crops)
            if allowed_crops is not None
            else set(self.crop_dict.values())
        )

        self.min_points = min_points
        self.savgol_window = savgol_window
        self.savgol_polyorder = savgol_polyorder

        self._validate()

    def _load_cdl_yaml(self):
        if not self.cdl_yaml_path.exists():
            raise FileNotFoundError(f"CDL config not found: {self.cdl_yaml_path}")

        with open(self.cdl_yaml_path, "r") as f:
            data = yaml.safe_load(f)

        # numeric label -> crop name
        self.crop_dict = {int(k): v for k, v in data.get("num2class", {}).items()}

        # numeric labels that exist in CDL
        self.valid_crop_labels = set(int(k) for k in data.get("crop_type", {}).keys())

    def _validate(self):
        unknown = self.allowed_crops - set(self.crop_dict.values())
        if unknown:
            raise ValueError(f"Allowed crops not in CDL config: {unknown}")

        if self.savgol_window % 2 == 0:
            raise ValueError("savgol_window must be odd")

        if self.savgol_window <= self.savgol_polyorder:
            raise ValueError("savgol_window must be larger than savgol_polyorder")

        for crop, (start, end) in self.gt_windows.items():
            if crop not in self.crop_dict.values():
                raise ValueError(f"GT window crop not in CDL: {crop}")

        for crop, settings in self.seeding_config.items():
            if "window_start" not in settings or "window_end" not in settings:
                raise ValueError(f"Incomplete seeding config for {crop}: {settings}")

    def is_valid_crop_label(self, label):
        return label in self.valid_crop_labels

    def crop_name(self, label):
        return self.crop_dict.get(label, f"Crop_{label}")

    @staticmethod
    def crop_dir_name(crop_name: str) -> str:
        safe_name = crop_name.replace("/", "-")
        safe_name = re.sub(r"\s+", " ", safe_name).strip()
        return safe_name

    def crop_name_from_dir(self, crop_dir_name: str) -> str | None:
        for crop_name in self.crop_dict.values():
            if self.crop_dir_name(crop_name) == crop_dir_name:
                return crop_name
        return None

    def seeding_settings_for(self, crop_name: str) -> dict[str, Any]:
        return dict(self.seeding_config.get(crop_name, {}))
