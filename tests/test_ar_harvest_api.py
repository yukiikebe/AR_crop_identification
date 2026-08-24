from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

import ar_harvest_api


class HarvestPrecomputedApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.predictions_root = self.root / "predictions"
        self.dataset_root = self.root / "dataset"
        self.dataset_root.mkdir()
        self.prediction_path = (
            self.predictions_root / "2025" / "1year" / "all_indices" / "predictions.csv"
        )
        self.prediction_path.parent.mkdir(parents=True)
        with self.prediction_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "year",
                    "tile",
                    "crop",
                    "pred_start_doy",
                    "pred_end_doy",
                    "num_observations",
                    "model_test_mae_days",
                ),
            )
            writer.writeheader()
            writer.writerows(
                (
                    {
                        "year": 2025,
                        "tile": "0_0",
                        "crop": "Corn",
                        "pred_start_doy": 100,
                        "pred_end_doy": 200,
                        "num_observations": 20,
                        "model_test_mae_days": 12.4,
                    },
                    {
                        "year": 2025,
                        "tile": "0_1",
                        "crop": "Corn",
                        "pred_start_doy": 120,
                        "pred_end_doy": 220,
                        "num_observations": 30,
                        "model_test_mae_days": 12.4,
                    },
                    {
                        "year": 2025,
                        "tile": "0_1",
                        "crop": "Cotton",
                        "pred_start_doy": 150,
                        "pred_end_doy": 250,
                        "num_observations": 25,
                        "model_test_mae_days": 18.7,
                    },
                )
            )
        self.prediction_path.with_name("metadata.json").write_text(
            json.dumps(
                {
                    "model": "hybrid",
                    "trained_years": [2022],
                    "evaluated_years": [2023],
                    "training_note": "Test artifact.",
                    "tile_bounds_wgs84": {
                        "0_0": [-92.20, 33.00, -92.10, 33.10],
                        "0_1": [-92.10, 33.00, -92.00, 33.10],
                    },
                }
            ),
            encoding="utf-8",
        )
        self.environment = patch.dict(
            os.environ,
            {
                "DEEPSAT_HARVEST_PRED_ROOT": str(self.predictions_root),
                "DEEPSAT_HARVEST_DATASET_ROOT": str(self.dataset_root),
                "DEEPSAT_HARVEST_MODEL_WINDOW": "1year",
                "DEEPSAT_HARVEST_FEATURE_SET": "all_indices",
            },
        )
        self.environment.start()

    def tearDown(self) -> None:
        self.environment.stop()
        self.temporary_directory.cleanup()

    def test_info_reports_precomputed_artifact(self) -> None:
        response = ar_harvest_api.info()

        self.assertTrue(response["ready"])
        self.assertEqual(response["serving_mode"], "precomputed")
        self.assertEqual(response["available_years"], [2025])
        self.assertTrue(response["artifacts"]["2025"]["ready"])
        self.assertEqual(response["artifacts"]["2025"]["tile_count"], 2)

    def test_predict_filters_tiles_and_aggregates_precomputed_rows(self) -> None:
        request = ar_harvest_api.HarvestRequest(
            year=2025,
            bbox=ar_harvest_api.BBox(
                lon_min=-92.20,
                lat_min=33.00,
                lon_max=-92.00,
                lat_max=33.10,
            ),
        )

        response = ar_harvest_api.predict(request)

        crops = {row["crop"]: row for row in response["predictions"]}
        self.assertEqual(response["serving_mode"], "precomputed")
        self.assertEqual(response["tiles_with_inputs"], 2)
        self.assertEqual(response["crop_count"], 2)
        self.assertEqual(crops["Corn"]["start_doy"], 110)
        self.assertEqual(crops["Corn"]["end_doy"], 210)
        self.assertEqual(crops["Corn"]["median_observations_per_tile"], 25)
        self.assertEqual(crops["Cotton"]["tiles_with_crop"], 1)

    def test_full_arkansas_bbox_is_accepted(self) -> None:
        ar_harvest_api._validate_bbox(
            ar_harvest_api.BBox(
                lon_min=-94.70,
                lat_min=36.40,
                lon_max=-94.60,
                lat_max=36.50,
            )
        )

    def test_predict_rejects_year_without_precomputed_artifact(self) -> None:
        request = ar_harvest_api.HarvestRequest(
            year=2024,
            bbox=ar_harvest_api.BBox(
                lon_min=-92.20,
                lat_min=33.00,
                lon_max=-92.00,
                lat_max=33.10,
            ),
        )

        with self.assertRaises(HTTPException) as context:
            ar_harvest_api.predict(request)

        self.assertEqual(context.exception.status_code, 404)
        self.assertIn(
            "Precomputed harvest predictions are unavailable", context.exception.detail
        )


if __name__ == "__main__":
    unittest.main()
