"""
Generate training data from S3 export SQLite database.

Reads datapoints from the exported SQLite, extracts encoder parameters,
and produces encoded numpy arrays (128x128 RGBA uint8) for ML training.

Bypasses the encoder's WindowBorderValidator (which rejects windows on
diagonal walls) by calling the image builder director directly.

Usage:
    python scripts/generate_training_data.py --db ./dataset.sqlite --dp 42
    python scripts/generate_training_data.py --db ./dataset.sqlite --dp 42 --output ./output/
    python scripts/generate_training_data.py --db ./dataset.sqlite --all --output ./output/
"""
import os
import sys
import json
import sqlite3
import argparse
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

# Add server_encoder root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.components.enums import ModelType, EncodingScheme, ParameterName
from src.components.image_builder import RoomImageBuilder, RoomImageDirector
from src.components.parameter_calculators import ParameterCalculatorRegistry
from src.server.services.logging import StructuredLogger
from src.server.enums import LogLevel


class MetaJsonExtractor:
    """Extracts encoder-compatible parameters from meta_json stored in the DB."""

    @staticmethod
    def extract(meta_json: dict) -> Dict[str, Any]:
        """
        Extract encoder parameters from meta_json for df_default model type.

        Args:
            meta_json: The parsed meta_json dict from the DB

        Returns:
            Flat parameter dict compatible with the encoder's image builder
        """
        room = meta_json.get("room", {})
        reflectance = meta_json.get("reflectance", {})
        refl = reflectance.get("default", {})

        # Shared parameters
        params: Dict[str, Any] = {
            "room_polygon": room.get("room_polygon", []),
            "floor_height_above_terrain": float(room.get("floor_height_above_terrain", 0)),
            "height_roof_over_floor": float(room.get("height_roof_over_floor", 2.7)),
            "wall_reflectance": float(refl.get("walls", 0.7)),
            "floor_reflectance": float(refl.get("floor", 0.3)),
            "ceiling_reflectance": float(refl.get("ceiling", 0.8)),
            "facade_reflectance": float(refl.get("facade", 0.3)),
            "terrain_reflectance": float(refl.get("terrain", 0.2)),
        }

        # Windows
        windows_section = meta_json.get("window", {})
        windows: Dict[str, Dict[str, Any]] = {}

        for win_id, win_data in sorted(windows_section.items()):
            if not isinstance(win_data, dict):
                continue
            windows[win_id] = {
                "x1": float(win_data["x1"]),
                "y1": float(win_data["y1"]),
                "z1": float(win_data["z1"]),
                "x2": float(win_data["x2"]),
                "y2": float(win_data["y2"]),
                "z2": float(win_data["z2"]),
                "window_frame_ratio": float(win_data.get("frame_ratio", 0.15)),
                "direction_angle": float(win_data.get("direction_angle", 0.0)),
                "wall_thickness": float(win_data.get("wall_thickness", 0.3)),
                "window_frame_reflectance": float(refl.get("window_frame", 0.5)),
                # Obstruction angles: 0 for now
                "horizon": 0.0,
                "zenith": 0.0,
                # Context/balcony reflectance: defaults for now
                "context_reflectance": 0.6,
                "balcony_reflectance": 0.8,
            }

        params["windows"] = windows
        return params


class TrainingDataGenerator:
    """Generates encoder training data from a SQLite database."""

    def __init__(self, db_path: str, output_dir: str, logger: StructuredLogger):
        self._db_path = db_path
        self._output_dir = output_dir
        self._logger = logger
        self._builder = RoomImageBuilder(encoding_scheme=EncodingScheme.HSV)
        self._director = RoomImageDirector(self._builder)

    def generate_for_datapoint(self, dp_id: int) -> Optional[str]:
        """
        Generate encoder output for a single datapoint.

        Returns:
            Path to the saved .npz file, or None if failed
        """
        meta_json = self._read_datapoint(dp_id)
        if meta_json is None:
            self._logger.error(f"Datapoint {dp_id} not found in database")
            return None

        model_type = ModelType.DF_DEFAULT
        params = MetaJsonExtractor.extract(meta_json)

        window_count = len(params.get("windows", {}))
        self._logger.info(
            f"dp={dp_id}: {window_count} window(s), model_type={model_type.value}"
        )

        try:
            # Clip height_roof_over_floor to encoder's valid range [15, 30]
            hrof = params.get("height_roof_over_floor", 2.7)
            if hrof < 15.0:
                params["height_roof_over_floor"] = 15.0
            elif hrof > 30.0:
                params["height_roof_over_floor"] = 30.0

            # Calculate derived parameters (window_sill_height, window_height)
            # for each window before building
            result = self._director.construct_multi_window_images(model_type, params)

            npz_path = os.path.join(
                self._output_dir, f"dp_{dp_id:04d}_{model_type.value}.npz"
            )
            arrays_dict = {}
            for window_id in result.window_ids():
                image = result.get_image(window_id).astype(np.uint8)
                arrays_dict[f"{window_id}_image"] = image
                mask = result.get_mask(window_id)
                if mask is not None:
                    arrays_dict[f"{window_id}_mask"] = mask

            np.savez_compressed(npz_path, **arrays_dict)
            self._logger.info(f"dp={dp_id}: saved to {npz_path}")
            return npz_path

        except Exception as e:
            self._logger.error(f"dp={dp_id}: encoding failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_all(self) -> List[Tuple[int, Optional[str]]]:
        """Generate encoder output for all datapoints in the database."""
        conn = sqlite3.connect(self._db_path)
        rows = conn.execute("SELECT dp FROM datapoints ORDER BY dp").fetchall()
        conn.close()

        self._logger.info(f"Found {len(rows)} datapoints in database")
        results = []

        for (dp_id,) in rows:
            path = self.generate_for_datapoint(dp_id)
            results.append((dp_id, path))

        success = sum(1 for _, p in results if p is not None)
        self._logger.info(
            f"Done: {success}/{len(results)} datapoints generated successfully"
        )
        return results

    def _read_datapoint(self, dp_id: int) -> Optional[dict]:
        """Read meta_json for a single datapoint from the DB."""
        conn = sqlite3.connect(self._db_path)
        row = conn.execute(
            "SELECT meta_json FROM datapoints WHERE dp = ?", (dp_id,)
        ).fetchone()
        conn.close()

        if row is None:
            return None
        return json.loads(row[0])


def main():
    parser = argparse.ArgumentParser(
        description="Generate encoder training data from S3 export SQLite"
    )
    parser.add_argument("--db", required=True, help="Path to the SQLite database")
    parser.add_argument("--dp", type=int, help="Specific datapoint ID to process")
    parser.add_argument("--all", action="store_true", help="Process all datapoints")
    parser.add_argument(
        "--output", default="./training_output", help="Output directory"
    )

    args = parser.parse_args()

    if args.dp is None and not args.all:
        parser.error("Specify either --dp <id> or --all")

    if not os.path.exists(args.db):
        print(f"Error: database not found: {args.db}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    logger = StructuredLogger("TrainingDataGen", LogLevel.INFO)
    generator = TrainingDataGenerator(args.db, args.output, logger)

    if args.dp is not None:
        path = generator.generate_for_datapoint(args.dp)
        if path:
            print(f"Saved: {path}")
        else:
            print(f"Failed to generate for dp={args.dp}")
            sys.exit(1)
    else:
        results = generator.generate_all()
        for dp_id, path in results:
            status = f"OK: {path}" if path else "FAILED"
            print(f"  dp={dp_id}: {status}")


if __name__ == "__main__":
    main()
