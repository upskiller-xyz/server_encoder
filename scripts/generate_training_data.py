"""
Generate training data from S3 export SQLite database.

Reads datapoints from the exported SQLite, extracts encoder parameters,
and produces encoded numpy arrays (128x128 RGBA uint8) for ML training.

Supports all 4 model types (df_default, df_custom, da_default, da_custom)
and optionally calculates real obstruction angles via server_obstruction.

Bypasses the encoder's WindowBorderValidator (which rejects windows on
diagonal walls) by calling the image builder director directly.

Usage:
    python scripts/generate_training_data.py --db ./dataset.sqlite --dp 42
    python scripts/generate_training_data.py --db ./dataset.sqlite --all --output ./output/
    python scripts/generate_training_data.py --db ./dataset.sqlite --dp 42 --no-obstruction
    python scripts/generate_training_data.py --db ./dataset.sqlite --all --model-types df_default df_custom
"""
import os
import sys
import json
import math
import sqlite3
import logging
import argparse
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import traceback

import numpy as np

# ---------------------------------------------------------------------------
# Module imports from two sibling projects that share the `src.*` namespace.
# We import obstruction modules FIRST, back them up from sys.modules,
# then import encoder modules so each project resolves its own dependencies.
# ---------------------------------------------------------------------------

_OBSTRUCTION_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "server_obstruction")
_ENCODER_ROOT = os.path.join(os.path.dirname(__file__), "..")

# --- Phase 1: obstruction imports -------------------------------------------
sys.path.insert(0, _OBSTRUCTION_ROOT)

from src.components.geometry import Point3D as ObsPoint3D, Vector3D, Mesh
from src.components.models import Window as ObsWindow
from src.components.calculators.intersection_calculator import IntersectionCalculator
from src.components.filter import CompositeTriangleFilter, CoarseTriangleFilter
from src.server.base.constants import ANGLES

# Back up obstruction modules and clear from sys.modules
_obs_keys = [k for k in sys.modules if k.startswith("src")]
_obs_backup = {k: sys.modules.pop(k) for k in _obs_keys}
sys.path.remove(_OBSTRUCTION_ROOT)

# --- Phase 2: encoder imports -----------------------------------------------
sys.path.insert(0, _ENCODER_ROOT)

from src.components.enums import ModelType, EncodingScheme, ParameterName
from src.components.image_builder import RoomImageBuilder, RoomImageDirector
from src.components.geometry import WindowGeometry, RoomPolygon
from src.server.services.logging import StructuredLogger
from src.server.enums import LogLevel

logger = logging.getLogger(__name__)

# Number of directions for obstruction sweep (matches server_obstruction default)
NUM_OBSTRUCTION_DIRECTIONS = 64
OBSTRUCTION_START_DEG = 17.5
OBSTRUCTION_END_DEG = 162.5


class ReflectanceSource(Enum):
    """Which reflectance set to use from meta_json."""
    DEFAULT = "default"
    CUSTOM = "custom"


MODEL_TYPE_CONFIG: Dict[ModelType, Dict[str, Any]] = {
    ModelType.DF_DEFAULT: {"reflectance_source": ReflectanceSource.DEFAULT},
    ModelType.DF_CUSTOM: {"reflectance_source": ReflectanceSource.CUSTOM},
    ModelType.DA_DEFAULT: {"reflectance_source": ReflectanceSource.DEFAULT},
    ModelType.DA_CUSTOM: {"reflectance_source": ReflectanceSource.CUSTOM},
}


@dataclass
class ObstructionData:
    """Holds computed obstruction data for a single window."""
    horizon: List[float]
    zenith: List[float]
    context_reflectance: float
    balcony_reflectance: float


class ObstructionCalculator:
    """Calculates horizon/zenith obstruction angles from meta_json geometry.

    Uses server_obstruction's IntersectionCalculator directly (no HTTP,
    no ProcessPoolExecutor) for deterministic sequential computation.

    The reference point is currently calculated in this script using the
    encoder's WindowGeometry.calculate_reference_point_from_polygon.
    Once Stasja updates the obstruction server to accept window endpoints
    + room_polygon directly, this intermediate calculation will move into
    the obstruction server itself.
    """

    def __init__(self, meta_json: dict):
        geometry = meta_json.get("geometry", {})
        self._context_mesh = self._build_mesh(
            geometry.get("context_buildings", [])
        )
        self._balcony_mesh = self._build_mesh(
            geometry.get("balcony_above", [])
        )

        reflectance = meta_json.get("reflectance", {})
        default_refl = reflectance.get("default", {})
        self._default_context_refl = float(default_refl.get("context_buildings", 0.6))
        self._default_balcony_refl = float(default_refl.get("balcony_ceiling", 0.8))

        custom_refl = reflectance.get("custom", {})
        self._custom_context_refl = float(
            custom_refl.get("context_buildings", self._default_context_refl)
        )
        self._custom_balcony_refl = float(
            custom_refl.get("balcony_ceiling", self._default_balcony_refl)
        )

    @staticmethod
    def _build_mesh(vertices: list) -> Optional[Mesh]:
        """Build an obstruction Mesh from a flat list of [x,y,z] triplets."""
        if not vertices or len(vertices) < 3:
            return None
        try:
            return Mesh.from_vertices(vertices)
        except (ValueError, TypeError):
            return None

    def _build_combined_mesh(self) -> Optional[Mesh]:
        """Combine context buildings and balcony meshes into one."""
        if self._context_mesh is None and self._balcony_mesh is None:
            return None
        ctx_tris = self._context_mesh.triangles if self._context_mesh else ()
        bal_tris = self._balcony_mesh.triangles if self._balcony_mesh else ()
        combined = tuple(ctx_tris) + tuple(bal_tris)
        if not combined:
            return None
        return Mesh(combined)

    def calculate_for_window(
        self,
        win_data: dict,
        room_polygon: RoomPolygon,
        reflectance_source: ReflectanceSource,
    ) -> ObstructionData:
        """Calculate obstruction angles for a window.

        Args:
            win_data: Window data dict with x1..z2, direction_angle
            room_polygon: Encoder RoomPolygon for reference point projection
            reflectance_source: DEFAULT or CUSTOM

        Returns:
            ObstructionData with horizon/zenith arrays and reflectances
        """
        combined_mesh = self._build_combined_mesh()

        if reflectance_source == ReflectanceSource.CUSTOM:
            ctx_refl = self._custom_context_refl
            bal_refl = self._custom_balcony_refl
        else:
            ctx_refl = self._default_context_refl
            bal_refl = self._default_balcony_refl

        if combined_mesh is None:
            return ObstructionData(
                horizon=[0.0] * NUM_OBSTRUCTION_DIRECTIONS,
                zenith=[0.0] * NUM_OBSTRUCTION_DIRECTIONS,
                context_reflectance=ctx_refl,
                balcony_reflectance=bal_refl,
            )

        # Calculate reference point (TEMPORARY: done here until Stasja
        # updates obstruction server to accept window endpoints directly)
        direction_angle = float(win_data.get("direction_angle", 0.0))
        encoder_win = WindowGeometry(
            float(win_data["x1"]), float(win_data["y1"]), float(win_data["z1"]),
            float(win_data["x2"]), float(win_data["y2"]), float(win_data["z2"]),
            direction_angle=direction_angle,
        )
        ref_point = encoder_win.calculate_reference_point_from_polygon(room_polygon)

        # Build obstruction Window
        obs_center = ObsPoint3D(x=ref_point.x, y=ref_point.y, z=ref_point.z)
        obs_normal = Vector3D.from_horizontal_angle(direction_angle)
        obs_window = ObsWindow(center=obs_center, normal=obs_normal)

        horizon_vals, zenith_vals = self._sweep_directions(obs_window, combined_mesh)

        return ObstructionData(
            horizon=horizon_vals,
            zenith=zenith_vals,
            context_reflectance=ctx_refl,
            balcony_reflectance=bal_refl,
        )

    @staticmethod
    def _sweep_directions(
        window: ObsWindow,
        mesh: Mesh,
        num_directions: int = NUM_OBSTRUCTION_DIRECTIONS,
    ) -> Tuple[List[float], List[float]]:
        """Sweep obstruction angles across directions synchronously."""
        coarse_filtered = CoarseTriangleFilter.call(mesh.triangles, window)
        filtered_mesh = Mesh(coarse_filtered)

        normal_arr = window.normal.to_array()
        base_angle = math.atan2(normal_arr[1], normal_arr[0])
        start_rad = math.radians(OBSTRUCTION_START_DEG)
        end_rad = math.radians(OBSTRUCTION_END_DEG)
        step = (end_rad - start_rad) / (num_directions - 1) if num_directions > 1 else 0

        horizons: List[float] = []
        zeniths: List[float] = []

        for i in range(num_directions):
            relative = start_rad + (i * step)
            abs_angle = base_angle - (math.pi * 0.5) + relative
            rotated = ObsWindow.set_angle(window, abs_angle)

            h_filtered, v_filtered = CompositeTriangleFilter.call(
                filtered_mesh.triangles, rotated
            )
            h_result = IntersectionCalculator.call(
                Mesh(h_filtered), rotated, ANGLES.HORIZON
            )
            z_result = IntersectionCalculator.call(
                Mesh(v_filtered), rotated, ANGLES.ZENITH
            )
            horizons.append(h_result.obstruction_angle_degrees)
            zeniths.append(z_result.obstruction_angle_degrees)

        return horizons, zeniths


class MetaJsonExtractor:
    """Extracts encoder-compatible parameters from meta_json stored in the DB."""

    # Reflectance keys mapping: encoder param name -> meta_json key
    _REFLECTANCE_KEYS = {
        "wall_reflectance": "walls",
        "floor_reflectance": "floor",
        "ceiling_reflectance": "ceiling",
        "facade_reflectance": "facade",
        "terrain_reflectance": "terrain",
        "window_frame_reflectance": "window_frame",
    }

    # Defaults when reflectance not found
    _REFLECTANCE_DEFAULTS = {
        "wall_reflectance": 0.7,
        "floor_reflectance": 0.3,
        "ceiling_reflectance": 0.8,
        "facade_reflectance": 0.3,
        "terrain_reflectance": 0.2,
        "window_frame_reflectance": 0.5,
    }

    @classmethod
    def extract_for_model_type(
        cls,
        meta_json: dict,
        model_type: ModelType,
        obstruction_data: Optional[Dict[str, ObstructionData]] = None,
    ) -> Dict[str, Any]:
        """Extract encoder parameters for a specific model type.

        Args:
            meta_json: The parsed meta_json dict from the DB
            model_type: Which model type to generate for
            obstruction_data: Pre-computed obstruction per window_id, or None for zeros

        Returns:
            Flat parameter dict compatible with the encoder's image builder
        """
        config = MODEL_TYPE_CONFIG[model_type]
        refl_source = config["reflectance_source"]

        room = meta_json.get("room", {})
        reflectance = meta_json.get("reflectance", {})
        refl = reflectance.get(refl_source.value, reflectance.get("default", {}))

        params: Dict[str, Any] = {
            "room_polygon": room.get("room_polygon", []),
            "floor_height_above_terrain": float(room.get("floor_height_above_terrain", 0)),
            "height_roof_over_floor": float(room.get("height_roof_over_floor", 2.7)),
        }

        # Reflectances
        for param_name, meta_key in cls._REFLECTANCE_KEYS.items():
            default = cls._REFLECTANCE_DEFAULTS[param_name]
            params[param_name] = float(refl.get(meta_key, default))

        # window_direction_angle is auto-populated by the image builder from
        # each window's direction_angle, so no explicit pass is needed here.

        # Windows
        windows_section = meta_json.get("window", {})
        windows: Dict[str, Dict[str, Any]] = {}

        for win_id, win_data in sorted(windows_section.items()):
            if not isinstance(win_data, dict):
                continue

            direction_angle = float(win_data.get("direction_angle", 0.0))

            win_params: Dict[str, Any] = {
                "x1": float(win_data["x1"]),
                "y1": float(win_data["y1"]),
                "z1": float(win_data["z1"]),
                "x2": float(win_data["x2"]),
                "y2": float(win_data["y2"]),
                "z2": float(win_data["z2"]),
                "window_frame_ratio": float(win_data.get("frame_ratio", 0.15)),
                "direction_angle": direction_angle,
                "wall_thickness": float(win_data.get("wall_thickness", 0.3)),
                "window_frame_reflectance": params["window_frame_reflectance"],
            }

            # Obstruction data
            if obstruction_data and win_id in obstruction_data:
                obs = obstruction_data[win_id]
                win_params["horizon"] = obs.horizon
                win_params["zenith"] = obs.zenith
                win_params["context_reflectance"] = obs.context_reflectance
                win_params["balcony_reflectance"] = obs.balcony_reflectance
            else:
                win_params["horizon"] = 0.0
                win_params["zenith"] = 0.0
                win_params["context_reflectance"] = 0.6
                win_params["balcony_reflectance"] = 0.8

            windows[win_id] = win_params

        params["windows"] = windows
        return params

    @classmethod
    def extract(cls, meta_json: dict) -> Dict[str, Any]:
        """Legacy wrapper: extract for df_default with no obstruction."""
        return cls.extract_for_model_type(meta_json, ModelType.DF_DEFAULT)


class TrainingDataGenerator:
    """Generates encoder training data from a SQLite database."""

    def __init__(
        self,
        db_path: str,
        output_dir: str,
        logger: StructuredLogger,
        model_types: List[ModelType],
        use_obstruction: bool = True,
    ):
        self._db_path = db_path
        self._output_dir = output_dir
        self._logger = logger
        self._model_types = model_types
        self._use_obstruction = use_obstruction
        self._builder = RoomImageBuilder(encoding_scheme=EncodingScheme.HSV)
        self._director = RoomImageDirector(self._builder)

    def generate_for_datapoint(self, dp_id: int) -> List[Tuple[ModelType, Optional[str]]]:
        """Generate encoder output for a single datapoint across all model types.

        Returns:
            List of (model_type, path) tuples
        """
        meta_json = self._read_datapoint(dp_id)
        if meta_json is None:
            self._logger.error(f"Datapoint {dp_id} not found in database")
            return [(mt, None) for mt in self._model_types]

        # Compute obstruction once (shared across model types)
        obstruction_data = self._compute_obstruction(meta_json, dp_id)

        results = []
        for model_type in self._model_types:
            config = MODEL_TYPE_CONFIG[model_type]
            refl_source = config["reflectance_source"]

            # Rebuild obstruction data with correct reflectance source
            obs_for_model = None
            if obstruction_data is not None:
                obs_for_model = {}
                for win_id, obs in obstruction_data.items():
                    if refl_source == ReflectanceSource.CUSTOM:
                        obs_for_model[win_id] = ObstructionData(
                            horizon=obs.horizon,
                            zenith=obs.zenith,
                            context_reflectance=obs.context_reflectance,
                            balcony_reflectance=obs.balcony_reflectance,
                        )
                    else:
                        obs_for_model[win_id] = obs

            path = self._generate_single(dp_id, meta_json, model_type, obs_for_model)
            results.append((model_type, path))

        return results

    def _compute_obstruction(
        self, meta_json: dict, dp_id: int
    ) -> Optional[Dict[str, ObstructionData]]:
        """Compute obstruction data for all windows in a datapoint."""
        if not self._use_obstruction:
            return None

        calculator = ObstructionCalculator(meta_json)
        room_data = meta_json.get("room", {})
        polygon_data = room_data.get("room_polygon", [])

        if not polygon_data:
            self._logger.warning(f"dp={dp_id}: no room_polygon, skipping obstruction")
            return None

        room_polygon = RoomPolygon.from_dict(polygon_data)
        windows_section = meta_json.get("window", {})
        result: Dict[str, ObstructionData] = {}

        for win_id, win_data in sorted(windows_section.items()):
            if not isinstance(win_data, dict):
                continue
            try:
                # Calculate with default reflectance; custom will be swapped later
                obs = calculator.calculate_for_window(
                    win_data, room_polygon, ReflectanceSource.DEFAULT
                )
                result[win_id] = obs
                max_h = max(obs.horizon) if obs.horizon else 0
                max_z = max(obs.zenith) if obs.zenith else 0
                self._logger.info(
                    f"dp={dp_id} win={win_id}: "
                    f"max_horizon={max_h:.1f}° max_zenith={max_z:.1f}°"
                )
            except Exception as e:
                self._logger.warning(
                    f"dp={dp_id} win={win_id}: obstruction failed: {e}, using zeros"
                )
                result[win_id] = ObstructionData(
                    horizon=[0.0] * NUM_OBSTRUCTION_DIRECTIONS,
                    zenith=[0.0] * NUM_OBSTRUCTION_DIRECTIONS,
                    context_reflectance=0.6,
                    balcony_reflectance=0.8,
                )

        return result

    def _generate_single(
        self,
        dp_id: int,
        meta_json: dict,
        model_type: ModelType,
        obstruction_data: Optional[Dict[str, ObstructionData]],
    ) -> Optional[str]:
        """Generate encoder output for one datapoint + one model type."""
        params = MetaJsonExtractor.extract_for_model_type(
            meta_json, model_type, obstruction_data
        )

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
            self._logger.info(f"dp={dp_id}: saved {model_type.value} to {npz_path}")
            return npz_path

        except Exception as e:
            self._logger.error(
                f"dp={dp_id}: encoding failed for {model_type.value}: {e}"
            )
            traceback.print_exc()
            return None

    def generate_all(self) -> List[Tuple[int, List[Tuple[ModelType, Optional[str]]]]]:
        """Generate encoder output for all datapoints in the database."""
        conn = sqlite3.connect(self._db_path)
        rows = conn.execute("SELECT dp FROM datapoints ORDER BY dp").fetchall()
        conn.close()

        self._logger.info(f"Found {len(rows)} datapoints in database")
        results = []

        for (dp_id,) in rows:
            model_results = self.generate_for_datapoint(dp_id)
            results.append((dp_id, model_results))

        total_dps = len(results)
        total_files = sum(
            1 for _, mrs in results for _, p in mrs if p is not None
        )
        expected = total_dps * len(self._model_types)
        self._logger.info(
            f"Done: {total_files}/{expected} files generated "
            f"({total_dps} datapoints x {len(self._model_types)} model types)"
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


def _parse_model_types(values: Optional[List[str]]) -> List[ModelType]:
    """Parse model type strings into ModelType enums."""
    if not values:
        return list(ModelType)

    valid = {mt.value: mt for mt in ModelType}
    result = []
    for v in values:
        if v not in valid:
            raise argparse.ArgumentTypeError(
                f"Unknown model type '{v}'. Valid: {list(valid.keys())}"
            )
        result.append(valid[v])
    return result


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
    parser.add_argument(
        "--no-obstruction",
        action="store_true",
        help="Skip obstruction calculation (use zeros)",
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        default=None,
        help="Model types to generate (default: all). "
        "Options: df_default df_custom da_default da_custom",
    )

    args = parser.parse_args()

    if args.dp is None and not args.all:
        parser.error("Specify either --dp <id> or --all")

    if not os.path.exists(args.db):
        print(f"Error: database not found: {args.db}")
        sys.exit(1)

    model_types = _parse_model_types(args.model_types)
    os.makedirs(args.output, exist_ok=True)

    log = StructuredLogger("TrainingDataGen", LogLevel.INFO)
    generator = TrainingDataGenerator(
        args.db,
        args.output,
        log,
        model_types=model_types,
        use_obstruction=not args.no_obstruction,
    )

    if args.dp is not None:
        model_results = generator.generate_for_datapoint(args.dp)
        for mt, path in model_results:
            status = f"OK: {path}" if path else "FAILED"
            print(f"  {mt.value}: {status}")
        if all(p is None for _, p in model_results):
            sys.exit(1)
    else:
        results = generator.generate_all()
        for dp_id, model_results in results:
            for mt, path in model_results:
                status = f"OK: {path}" if path else "FAILED"
                print(f"  dp={dp_id} {mt.value}: {status}")


if __name__ == "__main__":
    main()
