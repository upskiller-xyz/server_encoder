"""
Alignment validation: encoder input vs GT simulation results comparison.

Generates comparison images for N random datapoints x 4 model types showing:
- Left panel: encoder RGBA image (rendered as RGB)
- Right panel: GT simulation heatmap (DF or DA values, colormapped)

Also performs mask alignment checks and produces decoded values CSV.

Usage:
    python scripts/validate_alignment.py --db ../dataset.sqlite
    python scripts/validate_alignment.py --db ../dataset.sqlite --n 30
    python scripts/validate_alignment.py --db ../dataset.sqlite --model-types df_default da_default
    python scripts/validate_alignment.py --db ../dataset.sqlite --no-obstruction
"""
import os
import sys
import json
import sqlite3
import random
import math
import csv
import logging
import argparse
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import cv2
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.affinity import rotate as shapely_rotate

# ---------------------------------------------------------------------------
# Import juggling: obstruction first, then encoder (same namespace "src.*")
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ENCODER_ROOT = os.path.join(_SCRIPT_DIR, "..")
_OBSTRUCTION_ROOT = os.path.join(_SCRIPT_DIR, "..", "..", "server_obstruction")

sys.path.insert(0, _OBSTRUCTION_ROOT)

from src.components.geometry import Mesh
from src.components.models import Window as ObsWindow
from src.components.calculators.intersection_calculator import IntersectionCalculator
from src.components.filter import CoarseTriangleFilter, HeightTriangleFilter
from src.server.base.constants import ANGLES

_obs_keys = [k for k in sys.modules if k.startswith("src")]
_obs_backup = {k: sys.modules.pop(k) for k in _obs_keys}
sys.path.remove(_OBSTRUCTION_ROOT)

sys.path.insert(0, _ENCODER_ROOT)

from src.components.enums import ModelType, EncodingScheme, ParameterName
from src.components.image_builder import RoomImageBuilder, RoomImageDirector
from src.components.geometry import (
    RoomPolygon, WindowGeometry, ImageDimensions,
    Point2D, GeometryOps, GeometryAdapter
)
from src.components.graphics_constants import GRAPHICS_CONSTANTS
from src.server.services.logging import StructuredLogger
from src.server.enums import LogLevel

logger = logging.getLogger(__name__)

NUM_OBSTRUCTION_DIRECTIONS = 64
OBSTRUCTION_START_DEG = 17.5
OBSTRUCTION_END_DEG = 162.5
IMAGE_SIZE = 128


# =============================================================================
# RGB Decoder: converts pixel values back to physical values
# =============================================================================
@dataclass
class DecoderConfig:
    """Config for decoding a single channel back to physical value."""
    param_name: str
    min_val: float
    max_val: float
    out_min: float
    out_max: float

    def decode(self, pixel: int) -> float:
        output_normalized = pixel / 255.0
        out_range = self.out_max - self.out_min
        if abs(out_range) < 1e-9:
            return self.min_val
        normalized = (output_normalized - self.out_min) / out_range
        val_range = self.max_val - self.min_val
        return self.min_val + normalized * val_range


# Channel mapping: HSV scheme (R, G, B, A) per region
DECODERS = {
    "background": [
        DecoderConfig("terrain_reflectance",         0.0,  1.0,  0.0, 1.0),   # R
        DecoderConfig("floor_height_above_terrain",  0.0, 10.0,  0.1, 1.0),   # G
        DecoderConfig("facade_reflectance",          0.0,  1.0,  0.0, 1.0),   # B
        DecoderConfig("window_direction_angle_rad",  0.0, 2*math.pi, 0.0, 1.0),  # A
    ],
    "room": [
        DecoderConfig("wall_reflectance",            0.0,  1.0,  0.0, 1.0),   # R
        DecoderConfig("floor_reflectance",           0.0,  1.0,  0.0, 1.0),   # G
        DecoderConfig("height_roof_over_floor_m",    0.0, 30.0,  0.0, 1.0),   # B
        DecoderConfig("ceiling_reflectance",         0.5,  1.0,  0.0, 1.0),   # A
    ],
    "window": [
        DecoderConfig("window_height_m",             0.2,  5.0, 0.99, 0.01),  # R (reversed)
        DecoderConfig("window_frame_ratio",          1.0,  0.0,  0.0, 1.0),   # G (reversed)
        DecoderConfig("window_sill_height_m",        0.0,  5.0,  0.0, 1.0),   # B
        DecoderConfig("window_frame_reflectance",    0.0,  1.0,  0.0, 1.0),   # A
    ],
    "obstruction": [
        DecoderConfig("zenith_angle_deg",            0.0, 70.0,  0.2, 0.8),   # R
        DecoderConfig("context_reflectance",         0.1,  0.6,  0.0, 1.0),   # G
        DecoderConfig("horizon_angle_deg",           0.0, 90.0,  0.0, 1.0),   # B
        DecoderConfig("balcony_reflectance",         0.0,  1.0,  0.0, 1.0),   # A
    ],
}


def decode_rgba(pixel: np.ndarray, region: str) -> Dict[str, float]:
    """Decode a single RGBA pixel to physical values for a given region."""
    decoders = DECODERS[region]
    result = {}
    for i, dec in enumerate(decoders):
        result[dec.param_name] = round(dec.decode(int(pixel[i])), 4)
    return result


# =============================================================================
# Model type configuration (mirrors generate_training_data.py)
# =============================================================================
class ReflectanceSource(Enum):
    """Which reflectance set to use from meta_json."""
    DEFAULT = "default"
    CUSTOM = "custom"


MODEL_TYPE_CONFIG: Dict[ModelType, Dict[str, Any]] = {
    ModelType.DF_DEFAULT: {"reflectance_source": ReflectanceSource.DEFAULT, "uses_orientation": False},
    ModelType.DF_CUSTOM: {"reflectance_source": ReflectanceSource.CUSTOM, "uses_orientation": False},
    ModelType.DA_DEFAULT: {"reflectance_source": ReflectanceSource.DEFAULT, "uses_orientation": True},
    ModelType.DA_CUSTOM: {"reflectance_source": ReflectanceSource.CUSTOM, "uses_orientation": True},
}


# =============================================================================
# Obstruction Calculator
# =============================================================================
@dataclass
class ObstructionAngles:
    """Per-window obstruction angles (geometry-only, independent of reflectance)."""
    horizon: List[float]
    zenith: List[float]


@dataclass
class ObstructionReflectances:
    """Per-source reflectance values for obstruction encoding."""
    context: float
    balcony: float


class ObstructionCalculator:
    """Calculates horizon/zenith angles from geometry."""

    def __init__(self, meta_json: dict):
        geometry = meta_json.get("geometry", {})
        context_verts = self._unwrap("context_buildings",
                                     geometry.get("context_buildings", []))
        facade_verts = geometry.get("facade", [])
        horizon_verts = (context_verts or []) + (facade_verts or [])
        self._horizon_mesh = self._build_mesh(horizon_verts)
        self._zenith_mesh = self._build_mesh(geometry.get("balcony_above", []))

        reflectance = meta_json.get("reflectance", {})
        default_refl = reflectance.get("default", {})
        custom_refl = reflectance.get("custom", {})

        self._reflectances = {
            ReflectanceSource.DEFAULT: ObstructionReflectances(
                context=float(default_refl.get("context_buildings", 0.6)),
                balcony=float(default_refl.get("balcony_ceiling", 0.8)),
            ),
            ReflectanceSource.CUSTOM: ObstructionReflectances(
                context=float(self._unwrap(
                    "context_buildings",
                    custom_refl.get("context_buildings",
                                    default_refl.get("context_buildings", 0.6))
                )),
                balcony=float(custom_refl.get(
                    "balcony_ceiling",
                    default_refl.get("balcony_ceiling", 0.8)
                )),
            ),
        }

    @staticmethod
    def _unwrap(key, value):
        if isinstance(value, dict) and key in value and len(value) == 1:
            return value[key]
        return value

    @staticmethod
    def _build_mesh(verts):
        if not verts or len(verts) < 3:
            return None
        try:
            return Mesh.from_vertices(verts)
        except (ValueError, TypeError):
            return None

    def get_reflectances(self, source: ReflectanceSource) -> ObstructionReflectances:
        return self._reflectances[source]

    def calculate(self, win_data, polygon_data) -> ObstructionAngles:
        if self._horizon_mesh is None and self._zenith_mesh is None:
            return ObstructionAngles(
                horizon=[0.0] * NUM_OBSTRUCTION_DIRECTIONS,
                zenith=[0.0] * NUM_OBSTRUCTION_DIRECTIONS,
            )

        obs_window = ObsWindow.from_endpoints(
            x1=float(win_data["x1"]), y1=float(win_data["y1"]),
            z1=float(win_data["z1"]), x2=float(win_data["x2"]),
            y2=float(win_data["y2"]), z2=float(win_data["z2"]),
            direction_angle=float(win_data.get("direction_angle", 0)),
            room_polygon=polygon_data,
        )
        horizons, zeniths = self._sweep(obs_window)
        return ObstructionAngles(horizon=horizons, zenith=zeniths)

    def _sweep(self, window):
        h_coarse = None
        if self._horizon_mesh is not None:
            h_coarse = Mesh(CoarseTriangleFilter.call(
                self._horizon_mesh.triangles, window))
        z_coarse = None
        if self._zenith_mesh is not None:
            z_coarse = Mesh(CoarseTriangleFilter.call(
                self._zenith_mesh.triangles, window))

        normal_arr = window.normal.to_array()
        base_angle = math.atan2(normal_arr[1], normal_arr[0])
        start_rad = math.radians(OBSTRUCTION_START_DEG)
        end_rad = math.radians(OBSTRUCTION_END_DEG)
        step = (end_rad - start_rad) / (NUM_OBSTRUCTION_DIRECTIONS - 1)

        horizons, zeniths = [], []
        for i in range(NUM_OBSTRUCTION_DIRECTIONS):
            rel = start_rad + i * step
            abs_a = base_angle - math.pi * 0.5 + rel
            rotated = ObsWindow.set_angle(window, abs_a)

            if h_coarse and h_coarse.triangles:
                h_dir = Mesh(HeightTriangleFilter.call(
                    h_coarse.triangles, rotated, ANGLES.HORIZON))
            else:
                h_dir = Mesh(())
            if z_coarse and z_coarse.triangles:
                z_dir = Mesh(HeightTriangleFilter.call(
                    z_coarse.triangles, rotated, ANGLES.ZENITH))
            else:
                z_dir = Mesh(())

            h_r = IntersectionCalculator.call(h_dir, rotated, ANGLES.HORIZON)
            z_r = IntersectionCalculator.call(z_dir, rotated, ANGLES.ZENITH)
            horizons.append(h_r.obstruction_angle_degrees)
            zeniths.append(z_r.obstruction_angle_degrees)
        return horizons, zeniths


# =============================================================================
# Simulation data extractor
# =============================================================================
class SimulationDataExtractor:
    """Extracts simulation mesh and results from meta_json."""

    @staticmethod
    def extract_triangles(meta_json: dict) -> Optional[List]:
        """Extract 2D triangle mesh from simulation_grid.mesh_vertices."""
        sim_grid = meta_json.get("simulation_grid", {})
        mesh_verts = sim_grid.get("mesh_vertices")
        if not mesh_verts or not isinstance(mesh_verts, list):
            return None

        triangles = []
        for i in range(0, len(mesh_verts), 3):
            if i + 2 < len(mesh_verts):
                v0, v1, v2 = mesh_verts[i], mesh_verts[i + 1], mesh_verts[i + 2]
                triangles.append([
                    (float(v0[0]), float(v0[1])),
                    (float(v1[0]), float(v1[1])),
                    (float(v2[0]), float(v2[1])),
                ])
        return triangles if triangles else None

    @staticmethod
    def discover_results(meta_json: dict, num_triangles: int) -> Dict[str, List[float]]:
        """Discover all available DF/DA results from simulation_grid."""
        results = {}
        sim_grid = meta_json.get("simulation_grid", {})
        results_section = sim_grid.get("results", {})

        # DF results
        df_section = results_section.get("DF", {})
        for variant in ["default", "custom"]:
            vals = df_section.get(variant)
            if isinstance(vals, list) and len(vals) == num_triangles:
                results[f"df_{variant}"] = [float(v) for v in vals]

        # DA results
        da_section = results_section.get("DA", {})
        if isinstance(da_section, dict):
            for location, loc_data in sorted(da_section.items()):
                if not isinstance(loc_data, dict):
                    continue
                for variant in ["default", "custom"]:
                    var_data = loc_data.get(variant)
                    if not isinstance(var_data, dict):
                        continue
                    for threshold_key in sorted(var_data.keys()):
                        vals = var_data.get(threshold_key)
                        if isinstance(vals, list) and len(vals) == num_triangles:
                            results[f"{threshold_key}_{location}_{variant}"] = [
                                float(v) for v in vals
                            ]

        return results

    @staticmethod
    def find_gt_result(
        model_type: ModelType,
        available: Dict[str, List[float]]
    ) -> Optional[Tuple[str, List[float]]]:
        """Find the best GT result name + values for a model type."""
        if model_type == ModelType.DF_DEFAULT:
            key = "df_default"
            return (key, available[key]) if key in available else None

        if model_type == ModelType.DF_CUSTOM:
            key = "df_custom"
            return (key, available[key]) if key in available else None

        # DA models: search for da_300 first, then fallback thresholds
        variant = "default" if model_type == ModelType.DA_DEFAULT else "custom"
        for prefix in ["da_300", "da_500", "da_100", "da_750"]:
            for name, vals in available.items():
                if name.startswith(prefix) and name.endswith(f"_{variant}"):
                    return (name, vals)
        return None


# =============================================================================
# GT geometry transform + rasterization
# =============================================================================
@dataclass
class TransformParams:
    """Geometry transformation parameters for world-to-pixel mapping."""
    room_mask: np.ndarray
    room_facade_x: int
    window_y_pixels: int
    window_center_x: float
    window_center_y: float
    resolution: float
    rotation_deg: float

    def world_to_pixel(self, world_x: float, world_y: float) -> Tuple[int, int]:
        dx = world_x - self.window_center_x
        dy = world_y - self.window_center_y
        dx_px = round(dx / self.resolution)
        dy_px = round(dy / self.resolution)
        pixel_x = self.room_facade_x + dx_px
        pixel_y = self.window_y_pixels - dy_px
        return (pixel_x, pixel_y)


def compute_transform(
    room_polygon_data: list,
    win_data: dict,
    image_size: int = 128
) -> TransformParams:
    """Compute geometry transformation (mirrors encoder's to_pixel_array + GT generator)."""
    polygon = RoomPolygon.from_dict(room_polygon_data)
    direction_angle = float(win_data.get("direction_angle", 0))
    rotation_deg = -(direction_angle * 180 / math.pi)
    origin = Point2D(0, 0)

    rotated_poly = polygon.rotate(rotation_deg, origin)
    win_geom = WindowGeometry(
        float(win_data["x1"]), float(win_data["y1"]), 0,
        float(win_data["x2"]), float(win_data["y2"]), 0,
        direction_angle=direction_angle
    )
    rotated_win = win_geom.rotate(rotation_deg, origin)

    # Generate room mask via to_pixel_array (same as encoder)
    pixel_coords = rotated_poly.to_pixel_array(
        window_x1=rotated_win.x1, window_y1=rotated_win.y1,
        window_x2=rotated_win.x2, window_y2=rotated_win.y2,
        image_size=image_size, direction_angle=0.0
    )
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    cv2.fillPoly(mask, pixel_coords, 1)

    # Enforce 2px border (same as encoder's _enforce_border)
    border = GRAPHICS_CONSTANTS.BORDER_PX
    mask[0:border, :] = 0
    mask[image_size - border:image_size, :] = 0
    mask[:, 0:border] = 0

    # Find window center (same logic as encoder's to_pixel_array)
    rotated_room_poly = ShapelyPolygon(rotated_poly.get_coords())
    window_temp = WindowGeometry.from_corners(
        rotated_win.x1, rotated_win.y1, 0,
        rotated_win.x2, rotated_win.y2, 0
    )
    w_edges = window_temp.get_candidate_edges()
    tolerance = 0.01

    res = [e for e in w_edges
           if rotated_room_poly.boundary.buffer(tolerance).contains(e)]
    if res:
        edge_coords = list(res[0].coords)
        window_center_x = (edge_coords[0][0] + edge_coords[1][0]) / 2
        window_center_y = (edge_coords[0][1] + edge_coords[1][1]) / 2
    else:
        window_center_x = (rotated_win.x1 + rotated_win.x2) / 2
        window_center_y = (rotated_win.y1 + rotated_win.y2) / 2

    # Wall thickness from rotated window with direction_angle=0
    rotated_win_geom = WindowGeometry(
        rotated_win.x1, rotated_win.y1, 0,
        rotated_win.x2, rotated_win.y2, 0,
        direction_angle=0.0
    )
    wall_thickness_px = rotated_win_geom.wall_thickness_px

    window_left_edge_x = image_size - GRAPHICS_CONSTANTS.WINDOW_OFFSET_PX - wall_thickness_px
    room_facade_x = window_left_edge_x
    window_y_pixels = image_size // 2
    resolution = GRAPHICS_CONSTANTS.get_resolution(image_size)

    return TransformParams(
        room_mask=mask,
        room_facade_x=room_facade_x,
        window_y_pixels=window_y_pixels,
        window_center_x=window_center_x,
        window_center_y=window_center_y,
        resolution=resolution,
        rotation_deg=rotation_deg,
    )


def rotate_triangles(triangles_2d: List, direction_angle: float) -> List:
    """Rotate triangles by -direction_angle to align with encoder's rotation."""
    rotation_deg = -(direction_angle * 180 / math.pi)
    rotated = []
    for tri in triangles_2d:
        shapely_tri = ShapelyPolygon(tri)
        rotated_tri = shapely_rotate(shapely_tri, rotation_deg, origin=(0, 0))
        coords = list(rotated_tri.exterior.coords)[:-1]
        rotated.append([(c[0], c[1]) for c in coords])
    return rotated


def rasterize_gt(
    triangles_2d: List,
    values: List[float],
    transform: TransformParams
) -> np.ndarray:
    """Rasterize simulation values onto pixel grid (same as GT generator)."""
    size = transform.room_mask.shape[0]
    output = np.full((size, size), np.nan, dtype=np.float32)
    room_mask = transform.room_mask.astype(bool)

    for tri_idx, tri in enumerate(triangles_2d):
        val = values[tri_idx]
        pixel_verts = []
        for vx, vy in tri:
            px, py = transform.world_to_pixel(vx, vy)
            pixel_verts.append([px, py])

        pts = np.array(pixel_verts, dtype=np.int32)
        tri_mask = np.zeros((size, size), dtype=np.uint8)
        cv2.fillConvexPoly(tri_mask, pts, 1)

        valid = (tri_mask > 0) & room_mask
        output[valid] = val

    return output


# =============================================================================
# GT heatmap rendering
# =============================================================================
def gt_to_heatmap(
    gt_array: np.ndarray,
    vmin: float = 0,
    vmax: Optional[float] = None
) -> Tuple[np.ndarray, float, float]:
    """Convert float32 GT array to BGR heatmap image.

    Returns:
        (heatmap_bgr, vmin_used, vmax_used)
    """
    valid_vals = gt_array[~np.isnan(gt_array)]
    if vmax is None:
        vmax = float(np.max(valid_vals)) if len(valid_vals) > 0 else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    nan_mask = np.isnan(gt_array)
    normalized = np.clip((gt_array - vmin) / (vmax - vmin), 0, 1)
    normalized[nan_mask] = 0
    gray = (normalized * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
    heatmap[nan_mask] = [40, 40, 40]

    # Draw 6px colorbar at right edge (y=10 to y=height-10)
    h = gt_array.shape[0]
    bar_top, bar_bot = 10, h - 10
    bar_h = bar_bot - bar_top
    if bar_h > 0:
        gradient = np.linspace(1, 0, bar_h).reshape(-1, 1)
        gradient_u8 = (gradient * 255).astype(np.uint8)
        bar_colored = cv2.applyColorMap(gradient_u8, cv2.COLORMAP_VIRIDIS)
        bar_strip = np.repeat(bar_colored, 6, axis=1)
        x_start = gt_array.shape[1] - 8
        x_end = x_start + 6
        if x_end <= gt_array.shape[1]:
            heatmap[bar_top:bar_bot, x_start:x_end] = bar_strip

    return heatmap, vmin, vmax


# =============================================================================
# Encoder parameter extraction (supports all model types)
# =============================================================================
REFLECTANCE_KEYS = {
    "wall_reflectance": "walls",
    "floor_reflectance": "floor",
    "ceiling_reflectance": "ceiling",
    "facade_reflectance": "facade",
    "terrain_reflectance": "terrain",
    "window_frame_reflectance": "window_frame",
}
REFLECTANCE_DEFAULTS = {
    "wall_reflectance": 0.7, "floor_reflectance": 0.3,
    "ceiling_reflectance": 0.8, "facade_reflectance": 0.3,
    "terrain_reflectance": 0.2, "window_frame_reflectance": 0.5,
}


def extract_params(
    meta_json: dict,
    model_type: ModelType,
    obs_angles: Optional[Dict[str, ObstructionAngles]] = None,
    obs_calc: Optional[ObstructionCalculator] = None,
) -> Dict[str, Any]:
    """Extract encoder params for a model type from meta_json."""
    config = MODEL_TYPE_CONFIG[model_type]
    refl_source = config["reflectance_source"]
    uses_orientation = config["uses_orientation"]

    room = meta_json.get("room", {})
    reflectance = meta_json.get("reflectance", {})
    refl = reflectance.get(refl_source.value, reflectance.get("default", {}))

    params: Dict[str, Any] = {
        "room_polygon": room.get("room_polygon", []),
        "floor_height_above_terrain": float(room.get("floor_height_above_terrain", 0)),
        "height_roof_over_floor": float(room.get("height_roof_over_floor", 2.7)),
    }

    # Clip height
    hrof = params["height_roof_over_floor"]
    if hrof < 15.0:
        params["height_roof_over_floor"] = 15.0
    elif hrof > 30.0:
        params["height_roof_over_floor"] = 30.0

    for param_name, meta_key in REFLECTANCE_KEYS.items():
        params[param_name] = float(refl.get(meta_key, REFLECTANCE_DEFAULTS[param_name]))

    # Obstruction reflectances for this model type
    obs_refl = None
    if obs_calc is not None:
        obs_refl = obs_calc.get_reflectances(refl_source)

    windows: Dict[str, Dict] = {}
    for win_id, win_data in sorted(meta_json.get("window", {}).items()):
        if not isinstance(win_data, dict):
            continue
        direction_angle = float(win_data.get("direction_angle", 0))
        win_params: Dict[str, Any] = {
            "x1": float(win_data["x1"]), "y1": float(win_data["y1"]),
            "z1": float(win_data["z1"]), "x2": float(win_data["x2"]),
            "y2": float(win_data["y2"]), "z2": float(win_data["z2"]),
            "window_frame_ratio": float(win_data.get("frame_ratio", 0.15)),
            "direction_angle": direction_angle,
            "wall_thickness": float(win_data.get("wall_thickness", 0.3)),
            "window_frame_reflectance": params["window_frame_reflectance"],
        }

        if uses_orientation:
            win_params["window_direction_angle"] = direction_angle

        if obs_angles and win_id in obs_angles:
            angles = obs_angles[win_id]
            win_params["horizon"] = angles.horizon
            win_params["zenith"] = angles.zenith
            win_params["context_reflectance"] = obs_refl.context if obs_refl else 0.6
            win_params["balcony_reflectance"] = obs_refl.balcony if obs_refl else 0.8
        else:
            win_params["horizon"] = 0.0
            win_params["zenith"] = 0.0
            win_params["context_reflectance"] = 0.6
            win_params["balcony_reflectance"] = 0.8

        windows[win_id] = win_params
    params["windows"] = windows
    return params


# =============================================================================
# Comparison image builder
# =============================================================================
def rgba_to_rgb(image: np.ndarray) -> np.ndarray:
    """Render RGBA image as RGB with white background."""
    h, w = image.shape[:2]
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(float) / 255.0
        rgb = image[:, :, :3].astype(float)
        bg = np.full_like(rgb, 255.0)
        return (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
    return image[:, :, :3].copy()


def create_comparison_image(
    encoder_image: np.ndarray,
    gt_heatmap: Optional[np.ndarray],
    dp_id: int,
    win_id: str,
    model_type: ModelType,
    gt_result_name: Optional[str],
    vmin: float,
    vmax: float,
) -> np.ndarray:
    """
    Create side-by-side comparison:
    Left = encoder RGBA rendered as RGB
    Right = GT simulation heatmap (or "no data" placeholder)
    """
    h, w = encoder_image.shape[:2]

    # Left panel: RGBA to RGB (OpenCV BGR)
    left_rgb = rgba_to_rgb(encoder_image)
    left_bgr = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR)

    # Right panel: GT heatmap or placeholder
    if gt_heatmap is not None:
        right = gt_heatmap.copy()
    else:
        right = np.full((h, w, 3), 40, dtype=np.uint8)
        cv2.putText(right, "No simulation", (10, h // 2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        cv2.putText(right, "data available", (10, h // 2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    # Combine with separator
    sep = np.full((h, 4, 3), 128, dtype=np.uint8)
    combined = np.hstack([left_bgr, sep, right])

    # Labels
    label_top = f"dp={dp_id} win={win_id}"
    cv2.putText(combined, label_top, (4, 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1)
    cv2.putText(combined, model_type.value, (4, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (80, 80, 80), 1)
    cv2.putText(combined, "encoder", (4, h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 0), 1)

    # GT labels
    gt_x = w + 8
    if gt_result_name:
        cv2.putText(combined, gt_result_name, (gt_x, 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1)
        range_label = f"{vmin:.1f}-{vmax:.1f}"
        cv2.putText(combined, range_label, (gt_x, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (160, 160, 160), 1)
    else:
        cv2.putText(combined, "GT", (gt_x, 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1)

    return combined


# =============================================================================
# Sample pixels from each region
# =============================================================================
def sample_region_pixels(image: np.ndarray, mask: Optional[np.ndarray]) -> Dict:
    """Sample one representative pixel from each region of the encoder image."""
    h, w = image.shape[:2]
    result = {}

    # Background: pixel at (5, 5)
    bg_pixel = image[5, 5]
    result["background"] = {"pixel": bg_pixel, "x": 5, "y": 5}

    # Room: center of room mask
    if mask is not None and np.any(mask):
        room_ys, room_xs = np.where(mask)
        cy = int(np.median(room_ys))
        cx = int(np.median(room_xs))
        result["room"] = {"pixel": image[cy, cx], "x": cx, "y": cy}

    # Window: find window area
    center_y = h // 2
    if mask is not None and np.any(mask[center_y]):
        room_right = int(np.where(mask[center_y])[0].max())
        wx = min(room_right + 2, w - 1)
        if not np.array_equal(image[center_y, wx], bg_pixel):
            result["window"] = {"pixel": image[center_y, wx], "x": wx, "y": center_y}

    # Obstruction bar
    dims = ImageDimensions(w)
    obs_x, obs_y, _, obs_y_end = dims.get_obstruction_bar_position()
    obs_cy = (obs_y + obs_y_end) // 2
    if obs_x < w:
        result["obstruction"] = {"pixel": image[obs_cy, obs_x], "x": obs_x, "y": obs_cy}

    return result


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Alignment validation: encoder input vs GT simulation results"
    )
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    parser.add_argument("--n", type=int, default=30, help="Number of random DPs")
    parser.add_argument("--output", default="./validation_alignment",
                        help="Output directory")
    parser.add_argument("--no-obstruction", action="store_true",
                        help="Skip obstruction (faster)")
    parser.add_argument("--model-types", nargs="+", default=None,
                        help="Model types to generate (default: all). "
                        "Options: df_default df_custom da_default da_custom")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    # Parse model types
    if args.model_types:
        valid = {mt.value: mt for mt in ModelType}
        model_types = []
        for v in args.model_types:
            if v not in valid:
                parser.error(f"Unknown model type '{v}'. Valid: {list(valid.keys())}")
            model_types.append(valid[v])
    else:
        model_types = list(ModelType)

    # Connect to DB and pick random DPs
    conn = sqlite3.connect(args.db)
    all_dps = [r[0] for r in conn.execute(
        "SELECT dp FROM datapoints ORDER BY dp").fetchall()]
    conn.close()

    n = min(args.n, len(all_dps))
    selected_dps = sorted(random.sample(all_dps, n))
    print(f"Selected {n} random datapoints from {len(all_dps)} total")
    print(f"DPs: {selected_dps}")
    print(f"Model types: {[mt.value for mt in model_types]}")

    # Setup encoder
    log = StructuredLogger("AlignmentVal", LogLevel.INFO)
    builder = RoomImageBuilder(encoding_scheme=EncodingScheme.HSV)
    director = RoomImageDirector(builder)

    # CSV rows
    csv_rows = []
    csv_header = [
        "dp_id", "window_id", "model_type", "region", "x", "y",
        "R", "G", "B", "A",
        "param_R", "value_R",
        "param_G", "value_G",
        "param_B", "value_B",
        "param_A", "value_A",
    ]

    total_images = 0
    gt_available_count = 0
    gt_missing_count = 0

    for dp_id in selected_dps:
        conn = sqlite3.connect(args.db)
        row = conn.execute(
            "SELECT meta_json FROM datapoints WHERE dp = ?", (dp_id,)
        ).fetchone()
        conn.close()

        if row is None:
            print(f"  dp={dp_id}: not found, skipping")
            continue

        meta_json = json.loads(row[0])
        polygon_data = meta_json.get("room", {}).get("room_polygon", [])

        # Extract simulation data (shared across model types)
        triangles_2d = SimulationDataExtractor.extract_triangles(meta_json)
        sim_results = {}
        if triangles_2d:
            sim_results = SimulationDataExtractor.discover_results(
                meta_json, len(triangles_2d)
            )

        # Compute obstruction angles once (shared across model types)
        obs_angles: Optional[Dict[str, ObstructionAngles]] = None
        obs_calc: Optional[ObstructionCalculator] = None
        if not args.no_obstruction and polygon_data:
            try:
                obs_calc = ObstructionCalculator(meta_json)
                obs_angles = {}
                for win_id, win_data in meta_json.get("window", {}).items():
                    if not isinstance(win_data, dict):
                        continue
                    obs_angles[win_id] = obs_calc.calculate(win_data, polygon_data)
            except Exception as e:
                print(f"  dp={dp_id}: obstruction failed: {e}")
                obs_angles = None
                obs_calc = None

        # Pre-compute transform params per window (shared across model types)
        transforms: Dict[str, TransformParams] = {}
        for win_id, win_data in meta_json.get("window", {}).items():
            if not isinstance(win_data, dict):
                continue
            if polygon_data:
                try:
                    transforms[win_id] = compute_transform(polygon_data, win_data)
                except Exception as e:
                    print(f"  dp={dp_id} win={win_id}: transform failed: {e}")

        # Pre-rotate triangles per window (shared across model types)
        rotated_tris: Dict[str, List] = {}
        if triangles_2d:
            for win_id, win_data in meta_json.get("window", {}).items():
                if not isinstance(win_data, dict):
                    continue
                direction_angle = float(win_data.get("direction_angle", 0))
                try:
                    rotated_tris[win_id] = rotate_triangles(triangles_2d, direction_angle)
                except Exception as e:
                    print(f"  dp={dp_id} win={win_id}: triangle rotation failed: {e}")

        # Generate for each model type
        for model_type in model_types:
            params = extract_params(meta_json, model_type, obs_angles, obs_calc)

            try:
                result = director.construct_multi_window_images(model_type, params)
            except Exception as e:
                print(f"  dp={dp_id} {model_type.value}: encoder failed: {e}")
                continue

            # Find corresponding GT result for this model type
            gt_match = SimulationDataExtractor.find_gt_result(model_type, sim_results)

            for win_id in result.window_ids():
                encoder_image = result.get_image(win_id).astype(np.uint8)
                encoder_mask = result.get_mask(win_id)

                # Generate GT heatmap
                gt_heatmap = None
                gt_result_name = None
                vmin, vmax = 0.0, 1.0

                if gt_match and win_id in transforms and win_id in rotated_tris:
                    gt_result_name, gt_values = gt_match
                    try:
                        gt_array = rasterize_gt(
                            rotated_tris[win_id], gt_values, transforms[win_id]
                        )
                        gt_heatmap, vmin, vmax = gt_to_heatmap(gt_array)
                        gt_available_count += 1
                    except Exception as e:
                        print(f"  dp={dp_id} win={win_id} {model_type.value}: "
                              f"GT rasterize failed: {e}")
                        gt_missing_count += 1
                else:
                    gt_missing_count += 1

                # Create comparison image
                comparison = create_comparison_image(
                    encoder_image, gt_heatmap,
                    dp_id, win_id, model_type,
                    gt_result_name, vmin, vmax
                )
                png_path = os.path.join(
                    args.output,
                    f"dp_{dp_id:04d}_{win_id}_{model_type.value}.png"
                )
                cv2.imwrite(png_path, comparison)

                # Decode RGB values for CSV (only for first model type to avoid redundancy)
                if model_type == model_types[0]:
                    samples = sample_region_pixels(encoder_image, encoder_mask)
                    for region_name, sample in samples.items():
                        pixel = sample["pixel"]
                        decoded = decode_rgba(pixel, region_name)
                        decoders = DECODERS[region_name]

                        csv_rows.append([
                            dp_id, win_id, model_type.value, region_name,
                            sample["x"], sample["y"],
                            int(pixel[0]), int(pixel[1]),
                            int(pixel[2]), int(pixel[3]),
                            decoders[0].param_name, decoded[decoders[0].param_name],
                            decoders[1].param_name, decoded[decoders[1].param_name],
                            decoders[2].param_name, decoded[decoders[2].param_name],
                            decoders[3].param_name, decoded[decoders[3].param_name],
                        ])

                total_images += 1
                gt_status = gt_result_name if gt_result_name else "no GT"
                print(f"  dp={dp_id} win={win_id} {model_type.value}: {gt_status}")

    # Save CSV
    csv_path = os.path.join(args.output, "decoded_values.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_rows)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total comparison images: {total_images}")
    print(f"GT available:            {gt_available_count}")
    print(f"GT missing:              {gt_missing_count}")
    print(f"Model types:             {[mt.value for mt in model_types]}")
    print(f"\nOutput: {os.path.abspath(args.output)}/")
    print(f"  - {total_images} comparison PNGs")
    print(f"  - decoded_values.csv ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
