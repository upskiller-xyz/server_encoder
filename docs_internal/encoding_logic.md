# Room Image Encoding Logic

## Overview

This document explains the complete logic for encoding room geometry and parameters into images for Daylight Factor (DF) and Daylight Autonomy (DA) prediction models.

## 1. Spatial Scale and Resolution

**Base Resolution:** 128 × 128 pixels
- Each pixel represents **0.1m (10cm)** in real-world space
- Corresponds to a physical area of **12.8m × 12.8m** per frame
- Resolution scales proportionally for other image sizes (e.g., 256×256, 512×512, 1024×1024)

## 2. Coordinate System

### Reference Point
- **Origin:** Center of the window, aligned with outer plane of the façade wall
- **Position:** 12 pixels from right edge, vertically centered
- **Orientation:** Window normal vector always points toward right edge of image

### Coordinate Axes
- **X-axis:** Along façade (horizontal in real space) → Vertical in image (Y-axis)
- **Y-axis:** Perpendicular to façade (into room) → Horizontal in image (X-axis, REVERSED: deeper into room = leftward)
- **Z-axis:** Vertical (height above floor) → Encoded in channel values, not spatial position

## 3. Image Layout and Boundaries

### Region Hierarchy (Back to Front)
1. **Background** - Fills entire image
2. **Room Polygon** - Interior space mask
3. **Window** - Vertical bar at fixed position
4. **Obstruction Bar** - At right edge (4 pixels wide)

### Boundary Rules

#### C-Frame Border (2-pixel background border)
The outermost 2 pixel rows/columns must remain background:
- Top 2 rows: Background only
- Bottom 2 rows: Background only
- Left 2 columns: Background only
- Right 2 columns: Obstruction bar + background

#### Room Polygon Clipping
Room polygons are clipped to prevent overlap with:
- **Right boundary:** Room must end at least 2 pixels before obstruction bar
  - At 128×128: Obstruction bar starts at x=124, room clips at x≤121
  - Creates 2-pixel gap (x=122, 123) between room and obstruction bar
  - Total: 6 pixels from right edge to room boundary
- **Other boundaries:** Room must stay within 2-pixel margin from edges

### Multi-Window Support with Facade Rotation

For rooms with windows on different facades (west, north, east instead of south):
- **Each window** gets its own independently encoded image
- **Rotation:** Room polygon and window coordinates are automatically rotated so the window being encoded always faces right
- **Rotation angles:**
  - South facade (0°): No rotation
  - West facade (90°): Rotate -270° (equivalent to +90° CCW)
  - North facade (180°): Rotate -180°
  - East facade (270°): Rotate -90° (equivalent to +270° CCW)
- **Post-processing:** Daylight factor distributions from each window image can be juxtaposed and summed to obtain total DF across the full room

## 4. Region-Specific Encoding

### Background Region

**Coverage:** Entire image except room, window, and obstruction bar

**Parameters (all single float values):**

| Parameter | Channel | Input Range | Normalized Range | Default | Required |
|-----------|---------|-------------|------------------|---------|----------|
| Floor height above terrain | Green | 0-10m | 0.1-1.0 | None | ✓ |
| Facade reflectance | Red | 0-1 | 0-1 | 1.0 | - |
| Terrain reflectance | Blue | 0-1 | 0-1 | 1.0 | - |
| Window orientation | Alpha | 0-360° | 0-1 | 0.8 | - |

**Note:** Window orientation default is 0.8 normalized value, NOT 0.8°

### Room Polygon Region

**Construction:**
- Input: Array of (x, y) coordinates in meters
- Positioning: Rightmost side aligns with left edge of window area
- Transformation: 3D coordinates → 2D top-down view
  - X (along façade) → Image Y (vertical)
  - Y (into room) → Image X (horizontal, reversed: deeper = leftward)
- Clipping: Polygon clipped at boundaries using Shapely intersection

**Parameters (all single float values):**

| Parameter | Channel | Input Range | Normalized Range | Default | Required |
|-----------|---------|-------------|------------------|---------|----------|
| Height roof over floor | Red | 0-30m | 0-1 | None | ✓ |
| Horizontal reflectance | Green | 0-1 | 0-1 | 1.0 | - |
| Vertical reflectance | Blue | 0-1 | 0-1 | 1.0 | - |
| Ceiling reflectance | Alpha | 0.5-1 | 0-1 | 1.0 | - |

### Window Region

**Position:**
- 12 pixels from right edge
- 8 pixels from obstruction bar
- Vertically centered on window center in 3D space

**Appearance:**
- Viewed from top (plan view)
- Appears as vertical line/rectangle
- Horizontal extent: Wall thickness (~0.3m = 3 pixels at base scale)
- Vertical extent: Window width in 3D (x2-x1)

**Input Geometry:**
- Bounding box: (x1, y1, z1) to (x2, y2, z2) in meters
- x1, x2: Along façade
- y1, y2: Into room (not used in top view)
- z1, z2: Height above floor

**Parameters:**

| Parameter | Channel | Input Range | Normalized Range | Reversed | Required |
|-----------|---------|-------------|------------------|----------|----------|
| Sill height (z1) | Red | 0-5m | 0-1 | No | ✓ |
| Frame ratio | Green | 0-1 | 0-1 | Yes | ✓ |
| Window height (z2-z1) | Blue | 0.2-5m | 0.99-0.01 | Yes | ✓ |
| Frame reflectance | Alpha | 0-1 | 0-1 | No | - |

**Reversed Parameters:**
- **Frame ratio:** Input 1 → Output 0, Input 0 → Output 1
- **Window height:** Input 0.2m → Output 0.99, Input 5m → Output 0.01

### Obstruction Bar Region

**Position:**
- Right edge of image
- Width: 4 pixels
- Height: 64 pixels (±32 pixels from center)
- Vertically centered

**Angular Mapping:**
Each of 64 rows represents a vertical analysis plane at specific horizontal azimuth:
- **Central rows:** Perpendicular to façade (0° azimuth)
- **Upper/lower rows:** Up to ±72.5° from center
- **Excluded:** ±17.5° near façade plane (minor influence, typically occluded)

**Parameters:**

| Parameter | Channel | Input Type | Input Range | Normalized Range | Default | Required |
|-----------|---------|------------|-------------|------------------|---------|----------|
| Obstruction angle horizon | Red | Single/Array(64) | 0-90° | 0-1 | None | ✓ |
| Context reflectance | Green | Single/Array(64) | 0.1-0.6 | 0-1 | 1.0 | - |
| Obstruction angle zenith | Blue | Single/Array(64) | 0-70° | 0.2-0.8 | None | ✓ |
| Balcony reflectance | Alpha | Single only | 0-1 | 0-1 | 0.8 | - |

**Array Parameters:** Parameters 1-3 can be:
- Single value (replicated across all 64 rows)
- Array of 64 values (one per horizontal angle)

**Single Parameter:** Balcony reflectance is always a single float

## 5. Model Types

The encoding supports four model variants:

1. **DF Default:** Daylight Factor with fixed default materials (reflectance = 0.8)
2. **DA Default:** Daylight Autonomy with fixed default materials
3. **DF Custom:** Daylight Factor with customizable material reflectances
4. **DA Custom:** Daylight Autonomy with customizable material reflectances

**Alpha Channel Usage:**
- Default models: Alpha channels not used (or use default values)
- Custom models: Alpha channels encode reflectance parameters

## 6. Normalization and Encoding

### General Formula
```
pixel_value = int((normalized_value) * 255)
```

Where `normalized_value` is scaled from input range to [0, 1]

### Special Cases

**Floor height above terrain:**
```
normalized = (input_value - 0) / (10 - 0) * (1.0 - 0.1) + 0.1
```
Maps 0m→0.1, 10m→1.0 (never zero to distinguish from empty pixels)

**Ceiling reflectance:**
```
normalized = (input_value - 0.5) / (1.0 - 0.5)
```
Maps 0.5→0, 1.0→1

**Reversed parameters (frame_ratio, window_height):**
```
normalized = 1.0 - ((input_value - min) / (max - min))
```

## 7. Drawing Order and Compositing

**Rendering sequence:**
1. Initialize image with zeros (RGBA)
2. Draw background (fills entire image)
3. Draw room polygon (overwrites background in room area)
4. Draw window (overwrites room/background in window area)
5. Draw obstruction bar (overwrites everything at right edge)

**C-Frame enforcement:** After all drawing, first/last 2 rows/columns (except obstruction bar) are reset to background values

## 8. Scaling Behavior

All parameters and dimensions scale proportionally with image size:

**Resolution calculation:**
```python
scale = image_size / 128.0
resolution = 0.1 / scale  # meters per pixel
```

**Examples:**
- 128×128: 1 pixel = 0.10m
- 256×256: 1 pixel = 0.05m
- 512×512: 1 pixel = 0.025m
- 1024×1024: 1 pixel = 0.0125m

**Scaled dimensions:**
- Window offset: 12 * scale pixels from right
- Wall thickness: ~0.3m / resolution pixels
- Obstruction bar width: 4 * scale pixels
- Border width: 2 * scale pixels
- Obstruction bar gap: 2 * scale pixels (before obstruction bar)

## 9. Architecture Overview

### Object-Oriented Design

The encoder follows strict OOP principles with clear separation of concerns:

**Core Classes:**
- `RoomPolygon`: Geometry handling, rotation, pixel conversion
- `WindowGeometry`: Window positioning, facade detection, rotation
- `Point2D`: 2D coordinate representation
- `RegionEncoder` (abstract): Base class for all region encoders
  - `BackgroundRegionEncoder`: Background drawing
  - `RoomRegionEncoder`: Room polygon mask creation
  - `WindowRegionEncoder`: Window area drawing
  - `ObstructionBarRegionEncoder`: Obstruction bar drawing
- `RoomImageDirector`: Orchestrates image construction, applies rotation
- `EncodingService`: API endpoint handling, validation

**Design Patterns:**
- **Strategy Pattern:** Channel mappings, validation rules, facade rotations
- **Factory Pattern:** Region encoder creation
- **Builder Pattern:** Image construction (Director)
- **Adapter Pattern:** Parameter transformation and normalization
- **Enumerator Pattern:** All constants and magic strings replaced with enums

### Processing Pipeline

1. **Request Validation** (EncodingService)
   - Validate model type
   - Check required parameters
   - Apply defaults for optional parameters

2. **Parameter Transformation** (RoomImageDirector)
   - Convert input parameters to internal format
   - Detect facade orientation from window coordinates
   - Apply rotation if window not on south facade

3. **Region Encoding** (RegionEncoders)
   - Background: Fill entire image
   - Room: Create mask, clip to boundaries
   - Window: Draw at fixed position
   - Obstruction bar: Draw at right edge

4. **Post-processing**
   - Enforce C-frame border
   - Convert to PNG
   - Return image bytes

## 10. Coordinate Transformations

### 3D World → 2D Image Mapping

**Room vertices:**
```python
# World coordinates: (x_world, y_world) in meters
# Image coordinates: (x_pixel, y_pixel)

# Resolution
scale = image_size / 128.0
resolution = 0.1 / scale  # m/px

# Window center position
window_center_x = (x1 + x2) / 2.0
window_center_y = (y1 + y2) / 2.0

# Room facade position
window_left_edge_x = image_size - 12 - wall_thickness_px
room_facade_x = window_left_edge_x - 1  # 1px gap (C-frame)

# Transform
dx = vertex.x - window_center_x  # Along façade
dy = vertex.y - window_center_y  # Into room

x_pixel = room_facade_x - round(dy / resolution)  # Reversed: deeper = leftward
y_pixel = image_center_y + round(dx / resolution)  # Along façade = vertical
```

**Rotation (for non-south facades):**
```python
from shapely.affinity import rotate as shapely_rotate

# Rotate around origin (0, 0)
rotation_angle = FACADE_ROTATION_MAP[facade_orientation]
rotated_poly = shapely_rotate(polygon, rotation_angle, origin=(0, 0))
```

### Clipping

**Boundary calculation:**
```python
obs_bar_x_start = image_size - 4  # At 128×128: x=124
right_boundary = obs_bar_x_start - 3  # At 128×128: x=121
# This creates 2-pixel gap (x=122, 123) before obstruction bar
```

**Shapely intersection:**
```python
from shapely.geometry import Polygon, box

clip_box = box(0, 0, right_boundary, image_size)
clipped_polygon = room_polygon.intersection(clip_box)
```

## 11. Key Implementation Notes

### C-Frame Border
- First 2 rows/columns must be background
- Ensures background parameters always visible
- Critical for model training (avoids edge artifacts)

### Polygon Clipping
- Uses Shapely geometric operations for precision
- Handles complex polygon shapes (concave, L-shaped, etc.)
- Maintains room position (no shifting, only clipping)

### Facade Rotation
- Automatic detection based on window coordinate spans
- Applied before any pixel conversion
- Both room and window rotated together
- Enables multi-window rooms with windows on different facades

### Vectorized Operations
- Uses numpy for all array operations
- No explicit loops in coordinate transformations
- OpenCV (cv2) for polygon filling

### Validation
- Strategy pattern maps for required vs optional parameters
- Custom validators for each model type
- Clear error messages with parameter names

## Summary

This encoding system transforms 3D room geometry and physical parameters into 2D images where each pixel's RGBA channels encode specific geometric, radiometric, and contextual information. The system supports flexible room geometries, multiple windows on different facades, and various material properties, all while maintaining strict boundary conditions and proportional scaling across different image resolutions.
