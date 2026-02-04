# Request Schema

## Structure

```json
{
  "model_type": "<model_type>",
  "parameters": {
    "<shared_parameters>": "<values>",
    "windows": {
      "<window_id>": {
        "<window_parameters>": "<values>"
      }
    }
  }
}
```

## Model Type

**Field:** `model_type`
**Type:** String (required)
**Values:**
- `df_default` - Daylight Factor, default materials
- `da_default` - Daylight Autonomy, default materials
- `df_custom` - Daylight Factor, custom materials
- `da_custom` - Daylight Autonomy, custom materials

## Parameters

### Shared Room Parameters

Parameters applied to entire room, placed at root of `parameters` object.

| Parameter | Type | Range | Unit | Required | Default | Description |
|-----------|------|-------|------|----------|---------|-------------|
| `height_roof_over_floor` | float | 0-30 | m | ✓ | - | Ceiling height above floor |
| `floor_height_above_terrain` | float | 0-10 | m | ✓ | - | Floor elevation above ground |
| `room_polygon` | array | - | m | ✓ | - | Room vertices `[[x,y], ...]` |
| `ceiling_reflectance` | float | 0.5-1.0 | - | - | 1.0 | Ceiling material reflectance |
| `horizontal_reflectance` | float | 0-1 | - | - | 1.0 | Floor reflectance |
| `vertical_reflectance` | float | 0-1 | - | - | 1.0 | Wall reflectance |
| `facade_reflectance` | float | 0-1 | - | - | 1.0 | External facade reflectance |
| `terrain_reflectance` | float | 0-1 | - | - | 1.0 | Ground reflectance |

### Window Parameters

Parameters for each window, placed inside `windows.<window_id>` object.

| Parameter | Type | Range | Unit | Required | Default | Description |
|-----------|------|-------|------|----------|---------|-------------|
| `x1` | float | - | m | ✓ | - | Window corner 1, along facade |
| `y1` | float | - | m | ✓ | - | Window corner 1, into room |
| `z1` | float | 0-5 | m | ✓ | - | Window corner 1, height (sill) |
| `x2` | float | - | m | ✓ | - | Window corner 2, along facade |
| `y2` | float | - | m | ✓ | - | Window corner 2, into room |
| `z2` | float | 0-5 | m | ✓ | - | Window corner 2, height (top) |
| `window_sill_height` | float | 0-5 | m | ✓ | - | Height of window sill (z1) |
| `window_frame_ratio` | float | 0-1 | - | ✓ | - | Frame-to-glass ratio |
| `window_height` | float | 0.2-5 | m | ✓ | - | Window height (z2-z1) |
| `horizon` | float/array | 0-90 | deg | ✓ | - | Horizon obstruction angle(s) |
| `zenith` | float/array | 0-70 | deg | ✓ | - | Zenith obstruction angle(s) |
| `window_frame_reflectance` | float | 0-1 | - | - | 0.8 | Frame material reflectance |
| `context_reflectance` | float/array | 0.1-0.6 | - | - | 1.0 | External context reflectance |
| `balcony_reflectance` | float | 0-1 | - | - | 0.8 | Balcony material reflectance |

**Array Parameters:**
`horizon`, `zenith`, and `context_reflectance` can be:
- Single float (applied to all 64 analysis directions)
- Array of 64 floats (one per direction)

**Model-Specific:**
`window_direction_angle` (0-2π radians, math convention: 0=East, CCW) is auto-populated from each window's `direction_angle` for DA models. No need to pass it explicitly.

## Room Polygon Format

Array of `[x, y]` coordinate pairs in meters, defining room boundary in plan view.

**Example:**
```json
"room_polygon": [
  [0, 0],
  [5, 0],
  [5, 4],
  [2.5, 4],
  [2.5, 6],
  [0, 6]
]
```

**Notes:**
- Coordinates relative to facade
- Polygon can be concave (L-shaped, etc.)
- Automatically clipped to image boundaries
- Window direction angle is automatically calculated from the polygon edge containing the window
- Room geometry is automatically rotated so the window faces horizontally in the encoded image

## Window ID Format

**Type:** String
**Pattern:** Any valid JSON key
**Example:** `"main_window"`, `"window_1"`, `"south_window"`

Used as:
1. Key in `windows` object
2. Filename in multi-window ZIP output (`{window_id}.png`)

## Complete Examples

### Single Window - DF Default

```json
{
  "model_type": "df_default",
  "parameters": {
    "height_roof_over_floor": 2.7,
    "floor_height_above_terrain": 3.0,
    "room_polygon": [[0, 0], [5, 0], [5, 4], [0, 4]],
    "windows": {
      "main_window": {
        "x1": -0.6, "y1": 0.0, "z1": 0.9,
        "x2": 0.6, "y2": 0.0, "z2": 2.4,
        "window_sill_height": 0.9,
        "window_frame_ratio": 0.15,
        "window_height": 1.5,
        "horizon": 15.0,
        "zenith": 10.0
      }
    }
  }
}
```

### Multiple Windows - Different Facades

```json
{
  "model_type": "df_custom",
  "parameters": {
    "height_roof_over_floor": 2.7,
    "floor_height_above_terrain": 3.0,
    "room_polygon": [[0, 0], [5, 0], [5, 4], [0, 4]],
    "ceiling_reflectance": 0.8,
    "horizontal_reflectance": 0.5,
    "vertical_reflectance": 0.6,
    "windows": {
      "south_window": {
        "x1": -0.6, "y1": 0.0, "z1": 0.9,
        "x2": 0.6, "y2": 0.0, "z2": 2.4,
        "window_sill_height": 0.9,
        "window_frame_ratio": 0.15,
        "window_height": 1.5,
        "horizon": 15.0,
        "zenith": 10.0,
        "window_frame_reflectance": 0.7
      },
      "west_window": {
        "x1": 0.0, "y1": -0.5, "z1": 1.0,
        "x2": 0.0, "y2": 0.5, "z2": 2.0,
        "window_sill_height": 1.0,
        "window_frame_ratio": 0.2,
        "window_height": 1.0,
        "horizon": 25.0,
        "zenith": 15.0,
        "context_reflectance": 0.3
      }
    }
  }
}
```

### DA Custom with Array Obstructions

```json
{
  "model_type": "da_custom",
  "parameters": {
    "height_roof_over_floor": 3.0,
    "floor_height_above_terrain": 0.0,
    "room_polygon": [[0, 0], [6, 0], [6, 5], [0, 5]],
    "facade_reflectance": 0.5,
    "terrain_reflectance": 0.3,
    "windows": {
      "north_window": {
        "x1": -1.0, "y1": 0.0, "z1": 0.8,
        "x2": 1.0, "y2": 0.0, "z2": 2.2,
        "window_sill_height": 0.8,
        "window_frame_ratio": 0.1,
        "window_height": 1.4,
        "horizon": [10, 12, 15, ...],
        "zenith": [8, 9, 10, ...],
        "context_reflectance": [0.2, 0.25, 0.3, ...]
      }
    }
  }
}
```

## Validation Rules

1. **Model type** must be one of four valid values
2. **Windows** object must contain at least one window
3. **Required parameters** must be present for each model type
4. **Numeric ranges** validated per parameter (see table above)
5. **Array parameters** must have exactly 64 elements if provided as array
6. **Room polygon** must have at least 3 vertices
7. **Window coordinates** must form valid bounding box (x1≤x2, y1≤y2, z1≤z2)

## Error Examples

**Missing required field:**
```json
{
  "error": "Missing required parameters: window_sill_height"
}
```

**Out of range:**
```json
{
  "error": "Parameter 'window_height' value 6.0 outside valid range [0.2, 5.0]"
}
```

**No windows:**
```json
{
  "error": "Missing 'windows' field in parameters. At least one window must be provided."
}
```
