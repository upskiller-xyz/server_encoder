# Encoding Schemes

The server supports encoding schemes selected via the `encoding_scheme` field in the request (default: `v2`).

## Summary

| Scheme | Output dtype | Shape | Obstruction | Static vector | Description |
|--------|-------------|-------|-------------|---------------|-------------|
| `v1` | `uint8` | `(128, 128, 4)` | Right-edge bar (4 px) | — | Legacy RGB-style channel mapping |
| `v2` | `uint8` | `(128, 128, 4)` | Right-edge bar (4 px) | — | **Default.** HSV-style channel mapping |
| `v3` | `uint8` | `(128, 128, 4)` | None | — | HSV-style, no obstruction bar |
| `v4` | `uint8` | `(128, 128, 4)` | Bounding-box fill | — | HSV-style, compact obstruction |
| `v5` | `float32` | `(128, 128, 1)` | None | — | Geometry-only mask; no parameter encoding |
| `v6` | `float32` | `(128, 128, 1)` | Bounding-box fill | ✅ scalar params | Geometry mask + bounding-box obstruction + scalar static vector |
| `v7` | `uint8` | `(128, 128, 4)` | Bounding-box fill | — | Like V4; height params use fixed defaults (roof=15 m, floor=0 m) |
| `v8` | `uint8` | `(128, 128, 4)` | Bounding-box fill | — | Like V7; heights supplied per-window via `height_vector` |
| `v9` | `uint8` | `(128, 128, 3)` | Bounding-box fill | — | Like V7; alpha channel dropped |
| `v10` | `uint8` | `(128, 128, 3)` | Bounding-box fill | — | Like V8; alpha channel dropped |
| `v11` | `uint8` | `(128, 128, 3)` | Bounding-box fill | — | Like V10; obstruction encodes gap/midpoint instead of zenith/horizon |
| `v12` | `uint8` | `(128, 128, 4)` | Window-projection rectangle | ✅ 7-dim material vector | Window-height rectangle filled with V8 obstruction; window stripe kept |
| `v13` | `uint8` | `(128, 128, 4)` | Window-projection rectangle | ✅ 7-dim material vector | Like V12; window stripe (wall-thickness indicator) removed |

---

## V1 — RGB Encoding

**Default in:** legacy deployments
**Output:** `uint8` RGBA image `(128, 128, 4)`

Uses the original RGB-style parameter-to-channel mapping. Includes a 4-pixel-wide **obstruction bar** at the right edge of the image (pixels x=124–127) encoding horizon and zenith angles.

### Channel Mapping

| Region | Red | Green | Blue | Alpha |
|--------|-----|-------|------|-------|
| Background | Facade reflectance | Floor height above terrain | Terrain reflectance | Window orientation |
| Room | Height roof over floor | Horizontal reflectance | Vertical reflectance | Ceiling reflectance |
| Window | Sill height (z1) | Frame ratio (reversed) | Window height (reversed) | Frame reflectance |
| Obstruction bar | Horizon angle | Context reflectance | Zenith angle | Balcony reflectance |

### Required parameters

`height_roof_over_floor`, `floor_height_above_terrain`, `room_polygon`, `x1`, `y1`, `z1`, `x2`, `y2`, `z2`, `window_sill_height`, `window_frame_ratio`, `window_height`, `horizon`, `zenith`

---

## V2 — HSV Encoding (default)

**Default scheme** when `encoding_scheme` is omitted.
**Output:** `uint8` RGBA image `(128, 128, 4)`

Uses the HSV-style channel mapping — the name refers to the parameter assignment convention (Hue/Saturation/Value), not a color-space conversion. All channels are stored as raw `uint8` values. Includes the same obstruction bar as V1.

### Channel Mapping

Identical region layout as V1; differs in which physical parameter is assigned to each channel within each region. See `src/core/enums.py → REGION_CHANNEL_MAPPING_V2` for the complete map.

### Required parameters

Same as V1.

---

## V3 — HSV Encoding, No Obstruction Bar

**Output:** `uint8` RGBA image `(128, 128, 4)`

Identical to V2 except the **obstruction bar is omitted**. The right edge pixels are filled with background values. Useful when obstruction data is not available or not needed.

### Required parameters

Same as V1/V2 (obstruction parameters `horizon`, `zenith` are accepted but not drawn into the image).

---

## V4 — HSV Encoding, Bounding-Box Obstruction

**Output:** `uint8` RGBA image `(128, 128, 4)`

Like V3 in layout, but obstruction is encoded as a **compact bounding-box multiplication** applied to the window region rather than a separate bar. This condenses the obstruction representation without consuming the right-edge columns.

### Required parameters

Same as V1/V2.

---

## V5 — Geometry-Only Float32 Mask

**Output:** `float32` single-channel array `(128, 128, 1)`

Encodes only room geometry — no reflectance, height, or obstruction parameters are encoded. Each pixel gets a fixed intensity based on which region it belongs to:

| Region | Value |
|--------|-------|
| Background | `0.0` |
| Room polygon | `1.0` |
| Window | `0.6` |

### Key differences from V1–V4

- Output dtype is `float32` (not `uint8`).
- Only **one channel** is returned.
- Only **geometry parameters** are required (`room_polygon` and window bounding box `x1/y1/z1/x2/y2/z2`).
- `height_roof_over_floor` and `floor_height_above_terrain` are **not required** and are ignored if supplied.
- `encode_room_image()` (PNG endpoint) is not supported — use `encode_room_image_arrays()` (NPZ endpoint) only.

### Required parameters

`room_polygon`, `x1`, `y1`, `z1`, `x2`, `y2`, `z2`

### Example request

```json
{
  "model_type": "df_default",
  "encoding_scheme": "v5",
  "parameters": {
    "room_polygon": [[0, 0], [5, 0], [5, 4], [0, 4]],
    "windows": {
      "main_window": {
        "x1": -0.6, "y1": 0.0, "z1": 0.9,
        "x2":  0.6, "y2": 0.0, "z2": 2.4
      }
    }
  }
}
```

### Reading the response

```python
import requests, numpy as np, io

response = requests.post("http://localhost:8081/encode", json=payload)
npz = np.load(io.BytesIO(response.content))

image = npz["main_window_image"]  # (128, 128, 1) float32
mask  = npz["main_window_mask"]   # (128, 128) binary room mask

print(image.shape, image.dtype)   # (128, 128, 1) float32
print(np.unique(image.round(2)))  # [0.  0.6 1. ]
```

---

## V6 — Geometry Mask + Bounding-Box Obstruction + Static Vector

**Output:** `float32` single-channel array `(128, 128, 1)` + static scalar vector

Like V5 in image layout (geometry mask only), but applies **bounding-box obstruction** to the window region and returns room/window scalar parameters as a separate static vector (same format as V5's conditioning vector). Use `encode_room_image_arrays()` (NPZ endpoint).

### Required parameters

Same as V5, plus obstruction parameters and room/window scalars.

---

## V7 — HSV, Fixed Height Defaults

**Output:** `uint8` RGBA image `(128, 128, 4)`

Like V4 (bounding-box obstruction) but **`height_roof_over_floor` and `floor_height_above_terrain` use fixed defaults** (15 m and 0 m respectively) when not supplied. Simplifies deployment when height data is unavailable.

### Required parameters

`room_polygon`, `x1`, `y1`, `z1`, `x2`, `y2`, `z2`, `horizon`, `zenith`

---

## V8 — HSV, Per-Window Height Vector

**Output:** `uint8` RGBA image `(128, 128, 4)`

Like V7 but height values are supplied per-window via a `height_vector` field `[height_roof_over_floor, floor_height_above_terrain]` rather than as top-level parameters. Useful when different windows in the same room have different effective heights.

### Required parameters

Same as V7, plus `height_vector` per window.

---

## V9 — HSV, Alpha Dropped (Fixed Heights)

**Output:** `uint8` 3-channel image `(128, 128, 3)`

Like V7 with the alpha channel dropped from the output. Alpha-encoded parameters always use their defaults; no information is lost. Use when the downstream model expects 3 channels.

---

## V10 — HSV, Alpha Dropped (Height Vector)

**Output:** `uint8` 3-channel image `(128, 128, 3)`

Like V8 with the alpha channel dropped. Combines per-window height vectors with 3-channel output.

---

## V11 — HSV, Gap/Midpoint Obstruction

**Output:** `uint8` 3-channel image `(128, 128, 3)`

Like V10 but the obstruction encoding changes: instead of raw **zenith** and **horizon** angles, the obstruction bar encodes **gap** (clear-sky opening) and **midpoint** (vertical centre of the opening). Derived from the same zenith/horizon inputs.

### Required parameters

Same as V10.

---

## V12 — Window-Projection Rectangle + Static Material Vector

**Output:** `uint8` RGBA image `(128, 128, 4)` + `float32` static vector (7 values)

A new spatial obstruction representation. Instead of a bar at the right edge, a **rectangle protrudes leftward from the window stripe** into the room:

- **Width** (horizontal extent) = window height in pixels
- **Height** (vertical extent) = window width in pixels
- Filled uniformly with V8-style RGBA obstruction channel values

Room and window material parameters are returned as a separate 7-element `float32` static vector (`V12_STATIC_PARAMS`) normalised to `[0, 1]`. The **window stripe** (3 px, encodes wall thickness) is kept in the image.

### Static vector fields (in order)

| Index | Parameter | Range |
|-------|-----------|-------|
| 0 | `wall_reflectance` | [0, 1] |
| 1 | `floor_reflectance` | [0, 1] |
| 2 | `ceiling_reflectance` | [0, 1] |
| 3 | `height_roof_over_floor` | [0, 30] → /30 |
| 4 | `window_sill_height` | [0, 5] → /5 |
| 5 | `window_frame_ratio` | reversed: 1 − ratio |
| 6 | `window_frame_reflectance` | [0, 1] |

### Required parameters

Same as V4, plus all material parameters above.

---

## V13 — Window-Projection Rectangle, No Window Stripe

**Output:** `uint8` RGBA image `(128, 128, 4)` + `float32` static vector (7 values)

Identical to V12 except the **window stripe is omitted** — the wall-thickness indicator columns are filled with background values. The projection rectangle and static vector are unchanged.

Use V13 when the model should not receive any explicit wall-thickness signal.

---

## Image Layout

All schemes share the same 128×128 canvas and coordinate system.

```
┌────────────────────────────────────────────────────┐
│ background (params encoded into pixel values)      │
│                                                    │
│     ┌──────────────────────────┐  │    │           │
│     │                          │  │    │  V1/V2    │
│     │   room polygon           │ win   │  obstruct │
│     │   (room params)          │ dow   │  bar 4px  │
│     │                          │  │    │           │
│     └──────────────────────────┘  │    │           │
│                                                    │
└────────────────────────────────────────────────────┘
  ←—————————— 128 pixels ——————————→←12px→←—4px—→
```

- Window is placed 12 pixels from the right edge.
- Obstruction bar (V1/V2 only) occupies the rightmost 4 pixels, vertically centered over 64 rows.
- A 2-pixel background border is enforced on all four sides (C-frame).

---

## Selecting a Scheme

Pass `encoding_scheme` in the request body:

```python
payload = {
    "model_type": "df_default",
    "encoding_scheme": "v12",   # "v1"|"v2"|"v3"|"v4"|"v5"|"v6"|"v7"|"v8"|"v9"|"v10"|"v11"|"v12"|"v13"
    "parameters": { ... }
}
```

The default is `v2` when the field is omitted.
