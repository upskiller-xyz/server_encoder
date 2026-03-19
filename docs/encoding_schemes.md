# Encoding Schemes

The server supports five encoding schemes selected via the `encoding_scheme` field in the request (default: `v2`).

## Summary

| Scheme | Output dtype | Shape | Obstruction bar | Description |
|--------|-------------|-------|-----------------|-------------|
| `v1` | `uint8` | `(128, 128, 4)` | ✅ right edge (4 px) | Legacy RGB-style channel mapping |
| `v2` | `uint8` | `(128, 128, 4)` | ✅ right edge (4 px) | **Default.** HSV-style channel mapping |
| `v3` | `uint8` | `(128, 128, 4)` | ❌ | HSV-style, no obstruction bar |
| `v4` | `uint8` | `(128, 128, 4)` | Bounding-box vector | HSV-style, compact obstruction representation |
| `v5` | `float32` | `(128, 128, 1)` | ❌ | Geometry-only mask; no parameter encoding |

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
    "encoding_scheme": "v5",   # "v1" | "v2" | "v3" | "v4" | "v5"
    "parameters": { ... }
}
```

The default is `v2` when the field is omitted.
