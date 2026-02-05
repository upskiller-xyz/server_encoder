"""GhPython: Encoder Input + GT Output Viewer via Server HTTP calls.

Reads a datapoint from SQLite, calls the three pipeline servers
(obstruction, encoder, GT generator) via HTTP, and renders the
encoder input image alongside the GT simulation heatmap as Rhino
meshes with vertex colors.

SETUP  (Grasshopper component)
------
1.  Add a GhPython Script component (or Script component in Rhino 8).
2.  Right-click → Manage "Input parameters".  Add these inputs:
        run              bool       Button or Toggle
        db_path          str        Panel  (path to dataset.sqlite)
        dp_id            int        Slider (integer, e.g. 0-30)
        metric           str        Panel  ("DF" or "DA")
        variant          str        Panel  ("default" or "custom")
        encoder_url      str        Panel  (e.g. "http://localhost:8082")
        obstruction_url  str        Panel  (e.g. "http://localhost:8081")
        gt_url           str        Panel  (e.g. "http://localhost:8083")
3.  Right-click → Manage "Output parameters".  Add:
        meshes           Mesh list  (connect to a Mesh parameter / preview)
        labels           Geometry   (TextDot list — optional)
        info             str        Panel  (detailed log)
4.  Paste this entire file into the script editor.

LAYOUT (in Rhino viewport)
------
Each window produces one row.  Columns (left→right):
  Column 0:  Encoder RGBA image  (128×128 px, 0.1 m/px → 12.8 m)
  Column 1+: GT heatmap per matching result  (DF=grayscale, DA=black-purple-red-yellow-white)
  Colorbar:  Appended right of each GT block

NOTES
-----
-  ALL computation is performed on the servers — nothing is calculated
   locally.  This guarantees the viewer shows exactly what the servers
   produce.
-  IronPython 2.7 (Rhino 7) and CPython 3 (Rhino 8) compatible.
-  No numpy / cv2 / shapely required — NPZ files are parsed with
   zipfile + struct.
"""

# ────────────────────────────────────────────────────────────────────
#  Early bootstrap — always-available logging + safe output defaults
# ────────────────────────────────────────────────────────────────────
_log = []


def log(msg):
    s = str(msg)
    _log.append(s)
    print(s)          # also print to Rhino command line for live debug


def log_section(title):
    _log.append("")
    _log.append("=" * 60)
    _log.append(title)
    _log.append("=" * 60)
    print("=" * 40 + " " + title)


# Guarantee outputs always exist (even if imports crash)
meshes = []
labels = []
info = "Script not yet executed"

# ── Imports ─────────────────────────────────────────────────────────
log_section("IMPORTS")
try:
    import traceback
    log("  traceback OK")
except Exception as _e:
    log("  traceback FAILED: %s" % str(_e))

try:
    import System
    import System.IO
    log("  System / System.IO OK")
except Exception as _e:
    log("  System FAILED: %s" % str(_e))

try:
    from System.Net import WebClient, WebException
    from System.Text import Encoding
    log("  System.Net / System.Text OK")
except Exception as _e:
    log("  System.Net FAILED: %s" % str(_e))

try:
    import json
    log("  json OK")
except Exception as _e:
    log("  json FAILED: %s" % str(_e))

try:
    import zipfile
    log("  zipfile OK")
except Exception as _e:
    log("  zipfile FAILED: %s" % str(_e))

try:
    import struct
    log("  struct OK")
except Exception as _e:
    log("  struct FAILED: %s" % str(_e))

try:
    import re
    log("  re OK")
except Exception as _e:
    log("  re FAILED: %s" % str(_e))

try:
    import os
    import tempfile
    log("  os / tempfile OK")
except Exception as _e:
    log("  os FAILED: %s" % str(_e))

try:
    import math
    log("  math OK")
except Exception as _e:
    log("  math FAILED: %s" % str(_e))

try:
    import Rhino.Geometry as rg
    log("  Rhino.Geometry OK")
except Exception as _e:
    log("  Rhino.Geometry FAILED: %s" % str(_e))

try:
    import System.Drawing as sd
    log("  System.Drawing OK")
except Exception as _e:
    log("  System.Drawing FAILED: %s" % str(_e))

# sqlite3 may not exist in IronPython — try .NET fallback
_USE_DOTNET_SQLITE = False
try:
    import sqlite3
    log("  sqlite3 OK (CPython module)")
except ImportError:
    log("  sqlite3 not available — trying .NET System.Data.SQLite")
    try:
        import clr
        clr.AddReference("System.Data")
        from System.Data.SQLite import SQLiteConnection
        _USE_DOTNET_SQLITE = True
        log("  System.Data.SQLite OK (.NET fallback)")
    except Exception as _e2:
        log("  System.Data.SQLite ALSO FAILED: %s" % str(_e2))
        log("  WARNING: No SQLite driver available!")

log_section("IMPORTS DONE")

# Update info with import results so far
info = "\n".join(_log)

# ── Constants ───────────────────────────────────────────────────────
GRID_PX = 128
PIXEL_SIZE = 0.1          # metres per pixel at 128×128
GRID_M = GRID_PX * PIXEL_SIZE   # 12.8 m
COLUMN_GAP = 2.0          # gap between input / output columns
ROW_GAP = 3.0             # gap between window rows
COLORBAR_STEPS = 50
COLORBAR_WIDTH_PX = 3


# ────────────────────────────────────────────────────────────────────
#  NPZ / NPY parsing  (no numpy)
# ────────────────────────────────────────────────────────────────────
def _parse_npy(data):
    """Parse raw .npy bytes.  Returns (flat_values, shape, dtype_str)."""
    # Magic check — use ord() for Python 2/3 compat
    first_byte = data[0] if isinstance(data[0], int) else ord(data[0])
    if first_byte != 0x93 or data[1:6] not in (b'NUMPY', 'NUMPY'):
        raise ValueError("Bad .npy magic: first_byte=0x%02X" % first_byte)

    major = data[6] if isinstance(data[6], int) else ord(data[6])
    if major == 1:
        header_len = struct.unpack('<H', data[8:10])[0]
        hdr_start = 10
    elif major == 2:
        header_len = struct.unpack('<I', data[8:12])[0]
        hdr_start = 12
    else:
        raise ValueError("Unsupported npy v%d" % major)

    header_raw = data[hdr_start:hdr_start + header_len]
    header = header_raw.decode('ascii') if isinstance(header_raw, bytes) else str(header_raw)
    raw = data[hdr_start + header_len:]

    # Parse shape and descr from the header dict string
    sm = re.search(r"'shape':\s*\(([^)]*)\)", header)
    dm = re.search(r"'descr':\s*'([^']*)'", header)
    if not sm or not dm:
        raise ValueError("Cannot parse npy header: %s" % header)

    shape_s = sm.group(1).strip()
    shape = tuple(int(x.strip()) for x in shape_s.split(',') if x.strip()) if shape_s else ()
    descr = dm.group(1)

    total = 1
    for s in shape:
        total *= s

    if descr in ('|u1', '<u1', 'u1'):
        vals = struct.unpack('%dB' % total, raw[:total])
    elif descr in ('<f4', 'f4'):
        vals = struct.unpack('<%df' % total, raw[:total * 4])
    elif descr in ('<f8', 'f8'):
        vals = struct.unpack('<%dd' % total, raw[:total * 8])
    else:
        raise ValueError("Unsupported npy dtype: %s" % descr)

    return vals, shape, descr


def parse_npz(path):
    """Parse NPZ file.  Returns {key: (flat_values, shape, dtype)}."""
    result = {}
    zf = zipfile.ZipFile(path, 'r')
    try:
        for name in zf.namelist():
            if name.endswith('.npy'):
                key = name[:-4]
                npy_data = zf.read(name)
                vals, shape, dtype = _parse_npy(npy_data)
                result[key] = (vals, shape, dtype)
                log("    npz['%s']  shape=%s  dtype=%s  len=%d" % (
                    key, shape, dtype, len(vals)))
    finally:
        zf.close()
    return result


# ────────────────────────────────────────────────────────────────────
#  HTTP helpers
# ────────────────────────────────────────────────────────────────────
def _json_bytes(payload):
    """Serialize dict to UTF-8 .NET byte array."""
    return Encoding.UTF8.GetBytes(json.dumps(payload))


def http_post_json(url, payload):
    """POST JSON, return parsed JSON dict."""
    log("    POST %s" % url)
    wc = WebClient()
    wc.Headers.Add("Content-Type", "application/json")
    resp = wc.UploadData(url, "POST", _json_bytes(payload))
    text = Encoding.UTF8.GetString(resp)
    log("    response: %d chars" % len(text))
    return json.loads(text)


def http_post_save(url, payload, path):
    """POST JSON, save binary response to *path*."""
    log("    POST %s" % url)
    wc = WebClient()
    wc.Headers.Add("Content-Type", "application/json")
    resp = wc.UploadData(url, "POST", _json_bytes(payload))
    System.IO.File.WriteAllBytes(path, resp)
    log("    saved %d bytes -> %s" % (resp.Length, path))


def http_get_json(url):
    """GET, return parsed JSON dict."""
    log("    GET %s" % url)
    wc = WebClient()
    text = wc.DownloadString(url)
    log("    response: %d chars" % len(text))
    return json.loads(text)


# ────────────────────────────────────────────────────────────────────
#  Database
# ────────────────────────────────────────────────────────────────────
def read_meta_json(db_path, dp_id):
    """Read meta_json for *dp_id* from the SQLite database."""
    log("  db_path = %s" % db_path)
    log("  dp_id   = %s (type=%s)" % (dp_id, type(dp_id).__name__))
    log("  file exists? %s" % os.path.exists(str(db_path)))
    log("  _USE_DOTNET_SQLITE = %s" % _USE_DOTNET_SQLITE)

    if _USE_DOTNET_SQLITE:
        # .NET fallback for IronPython without sqlite3
        log("  Using .NET SQLiteConnection...")
        conn_str = "Data Source=%s;Version=3;Read Only=True;" % str(db_path)
        conn = SQLiteConnection(conn_str)
        conn.Open()
        cmd = conn.CreateCommand()
        cmd.CommandText = "SELECT meta_json FROM datapoints WHERE dp = @dp"
        cmd.Parameters.AddWithValue("@dp", int(dp_id))
        reader = cmd.ExecuteReader()
        if not reader.Read():
            conn.Close()
            raise ValueError("dp=%d not found in database" % dp_id)
        raw = reader.GetString(0)
        conn.Close()
    else:
        log("  Using sqlite3.connect...")
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT meta_json FROM datapoints WHERE dp = ?", (int(dp_id),))
        row = cur.fetchone()
        conn.close()
        if not row:
            raise ValueError("dp=%d not found in database" % dp_id)
        raw = row[0]

    log("  meta_json loaded (%d chars)" % len(raw))
    return json.loads(raw)


# ────────────────────────────────────────────────────────────────────
#  Data extraction helpers
# ────────────────────────────────────────────────────────────────────
def _unwrap_nested(key, value):
    """Unwrap legacy double-nested DB format (e.g. context_buildings)."""
    if isinstance(value, dict) and key in value and len(value) == 1:
        return value[key]
    return value


def extract_mesh_vertices(meta_json):
    """Return combined [x,y,z] vertex list for the obstruction server."""
    geometry = meta_json.get("geometry", {})
    context = _unwrap_nested("context_buildings",
                             geometry.get("context_buildings", []))
    facade = geometry.get("facade", [])
    balcony = geometry.get("balcony_above", [])

    combined = []
    for src, name in [(context, "context_buildings"),
                      (facade, "facade"),
                      (balcony, "balcony_above")]:
        n = len(src) if src else 0
        if src:
            combined.extend(src)
        log("    %s: %d verts (%d tris)" % (name, n, n // 3))

    log("    combined: %d verts (%d tris)" % (
        len(combined), len(combined) // 3))
    return combined


# ────────────────────────────────────────────────────────────────────
#  Colormaps
# ────────────────────────────────────────────────────────────────────
def _clamp01(t):
    return max(0.0, min(1.0, t))


def _lerp(a, b, f):
    """Linear interpolation between a and b by factor f."""
    return int(round(a + (b - a) * f))


def _multi_stop(t, stops):
    """Interpolate through a list of (r, g, b) colour stops.

    *stops* must have at least 2 entries, evenly spaced over [0, 1].
    """
    t = _clamp01(t)
    n = len(stops) - 1
    idx = t * n
    lo = int(idx)
    if lo >= n:
        lo = n - 1
    hi = lo + 1
    f = idx - lo
    r = _lerp(stops[lo][0], stops[hi][0], f)
    g = _lerp(stops[lo][1], stops[hi][1], f)
    b = _lerp(stops[lo][2], stops[hi][2], f)
    return sd.Color.FromArgb(255, r, g, b)


# DF: grayscale  (0,0,0) → (255,255,255) in 11 even steps
_DF_STOPS = [
    (0, 0, 0), (26, 26, 26), (51, 51, 51), (76, 76, 76),
    (102, 102, 102), (128, 128, 128), (153, 153, 153),
    (178, 178, 178), (204, 204, 204), (230, 230, 230), (255, 255, 255),
]

# DA: black → purple → red → yellow → white  (5 stops)
_DA_STOPS = [
    (0, 0, 0), (110, 0, 153), (255, 0, 0),
    (255, 255, 102), (255, 255, 255),
]


def _colormap_df(t):
    """DF grayscale colormap."""
    return _multi_stop(t, _DF_STOPS)


def _colormap_da(t):
    """DA black-purple-red-yellow-white colormap."""
    return _multi_stop(t, _DA_STOPS)


def _val_to_color(val, vmin, vmax, cmap_fn):
    if vmax <= vmin:
        return sd.Color.FromArgb(255, 128, 128, 128)
    t = (val - vmin) / (vmax - vmin)
    return cmap_fn(t)


# ────────────────────────────────────────────────────────────────────
#  Mesh builders
# ────────────────────────────────────────────────────────────────────
def _add_quad(mesh, x0, y0, x1, y1, color):
    """Append a coloured quad to *mesh*."""
    vi = mesh.Vertices.Count
    mesh.Vertices.Add(x0, y0, 0)
    mesh.Vertices.Add(x1, y0, 0)
    mesh.Vertices.Add(x1, y1, 0)
    mesh.Vertices.Add(x0, y1, 0)
    mesh.Faces.AddFace(vi, vi + 1, vi + 2, vi + 3)
    mesh.VertexColors.Add(color)
    mesh.VertexColors.Add(color)
    mesh.VertexColors.Add(color)
    mesh.VertexColors.Add(color)


def build_rgba_mesh(rgba, ox, oy):
    """Encoder RGBA image -> Rhino Mesh.  *rgba* is flat uint8 list."""
    mesh = rg.Mesh()
    for row in range(GRID_PX):
        for col in range(GRID_PX):
            base = (row * GRID_PX + col) * 4
            r, g, b, a = rgba[base], rgba[base + 1], rgba[base + 2], rgba[base + 3]
            if a == 0:
                continue
            x0 = ox + col * PIXEL_SIZE
            y0 = oy + (GRID_PX - 1 - row) * PIXEL_SIZE
            c = sd.Color.FromArgb(a, r, g, b)
            _add_quad(mesh, x0, y0, x0 + PIXEL_SIZE, y0 + PIXEL_SIZE, c)
    return mesh


def build_gt_mesh(values, ox, oy, vmin, vmax, cmap_fn):
    """GT float array -> coloured Rhino Mesh.  NaN pixels skipped."""
    mesh = rg.Mesh()
    for row in range(GRID_PX):
        for col in range(GRID_PX):
            val = values[row * GRID_PX + col]
            if val != val:  # NaN
                continue
            x0 = ox + col * PIXEL_SIZE
            y0 = oy + (GRID_PX - 1 - row) * PIXEL_SIZE
            c = _val_to_color(val, vmin, vmax, cmap_fn)
            _add_quad(mesh, x0, y0, x0 + PIXEL_SIZE, y0 + PIXEL_SIZE, c)
    return mesh


def build_colorbar(ox, oy, cmap_fn):
    """Vertical colorbar strip."""
    mesh = rg.Mesh()
    step_h = GRID_M / COLORBAR_STEPS
    bw = COLORBAR_WIDTH_PX * PIXEL_SIZE
    for i in range(COLORBAR_STEPS):
        t = i / float(COLORBAR_STEPS - 1)
        y = oy + i * step_h
        c = cmap_fn(t)
        _add_quad(mesh, ox, y, ox + bw, y + step_h, c)
    return mesh


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════
log_section("CHECKING INPUTS")
log("  run              = %s (type=%s)" % (run, type(run).__name__))
try:
    log("  db_path          = %s" % db_path)
except:
    log("  db_path          = <not connected>")
try:
    log("  dp_id            = %s" % dp_id)
except:
    log("  dp_id            = <not connected>")
try:
    log("  metric           = %s" % metric)
except:
    log("  metric           = <not connected>")
try:
    log("  variant          = %s" % variant)
except:
    log("  variant          = <not connected>")
try:
    log("  encoder_url      = %s" % encoder_url)
except:
    log("  encoder_url      = <not connected>")
try:
    log("  obstruction_url  = %s" % obstruction_url)
except:
    log("  obstruction_url  = <not connected>")
try:
    log("  gt_url           = %s" % gt_url)
except:
    log("  gt_url           = <not connected>")

# Always update info so at minimum the import + input check logs show
info = "\n".join(_log)

if run:
    try:
        # ── Configuration ───────────────────────────────────────────
        log_section("CONFIGURATION")
        log("dp_id:           %d" % int(dp_id))
        log("db_path:         %s" % db_path)
        log("metric:          %s" % metric)
        log("variant:         %s" % variant)
        log("encoder_url:     %s" % encoder_url)
        log("obstruction_url: %s" % obstruction_url)
        log("gt_url:          %s" % gt_url)

        model_type = "%s_%s" % (metric.lower(), variant.lower())
        uses_orientation = (metric.upper() == "DA")

        # Hardcoded color ranges and colormaps per metric
        if metric.upper() == "DF":
            color_min = 0.0
            color_max = 10.0
            cmap_fn = _colormap_df
        else:
            color_min = 0.0
            color_max = 100.0
            cmap_fn = _colormap_da

        log("model_type:      %s" % model_type)
        log("uses_orientation: %s" % uses_orientation)
        log("color_range:     [%.1f - %.1f]" % (color_min, color_max))
        log("colormap:        %s" % ("grayscale" if metric.upper() == "DF" else "black-purple-red-yellow-white"))

        # ── 1. Read database ────────────────────────────────────────
        log_section("STEP 1: READ DATABASE")
        meta = read_meta_json(db_path, int(dp_id))

        room = meta.get("room", {})
        room_polygon = room.get("room_polygon", [])
        floor_h = float(room.get("floor_height_above_terrain", 0))
        roof_h = float(room.get("height_roof_over_floor", 2.7))
        log("  room_polygon:  %d verts" % len(room_polygon))
        log("  floor_height:  %.3f m" % floor_h)
        log("  roof_height:   %.3f m" % roof_h)

        # Reflectances
        refl_section = meta.get("reflectance", {})
        refl_src = refl_section.get(variant.lower(),
                                    refl_section.get("default", {}))
        REFL_MAP = {
            "wall_reflectance":         ("walls",        0.7),
            "floor_reflectance":        ("floor",        0.3),
            "ceiling_reflectance":      ("ceiling",      0.8),
            "facade_reflectance":       ("facade",       0.3),
            "terrain_reflectance":      ("terrain",      0.2),
            "window_frame_reflectance": ("window_frame", 0.5),
        }
        refls = {}
        for pname, (mkey, dval) in REFL_MAP.items():
            refls[pname] = float(refl_src.get(mkey, dval))
            log("  %s: %.3f" % (pname, refls[pname]))

        # Windows
        win_section = meta.get("window", {})
        win_ids = sorted([k for k in win_section if isinstance(win_section[k], dict)])
        log("  windows (%d): %s" % (len(win_ids), win_ids))
        for wid in win_ids:
            wd = win_section[wid]
            log("    %s  (%.2f,%.2f,%.2f)-(%.2f,%.2f,%.2f)  dir=%.4f" % (
                wid,
                float(wd.get("x1", 0)), float(wd.get("y1", 0)),
                float(wd.get("z1", 0)),
                float(wd.get("x2", 0)), float(wd.get("y2", 0)),
                float(wd.get("z2", 0)),
                float(wd.get("direction_angle", 0))))

        # ── 2. Extract obstruction geometry ─────────────────────────
        log_section("STEP 2: EXTRACT OBSTRUCTION GEOMETRY")
        mesh_verts = extract_mesh_vertices(meta)

        # ── 3. Call obstruction server ──────────────────────────────
        log_section("STEP 3: CALL OBSTRUCTION SERVER")
        obs_results = {}

        if not mesh_verts:
            log("  No obstruction geometry — skipping server call")
            for wid in win_ids:
                obs_results[wid] = {"horizon": [0.0] * 64,
                                    "zenith": [0.0] * 64}
        else:
            for wid in win_ids:
                wd = win_section[wid]
                payload = {
                    "x1": float(wd["x1"]),
                    "y1": float(wd["y1"]),
                    "z1": float(wd["z1"]),
                    "x2": float(wd["x2"]),
                    "y2": float(wd["y2"]),
                    "z2": float(wd["z2"]),
                    "direction_angle": float(wd.get("direction_angle", 0.0)),
                    "room_polygon": room_polygon,
                    "mesh": mesh_verts,
                    "num_directions": 64,
                    "start_angle_degrees": 17.5,
                    "end_angle_degrees": 162.5,
                }
                log("  window '%s' (%d mesh verts)" % (wid, len(mesh_verts)))
                obs_url = "%s/obstruction_all" % obstruction_url.rstrip("/")

                try:
                    resp = http_post_json(obs_url, payload)
                except WebException as ex:
                    log("  HTTP ERROR: %s" % ex.Message)
                    obs_results[wid] = {"horizon": [0.0] * 64,
                                        "zenith": [0.0] * 64}
                    continue
                except Exception as ex:
                    log("  ERROR: %s" % ex)
                    obs_results[wid] = {"horizon": [0.0] * 64,
                                        "zenith": [0.0] * 64}
                    continue

                if resp.get("status") != "success":
                    log("    server error: %s" % resp.get("error", "unknown"))
                    obs_results[wid] = {"horizon": [0.0] * 64,
                                        "zenith": [0.0] * 64}
                    continue

                dirs = resp.get("data", {}).get("results", [])
                h_vals = [r["horizon"]["obstruction_angle_degrees"] for r in dirs]
                z_vals = [r["zenith"]["obstruction_angle_degrees"] for r in dirs]
                log("    %d dirs   horizon=[%.1f .. %.1f]   zenith=[%.1f .. %.1f]" % (
                    len(dirs), min(h_vals), max(h_vals), min(z_vals), max(z_vals)))
                obs_results[wid] = {"horizon": h_vals, "zenith": z_vals}

        # Context / balcony reflectances
        def_refl = refl_section.get("default", {})
        cus_refl = refl_section.get("custom", {})
        if variant.lower() == "custom":
            ctx_r = float(_unwrap_nested(
                "context_buildings",
                cus_refl.get("context_buildings", 0.6)))
            bal_r = float(cus_refl.get("balcony_ceiling", 0.8))
        else:
            ctx_r = float(_unwrap_nested(
                "context_buildings",
                def_refl.get("context_buildings", 0.6)))
            bal_r = float(def_refl.get("balcony_ceiling", 0.8))
        log("  context_reflectance: %.3f" % ctx_r)
        log("  balcony_reflectance: %.3f" % bal_r)

        # ── 4. Call encoder server ──────────────────────────────────
        log_section("STEP 4: CALL ENCODER SERVER")
        enc_params = {
            "room_polygon": room_polygon,
            "floor_height_above_terrain": floor_h,
            "height_roof_over_floor": roof_h,
        }
        enc_params.update(refls)

        enc_windows = {}
        for wid in win_ids:
            wd = win_section[wid]
            da = float(wd.get("direction_angle", 0.0))
            wp = {
                "x1": float(wd["x1"]),
                "y1": float(wd["y1"]),
                "z1": float(wd["z1"]),
                "x2": float(wd["x2"]),
                "y2": float(wd["y2"]),
                "z2": float(wd["z2"]),
                "window_frame_ratio": float(wd.get("frame_ratio", 0.15)),
                "direction_angle": da,
                "wall_thickness": float(wd.get("wall_thickness", 0.3)),
                "window_frame_reflectance": refls["window_frame_reflectance"],
                "context_reflectance": ctx_r,
                "balcony_reflectance": bal_r,
            }
            if uses_orientation:
                wp["window_direction_angle"] = da

            od = obs_results.get(wid, {})
            wp["horizon"] = od.get("horizon", 0.0)
            wp["zenith"] = od.get("zenith", 0.0)

            enc_windows[wid] = wp
            log("  window '%s' ready  (horizon len=%s, zenith len=%s)" % (
                wid,
                len(wp["horizon"]) if isinstance(wp["horizon"], list) else "scalar",
                len(wp["zenith"]) if isinstance(wp["zenith"], list) else "scalar"))

        enc_params["windows"] = enc_windows

        enc_payload = {
            "model_type": model_type,
            "encoding_scheme": "hsv",
            "parameters": enc_params,
        }
        log("  payload model_type=%s  windows=%s" % (model_type, win_ids))

        tmp_dir = tempfile.gettempdir()
        enc_npz = os.path.join(tmp_dir, "gh_enc_dp%d.npz" % int(dp_id))
        enc_url = "%s/encode" % encoder_url.rstrip("/")

        try:
            http_post_save(enc_url, enc_payload, enc_npz)
        except WebException as ex:
            log("  ENCODER HTTP ERROR: %s" % ex.Message)
            # Try to read error body
            if ex.Response:
                sr = System.IO.StreamReader(ex.Response.GetResponseStream())
                log("  response body: %s" % sr.ReadToEnd())
                sr.Close()
            raise
        except Exception as ex:
            log("  ENCODER ERROR: %s" % ex)
            raise

        log("  Parsing encoder NPZ...")
        enc_arrays = parse_npz(enc_npz)

        # ── 5. Call GT server ───────────────────────────────────────
        log_section("STEP 5: CALL GT SERVER")

        # 5a. Get info to discover available results
        gt_info_url = "%s/info/%d" % (gt_url.rstrip("/"), int(dp_id))
        try:
            gt_info = http_get_json(gt_info_url)
        except WebException as ex:
            log("  GT INFO HTTP ERROR: %s" % ex.Message)
            gt_info = {"results": [], "windows": []}
        except Exception as ex:
            log("  GT INFO ERROR: %s" % ex)
            gt_info = {"results": [], "windows": []}

        avail_results = gt_info.get("results", [])
        avail_windows = gt_info.get("windows", [])
        log("  available results: %s" % avail_results)
        log("  available windows: %s" % avail_windows)

        # 5b. Find matching result names
        if metric.upper() == "DF":
            matching = [r for r in avail_results if r == model_type]
        else:
            # DA results are named da_{threshold}_{location}_{variant}
            suffix = "_%s" % variant.lower()
            matching = [r for r in avail_results
                        if r.startswith("da_") and r.endswith(suffix)]

        log("  matching results for '%s': %s" % (model_type, matching))

        # 5c. Generate GT
        gt_payload = {"dp_id": int(dp_id)}
        if matching:
            gt_payload["result_filter"] = matching

        gt_npz = os.path.join(tmp_dir, "gh_gt_dp%d.npz" % int(dp_id))
        gt_gen_url = "%s/generate" % gt_url.rstrip("/")
        log("  generate payload: %s" % gt_payload)

        gt_arrays = {}
        try:
            http_post_save(gt_gen_url, gt_payload, gt_npz)
            log("  Parsing GT NPZ...")
            gt_arrays = parse_npz(gt_npz)
        except WebException as ex:
            log("  GT GENERATE HTTP ERROR: %s" % ex.Message)
            if ex.Response:
                sr = System.IO.StreamReader(ex.Response.GetResponseStream())
                log("  response body: %s" % sr.ReadToEnd())
                sr.Close()
        except Exception as ex:
            log("  GT GENERATE ERROR: %s" % ex)
            log("  %s" % traceback.format_exc())

        # ── 6. Build Rhino meshes ──────────────────────────────────
        log_section("STEP 6: BUILD RHINO MESHES")
        all_meshes = []
        all_labels = []

        for wi, wid in enumerate(win_ids):
            oy = wi * (GRID_M + ROW_GAP)

            # ─ Encoder image ─
            if len(win_ids) == 1:
                enc_key = "image"
            else:
                enc_key = "%s_image" % wid

            if enc_key in enc_arrays:
                rgba, shape, dtype = enc_arrays[enc_key]
                log("  [%s] encoder key='%s'  shape=%s" % (wid, enc_key, shape))
                m = build_rgba_mesh(rgba, 0, oy)
                all_meshes.append(m)
                log("    -> %d faces" % m.Faces.Count)

                dot = rg.TextDot("INPUT  %s\n%s" % (wid, model_type),
                                 rg.Point3d(0, oy + GRID_M + 0.3, 0))
                all_labels.append(dot)
            else:
                log("  WARNING: enc key '%s' not found.  keys=%s" % (
                    enc_key, list(enc_arrays.keys())))

            # ─ GT result(s) ─
            col = 1
            for rname in matching:
                gt_key = "%s__%s" % (wid, rname)
                if gt_key in gt_arrays:
                    gvals, gshape, gdtype = gt_arrays[gt_key]
                    ox = col * (GRID_M + COLUMN_GAP)
                    log("  [%s] GT key='%s'  shape=%s  ox=%.1f" % (
                        wid, gt_key, gshape, ox))

                    gm = build_gt_mesh(gvals, ox, oy, color_min, color_max, cmap_fn)
                    all_meshes.append(gm)

                    # Value stats
                    valid = [v for v in gvals if v == v]
                    if valid:
                        log("    values: min=%.2f  max=%.2f  mean=%.2f  (%d px)" % (
                            min(valid), max(valid),
                            sum(valid) / len(valid), len(valid)))
                    log("    -> %d faces" % gm.Faces.Count)

                    # Colorbar
                    cb_ox = ox + GRID_M + PIXEL_SIZE
                    cb = build_colorbar(cb_ox, oy, cmap_fn)
                    all_meshes.append(cb)

                    # Labels
                    dot = rg.TextDot(
                        "GT  %s\n%s\n[%.1f - %.1f]" % (
                            wid, rname, color_min, color_max),
                        rg.Point3d(ox, oy + GRID_M + 0.3, 0))
                    all_labels.append(dot)

                    # Min / max value labels on colorbar
                    dot_min = rg.TextDot(
                        "%.1f" % color_min,
                        rg.Point3d(cb_ox + COLORBAR_WIDTH_PX * PIXEL_SIZE + 0.1,
                                   oy, 0))
                    dot_max = rg.TextDot(
                        "%.1f" % color_max,
                        rg.Point3d(cb_ox + COLORBAR_WIDTH_PX * PIXEL_SIZE + 0.1,
                                   oy + GRID_M, 0))
                    all_labels.append(dot_min)
                    all_labels.append(dot_max)

                    col += 1
                else:
                    log("  WARNING: GT key '%s' not found.  gt_keys=%s" % (
                        gt_key,
                        [k for k in gt_arrays if k.startswith(wid)]))

            # If no matching GT results, log it
            if not matching:
                log("  [%s] no matching GT results to render" % wid)

        # ── Done ────────────────────────────────────────────────────
        meshes = all_meshes
        labels = all_labels

        log_section("DONE")
        log("Total meshes:  %d" % len(all_meshes))
        log("Total labels:  %d" % len(all_labels))
        info = "\n".join(_log)

    except Exception as e:
        log_section("FATAL ERROR")
        log(str(e))
        log(traceback.format_exc())
        meshes = []
        labels = []
        info = "\n".join(_log)

else:
    meshes = []
    labels = []
    info = "Set run=True to execute"
