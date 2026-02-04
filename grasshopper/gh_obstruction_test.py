"""
GhPython: Obstruction Angle Calculator
Connects to the obstruction server to calculate horizon/zenith angles
for a given datapoint, using the new split mesh format.

Outputs vectors already rotated to each azimuth direction with the
obstruction elevation angle applied:
  - HorizonVecs: unit vectors pointing at horizon obstruction boundary
  - ZenithVecs: unit vectors pointing at zenith obstruction boundary

Inputs (GH):
  run: bool
  db_path: str                  # path to dataset.sqlite
  dp_id: int                    # datapoint ID
  server_url: str               # e.g. "http://localhost:8081"

Outputs (GH):
  HorizonVecs: list[Vector3d]   # 64 unit vectors (azimuth + horizon elevation)
  ZenithVecs: list[Vector3d]    # 64 unit vectors (azimuth + zenith elevation)
  RefPoint: Point3d             # window reference point (vector origin)
  HorizonMesh: Mesh             # geometry used for horizon (context_buildings + facade)
  ZenithMesh: Mesh              # geometry used for zenith (balcony_above)
  info: str
"""

import json
import math
import sqlite3
import traceback

import Rhino.Geometry as rg

try:
    from urllib.request import Request, urlopen
    from urllib.error import URLError, HTTPError
    print("[IMPORT] Using urllib.request (Python 3)")
except ImportError:
    from urllib2 import Request, urlopen, URLError, HTTPError
    print("[IMPORT] Using urllib2 (Python 2 / IronPython)")


# ---------- Helpers ----------

def unwrap_nested(key, value):
    """Unwrap legacy double-nested DB format {key: actual_value} -> actual_value."""
    if isinstance(value, dict) and key in value and len(value) == 1:
        return value[key]
    return value


def build_rhino_mesh(vertices):
    """Build a Rhino Mesh from a flat list of [x,y,z] vertex triplets.

    Every 3 consecutive vertices form one triangle face.
    Returns None if fewer than 3 vertices.
    """
    if not vertices or len(vertices) < 3:
        return None
    m = rg.Mesh()
    for v in vertices:
        m.Vertices.Add(float(v[0]), float(v[1]), float(v[2]))
    for i in range(0, len(vertices) - 2, 3):
        m.Faces.AddFace(i, i + 1, i + 2)
    m.Normals.ComputeNormals()
    m.Compact()
    return m


def coarse_filter_verts(vertices, ref_x, ref_y, ref_z, direction_angle):
    """Replicate server CoarseTriangleFilter: keep triangles above window AND not fully behind.

    Processes vertices in groups of 3 (triangles).
    Returns filtered vertex list (only surviving triangles).
    """
    if not vertices or len(vertices) < 3:
        return []

    # Window normal direction (horizontal component)
    nx = math.cos(direction_angle)
    ny = math.sin(direction_angle)

    filtered = []
    total = 0
    kept = 0
    for i in range(0, len(vertices) - 2, 3):
        v0 = vertices[i]
        v1 = vertices[i + 1]
        v2 = vertices[i + 2]
        total += 1

        # Filter 1: height — at least one vertex above window center Z
        max_z = max(float(v0[2]), float(v1[2]), float(v2[2]))
        if max_z <= ref_z:
            continue

        # Filter 2: not fully behind — at least one vertex has positive dot product
        d0 = (float(v0[0]) - ref_x) * nx + (float(v0[1]) - ref_y) * ny
        d1 = (float(v1[0]) - ref_x) * nx + (float(v1[1]) - ref_y) * ny
        d2 = (float(v2[0]) - ref_x) * nx + (float(v2[1]) - ref_y) * ny
        if d0 <= 0 and d1 <= 0 and d2 <= 0:
            continue

        filtered.extend([v0, v1, v2])
        kept += 1

    print("[COARSE-FILTER] %d/%d triangles survived (height + behind)" % (kept, total))
    return filtered


def post_json(url, payload):
    """POST JSON payload to url, return parsed JSON response."""
    body = json.dumps(payload).encode("utf-8")
    print("[HTTP] POST %s (%d bytes)" % (url, len(body)))
    req = Request(url, data=body)
    req.add_header("Content-Type", "application/json")
    try:
        resp = urlopen(req, timeout=120)
        raw = resp.read().decode("utf-8")
        print("[HTTP] Response: %d bytes" % len(raw))
        return json.loads(raw)
    except HTTPError as e:
        error_body = e.read().decode("utf-8") if hasattr(e, "read") else ""
        print("[HTTP] HTTPError %s: %s" % (e.code, error_body[:500]))
        raise ValueError("HTTP %s: %s" % (e.code, error_body[:500]))
    except URLError as e:
        print("[HTTP] URLError: %s" % str(e.reason))
        raise ValueError("Connection failed: %s (is the server running?)" % str(e.reason))


def obstruction_to_vector(azimuth_rad, obstruction_deg, from_horizon):
    """Convert azimuth + obstruction angle into a unit Vector3d.

    Args:
        azimuth_rad: horizontal direction angle (radians, math convention)
        obstruction_deg: obstruction angle (degrees)
        from_horizon: True  = angle measured UP from horizontal (horizon)
                      False = angle measured DOWN from vertical (zenith)

    Returns:
        Unit Vector3d in the azimuth direction with the correct elevation.
    """
    if from_horizon:
        elevation_rad = math.radians(obstruction_deg)
    else:
        elevation_rad = math.radians(90.0 - obstruction_deg)

    cos_el = math.cos(elevation_rad)
    sin_el = math.sin(elevation_rad)

    x = math.cos(azimuth_rad) * cos_el
    y = math.sin(azimuth_rad) * cos_el
    z = sin_el

    return rg.Vector3d(x, y, z)


# ---------- Outputs (defaults) ----------
HorizonVecs = []
ZenithVecs = []
RefPoint = None
HorizonMesh = None
ZenithMesh = None
info = ""

try:
    print("=" * 50)
    print("[START] gh_obstruction_test")
    print("[INPUT] run=%s, dp_id=%s, server_url=%s" % (run, dp_id, server_url if server_url else "None"))
    print("[INPUT] db_path=%s" % (db_path if db_path else "None"))

    if not run:
        info = "run=False (toggle run to True to execute)"
        print("[SKIP] run=False")
    else:
        if not db_path:
            raise ValueError("db_path is empty")
        if dp_id is None:
            raise ValueError("dp_id is None")
        if not server_url:
            server_url = "http://localhost:8081"
            print("[DEFAULT] server_url set to %s" % server_url)

        # ---- 1. Load meta_json from DB ----
        print("[DB] Connecting to %s ..." % db_path)
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        cur.execute("SELECT meta_json FROM datapoints WHERE dp = ?", (int(dp_id),))
        row = cur.fetchone()
        con.close()
        if not row:
            raise ValueError("dp %s not found in database" % dp_id)
        meta = json.loads(row[0])
        print("[DB] Loaded meta_json for dp=%s (%d chars)" % (dp_id, len(row[0])))
        info += "dp=%s | meta_json loaded (%d chars)\n" % (dp_id, len(row[0]))

        # ---- 2. Extract window data ----
        window_dict = meta.get("window", {})
        print("[WINDOW] meta keys: %s" % str(list(meta.keys())))
        print("[WINDOW] window dict keys: %s" % str(list(window_dict.keys())))
        if not window_dict:
            raise ValueError("No 'window' key in meta_json. Available keys: %s" % str(list(meta.keys())))
        win_name = sorted(window_dict.keys())[0]
        win_data = window_dict[win_name]
        print("[WINDOW] Using window '%s', keys: %s" % (win_name, str(list(win_data.keys()))))

        ref_pt = win_data.get("reference_point", None)
        print("[WINDOW] reference_point raw: %s" % str(ref_pt))
        if not ref_pt or len(ref_pt) < 2:
            raise ValueError("No reference_point in window '%s'. win_data keys: %s" % (win_name, str(list(win_data.keys()))))
        ref_x = float(ref_pt[0])
        ref_y = float(ref_pt[1])
        ref_z = float(ref_pt[2]) if len(ref_pt) > 2 else 0.0
        RefPoint = rg.Point3d(ref_x, ref_y, ref_z)

        direction_angle = float(win_data.get("direction_angle", 0.0))

        info += "window=%s\n" % win_name
        info += "reference_point=(%.4f, %.4f, %.4f)\n" % (ref_x, ref_y, ref_z)
        info += "direction_angle=%.6f rad (%.2f deg)\n" % (direction_angle, math.degrees(direction_angle))
        print("[WINDOW] ref=(%.4f, %.4f, %.4f), dir_angle=%.4f rad" % (ref_x, ref_y, ref_z, direction_angle))

        # ---- 3. Build meshes from geometry ----
        geo = meta.get("geometry", {})
        print("[GEO] geometry keys: %s" % str(list(geo.keys())))

        context_raw = geo.get("context_buildings", [])
        context_verts = unwrap_nested("context_buildings", context_raw) or []
        facade_verts = geo.get("facade", []) or []
        balcony_verts = geo.get("balcony_above", []) or []

        print("[GEO] context_buildings: %d verts (raw type: %s)" % (len(context_verts), type(context_raw).__name__))
        print("[GEO] facade: %d verts" % len(facade_verts))
        print("[GEO] balcony_above: %d verts" % len(balcony_verts))

        horizon_verts = context_verts + facade_verts
        zenith_verts = balcony_verts

        info += "horizon_mesh: %d verts (context=%d + facade=%d)\n" % (
            len(horizon_verts), len(context_verts), len(facade_verts)
        )
        info += "zenith_mesh: %d verts (balcony_above=%d)\n" % (len(zenith_verts), len(balcony_verts))

        # Apply same filtering as server (CoarseTriangleFilter) for visualization
        print("[FILTER] Applying coarse filter to horizon_verts ...")
        horizon_filtered = coarse_filter_verts(horizon_verts, ref_x, ref_y, ref_z, direction_angle)
        print("[FILTER] Applying coarse filter to zenith_verts ...")
        zenith_filtered = coarse_filter_verts(zenith_verts, ref_x, ref_y, ref_z, direction_angle)

        info += "horizon filtered: %d -> %d verts\n" % (len(horizon_verts), len(horizon_filtered))
        info += "zenith filtered: %d -> %d verts\n" % (len(zenith_verts), len(zenith_filtered))

        # Build Rhino meshes from FILTERED vertices (what the server actually uses)
        print("[MESH] Building Rhino meshes from filtered verts ...")
        HorizonMesh = build_rhino_mesh(horizon_filtered)
        ZenithMesh = build_rhino_mesh(zenith_filtered)
        print("[MESH] HorizonMesh: %s" % ("OK (%d faces)" % HorizonMesh.Faces.Count if HorizonMesh else "None (no verts)"))
        print("[MESH] ZenithMesh: %s" % ("OK (%d faces)" % ZenithMesh.Faces.Count if ZenithMesh else "None (no verts)"))

        # ---- 4. POST to obstruction server ----
        endpoint = server_url.rstrip("/") + "/obstruction_parallel"

        payload = {
            "x": ref_x,
            "y": ref_y,
            "z": ref_z,
            "direction_angle": direction_angle,
        }

        if horizon_verts:
            payload["horizon_mesh"] = horizon_verts
            print("[PAYLOAD] horizon_mesh: %d verts" % len(horizon_verts))
        else:
            print("[PAYLOAD] horizon_mesh: EMPTY (no context_buildings or facade)")

        if zenith_verts:
            payload["zenith_mesh"] = zenith_verts
            print("[PAYLOAD] zenith_mesh: %d verts" % len(zenith_verts))
        else:
            print("[PAYLOAD] zenith_mesh: EMPTY (no balcony_above)")

        info += "\nPOST %s\n" % endpoint
        info += "payload keys: %s\n" % ", ".join(sorted(payload.keys()))
        print("[HTTP] Sending request to %s ..." % endpoint)

        response = post_json(endpoint, payload)

        # ---- 5. Parse response -> build rotated vectors ----
        status = response.get("status", "unknown")
        print("[RESPONSE] status=%s, keys=%s" % (status, str(list(response.keys()))))
        info += "response status: %s\n" % status

        if status == "error":
            err = response.get("error", response.get("message", "unknown error"))
            print("[RESPONSE] ERROR: %s" % str(err))
            raise ValueError("Server error: %s" % err)

        data = response.get("data", {})
        results = data.get("results", [])

        print("[RESPONSE] %d direction results" % len(results))
        info += "directions returned: %d\n" % len(results)

        if results:
            print("[RESPONSE] First result keys: %s" % str(list(results[0].keys())))

        HorizonVecs = []
        ZenithVecs = []
        h_angles = []
        z_angles = []

        for i, r in enumerate(results):
            azimuth = float(r.get("direction_angle", 0.0))
            h_deg = float(r.get("horizon", {}).get("obstruction_angle_degrees", 0.0))
            z_deg = float(r.get("zenith", {}).get("obstruction_angle_degrees", 0.0))

            h_angles.append(h_deg)
            z_angles.append(z_deg)

            HorizonVecs.append(obstruction_to_vector(azimuth, h_deg, from_horizon=True))
            ZenithVecs.append(obstruction_to_vector(azimuth, z_deg, from_horizon=False))

            if i < 3 or i == len(results) - 1:
                print("[VEC %d] azimuth=%.2f rad, horizon=%.2f deg, zenith=%.2f deg" % (i, azimuth, h_deg, z_deg))

        # Summary
        h_nonzero = sum(1 for v in h_angles if v > 0.01)
        z_nonzero = sum(1 for v in z_angles if v > 0.01)
        h_max = max(h_angles) if h_angles else 0.0
        z_max = max(z_angles) if z_angles else 0.0

        info += "\n=== RESULTS ===\n"
        info += "horizon: %d/%d non-zero, max=%.2f deg\n" % (h_nonzero, len(h_angles), h_max)
        info += "zenith:  %d/%d non-zero, max=%.2f deg\n" % (z_nonzero, len(z_angles), z_max)
        info += "\nhorizon angles (deg):\n  %s\n" % str([round(v, 2) for v in h_angles])
        info += "\nzenith angles (deg):\n  %s\n" % str([round(v, 2) for v in z_angles])

        print("[DONE] HorizonVecs=%d, ZenithVecs=%d" % (len(HorizonVecs), len(ZenithVecs)))
        print("[DONE] horizon max=%.2f, zenith max=%.2f" % (h_max, z_max))

except Exception as e:
    tb = traceback.format_exc()
    print("[ERROR] %s" % str(e))
    print(tb)
    HorizonVecs = []
    ZenithVecs = []
    RefPoint = None
    HorizonMesh = None
    ZenithMesh = None
    info = "ERROR: %s\n\n%s\n\nTraceback:\n%s" % (str(e), info, tb)
