#!/usr/bin/env python3
"""
unified_inspector.py
IS-Code Compliant Hybrid Pipeline
--------------------------------
Runs Open3D pipeline if available, otherwise NumPy + SciPy fallback.
 
Usage:
    python unified_inspector.py cloud.ply
 
Outputs:
    - cleaned_cloud.ply
    - wall_inliers.ply
    - unified_report.json
"""
 
import sys, os, json
import numpy as np
import math
from math import acos, degrees
from datetime import datetime, timezone
 
# Try Open3D
_use_open3d = False
try:
    import open3d as o3d
    _use_open3d = True
except Exception:
    _use_open3d = False
 
# Try SciPy
_try_scipy = True
try:
    from scipy.spatial import cKDTree
except Exception:
    _try_scipy = False
 
 
# ---------------------------------------------------------------------------
# IS-CODE CHECKS — OFFICIAL STRUCTURAL LIMITS (IS 456, IS 1200)
# ---------------------------------------------------------------------------
def is_code_checks(pts, distances,
                   exposure_class='mild',
                   tolerance_plumb_mm_per_3m=5.0,      # IS 1200 conservative
                   deflection_limit_ratio=200):          # L/200 (IS 456 guidance)
    """
    Performs IS-based Plumbness, Deflection & Crack Width compliance checks.
    pts = Nx3 wall-inlier coordinates (meters)
    distances = signed point-to-plane distances (meters)
    """
 
    if len(pts) == 0:
        return {"status": "ERROR - No wall inlier points found"}
 
    # 1) Estimate height (Z-range)
    z_vals = pts[:, 2]
    H_m = float(np.max(z_vals) - np.min(z_vals))
    H_mm = H_m * 1000.0
 
    if H_mm < 100.0:     # If height looks incorrect due to scan angle cropping
        H_mm_actual = H_mm
        H_mm = 3000.0    # Safe fallback using 3 m height
        height_note = f"Measured height only {round(H_mm_actual,2)}mm; using 3000mm fallback."
    else:
        height_note = "Measured from point cloud."
 
    # 2) Lateral deviation (Plumb)
    abs_d = np.abs(distances)
    delta_mm_98 = float(np.percentile(abs_d, 98) * 1000.0)
    delta_mm_max = float(np.max(abs_d) * 1000.0)
 
    # IS 1200 Plumb tolerance (e.g., 5 mm per 3 m)
    T_mm = (H_mm / 3000.0) * tolerance_plumb_mm_per_3m
 
    theta_meas = math.degrees(math.atan(delta_mm_max / H_mm))
    theta_allow = math.degrees(math.atan(T_mm / H_mm))
    plumb_pass = delta_mm_max <= T_mm
 
    # 3) Deflection — IS serviceability L/ratio
    delta_allow_mm = H_mm / deflection_limit_ratio
    deflect_pass = delta_mm_98 <= delta_allow_mm
 
    # 4) Crack width limits (IS 456 table)
    crack_map = {'mild':0.3, 'moderate':0.2, 'severe':0.1}
    crack_allow_mm = crack_map.get(exposure_class, 0.3)
 
    report = {
        "wall_height_mm": round(H_mm, 2),
        "height_estimation_note": height_note,
 
        "plumb": {
            "measured_max_deviation_mm": round(delta_mm_max, 3),
            "measured_98pct_deviation_mm": round(delta_mm_98, 3),
            "allowed_deviation_mm": round(T_mm, 3),
            "measured_angle_deg": round(theta_meas, 4),
            "allowed_angle_deg": round(theta_allow, 4),
            "pass": plumb_pass,
            "note": f"IS 1200 tolerance = {tolerance_plumb_mm_per_3m} mm per 3 m height"
        },
 
        "deflection": {
            "measured_deflection_mm_98pct": round(delta_mm_98, 3),
            "allowed_deflection_mm": round(delta_allow_mm, 3),
            "ratio_used": f"L/{deflection_limit_ratio}",
            "pass": deflect_pass,
            "note": f"IS 456:2000 serviceability guideline"
        },
 
        "crack": {
            "exposure_class": exposure_class,
            "allowed_crack_width_mm": crack_allow_mm,
            "note": "Actual crack width must be measured physically; LiDAR provides proxy curvature only."
        }
    }
 
    # Final decision
    if not plumb_pass or not deflect_pass:
        report["status"] = "VIOLATION - IMMEDIATE ENGINEER REVIEW REQUIRED"
        report["recommendation"] = "Wall exceeds IS limits. Detailed structural inspection required urgently."
    else:
        report["status"] = "SERVICEABLE (Crack verification needed)"
        report["recommendation"] = "Continue monitoring; verify cracks manually."
 
    return report
 
 
# ---------------------------------------------------------------------------
# OPEN3D PIPELINE
# ---------------------------------------------------------------------------
def with_open3d(filepath,
                voxel_size=0.01,
                sor_neighbors=6, sor_std=1.0,
                plane_thresh=0.02,
                plane_iters=2000,
                max_candidates=6):
 
    print(">>> Running Open3D pipeline (IS-Compliant)")
 
    pcd = o3d.io.read_point_cloud(filepath)
    raw_n = len(pcd.points)
 
    # 1) Downsample + Clean
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=sor_neighbors,
                                             std_ratio=sor_std)
    pcd_clean = pcd.select_by_index(ind)
    cleaned_n = len(pcd_clean.points)
    o3d.io.write_point_cloud("cleaned_cloud.ply", pcd_clean)
 
    # 2) Normals
    pcd_clean.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 3,
            max_nn=30
        )
    )
    pcd_clean.normalize_normals()
 
    # 3) Extract multiple plane candidates
    remaining = pcd_clean
    candidates = []
    for _ in range(max_candidates):
        if len(remaining.points) < 50:
            break
        model, inliers = remaining.segment_plane(
            distance_threshold=plane_thresh,
            ransac_n=3,
            num_iterations=plane_iters,
        )
        inlier_cloud = remaining.select_by_index(inliers)
        candidates.append({"model": model,
                           "cloud": inlier_cloud,
                           "count": len(inliers)})
        remaining = remaining.select_by_index(inliers, invert=True)
 
    if not candidates:
        raise RuntimeError("Plane extraction failed — no planar surface found.")
 
    # 4) Choose MOST VERTICAL plane (robust scoring)
    up = np.array([0,0,1.0])
    best_score = -1
    best = None
 
    for idx, cand in enumerate(candidates):
        a, b, c, d = cand["model"]
        normal = np.array([a,b,c])
        angle = degrees(acos(abs(np.dot(normal, up)) /
                             (np.linalg.norm(normal)*np.linalg.norm(up))))
        tilt_dev = abs(90.0 - angle)
 
        weight = 1.0 / (1.0 + tilt_dev)   # robust, never zero
        score = cand["count"] * weight
 
        if score > best_score:
            best_score = score
            best = (idx, cand, tilt_dev)
 
    best_idx, best_cand, tilt_deg = best
    a, b, c, d = best_cand["model"]
    normal = np.array([a,b,c])
    wall_pcd = best_cand["cloud"]
    o3d.io.write_point_cloud("wall_inliers.ply", wall_pcd)
 
    # 5) Compute signed distances
    pts = np.asarray(wall_pcd.points)
    normal_norm = np.linalg.norm(normal)
    signed = (pts @ normal + d) / normal_norm
 
    # 6) Crack proxy
    normals = np.asarray(wall_pcd.normals)
    try:
        tree = cKDTree(pts)
        k = min(20, len(pts)-1)
        neighs = tree.query(pts, k=k+1)[1][:,1:]
        variations = np.linalg.norm(np.std(normals[neighs], axis=1), axis=1)
        frac_minor = float(np.mean(variations > 0.08))
        frac_critical = float(np.mean(variations > 0.18))
    except Exception:
        frac_minor = 0.0
        frac_critical = 0.0
 
    # 7) IS Code Checks
    is_report = is_code_checks(pts, signed)
 
    # Final report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_file": os.path.abspath(filepath),
        "raw_points": raw_n,
        "cleaned_points": cleaned_n,
        "plane_inliers": len(pts),
 
        "plane_model": {"a": float(a), "b": float(b),
                        "c": float(c), "d": float(d)},
        "max_tilt_degrees": float(tilt_deg),
 
        "wall_height_mm": is_report["wall_height_mm"],
        "plumb": is_report["plumb"],
        "deflection": is_report["deflection"],
        "crack": is_report["crack"],
 
        "normal_variation_stats": {
            "frac_var_gt_0.08": frac_minor,
            "frac_var_gt_0.18": frac_critical
        },
 
        "status": is_report["status"],
        "recommendation": is_report["recommendation"]
    }
 
    with open("unified_report.json", "w") as f:
        json.dump(report, f, indent=2)
 
    print("Saved unified_report.json (IS-Compliant)")
    return report
 
 
# ---------------------------------------------------------------------------
# FALLBACK PIPELINE (NUMPY + SCIPY)
# ---------------------------------------------------------------------------
def read_ply_minimal_xyz(filepath):
    header = []
    fmt = "ascii"
    vertex_count = 0
 
    with open(filepath, "rb") as f:
        while True:
            line = f.readline().decode("utf-8").strip()
            header.append(line)
            if line.startswith("format"):
                if "binary_little_endian" in line:
                    fmt = "binary_le"
                elif "binary_big_endian" in line:
                    fmt = "binary_be"
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            if line == "end_header":
                break
 
        if fmt == "ascii":
            arr = np.loadtxt(f)
            return arr[:, :3]
 
        data = f.read()
        arr = np.frombuffer(data, dtype=("<f4" if fmt == "binary_le" else ">f4"))
        stride = arr.size // vertex_count
        return arr.reshape(vertex_count, stride)[:, :3]
 
 
def fallback_pipeline(filepath,
                      voxel_size=0.02,
                      sor_k=6, sor_sigma=1.0,
                      ransac_iters=1000, ransac_thresh=0.03,
                      max_candidates=6):
 
    if not _try_scipy:
        raise RuntimeError("SciPy not installed — fallback cannot run.")
 
    print(">>> Running Fallback Pipeline (IS-Compliant)")
 
    pts = read_ply_minimal_xyz(filepath)
    raw_n = len(pts)
 
    # 1) voxel downsample
    q = np.floor(pts / voxel_size).astype(np.int64)
    _, idx = np.unique(q, axis=0, return_index=True)
    pts_ds = pts[idx]
 
    # 2) SOR cleaning
    tree = cKDTree(pts_ds)
    k = min(sor_k+1, len(pts_ds))
    dists, _ = tree.query(pts_ds, k=k)
    mean_d = np.mean(dists[:,1:], axis=1)
    th = np.mean(mean_d) + sor_sigma * np.std(mean_d)
    pts_clean = pts_ds[mean_d < th]
    cleaned_n = len(pts_clean)
 
    # 3) RANSAC planes
    def ransac_plane(points, iters, thresh):
        best = None
        best_count = 0
        n = len(points)
        if n < 3: 
            return None
 
        for _ in range(iters):
            ids = np.random.choice(n, 3, replace=False)
            p1, p2, p3 = points[ids]
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            ln = np.linalg.norm(normal)
            if ln == 0: 
                continue
 
            normal = normal / ln
            d = -np.dot(normal, p1)
            distances = np.dot(points, normal) + d
            mask = np.abs(distances) < thresh
            count = mask.sum()
 
            if count > best_count:
                best_count = count
                best = (normal, d, mask, distances[mask])
 
        return best
 
    remaining = pts_clean.copy()
    candidates = []
 
    for _ in range(max_candidates):
        if len(remaining) < 50:
            break
 
        out = ransac_plane(remaining, ransac_iters, ransac_thresh)
        if out is None:
            break
 
        normal, d, mask, signed = out
        in_pts = remaining[mask]
        candidates.append((normal, d, in_pts, signed))
        remaining = remaining[~mask]
 
    if not candidates:
        raise RuntimeError("No plane found.")
 
    # 4) Select best (vertical)
    up = np.array([0,0,1.0])
    best_idx = None
    best_score = -1
 
    for i, (normal, d, in_pts, _) in enumerate(candidates):
        angle = degrees(acos(abs(np.dot(normal, up))))
        tilt_dev = abs(90.0 - angle)
        weight = 1.0 / (1.0 + tilt_dev)
        score = len(in_pts) * weight
 
        if score > best_score:
            best_score = score
            best_idx = i
 
    normal, d, pts_wall, signed = candidates[best_idx]
    angle = degrees(acos(abs(np.dot(normal, up))))
    tilt_deg = abs(90.0 - angle)
 
    # Crack proxy
    tree2 = cKDTree(pts_wall)
    k2 = min(20, len(pts_wall)-1)
    if k2 >= 3:
        neigh = tree2.query(pts_wall, k=k2+1)[1][:,1:]
        normals_est = np.zeros((len(pts_wall),3))
        for i,p in enumerate(pts_wall):
            neigh_pts = pts_wall[neigh[i]]
            cov = np.cov(neigh_pts.T)
            w,v = np.linalg.eigh(cov)
            normals_est[i] = v[:,0]
 
        variations = np.linalg.norm(np.std(normals_est[neigh], axis=1), axis=1)
        frac_minor = float(np.mean(variations > 0.08))
        frac_critical = float(np.mean(variations > 0.18))
    else:
        frac_minor = 0.0
        frac_critical = 0.0
 
    # IS checks
    is_report = is_code_checks(pts_wall, signed)
 
    # Save PLYs
    def save_ply(fn, arr):
        with open(fn,"w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(arr)}\n")
            f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
            for x, y, z in arr:
                f.write(f"{x} {y} {z}\n")
 
    save_ply("cleaned_cloud.ply", pts_clean)
    save_ply("wall_inliers.ply", pts_wall)
 
    # Final report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_file": os.path.abspath(filepath),
        "raw_points": raw_n,
        "cleaned_points": cleaned_n,
        "plane_inliers": len(pts_wall),
 
        "max_tilt_degrees": float(tilt_deg),
 
        "wall_height_mm": is_report["wall_height_mm"],
        "plumb": is_report["plumb"],
        "deflection": is_report["deflection"],
        "crack": is_report["crack"],
 
        "normal_variation_stats": {
            "frac_var_gt_0.08": frac_minor,
            "frac_var_gt_0.18": frac_critical
        },
 
        "status": is_report["status"],
        "recommendation": is_report["recommendation"]
    }
 
    with open("unified_report.json","w") as f:
        json.dump(report, f, indent=2)
 
    print("Saved unified_report.json (IS-Compliant Fallback)")
    return report
 
 
# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python unified_inspector.py cloud.ply")
        sys.exit(1)
 
    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print("File not found:", filepath)
        sys.exit(1)
 
    try:
        if _use_open3d:
            report = with_open3d(filepath)
        else:
            if not _try_scipy:
                raise RuntimeError("Neither Open3D nor SciPy available.")
            report = fallback_pipeline(filepath)
 
        print(json.dumps(report, indent=2))
 
    except Exception as e:
        print("Processing failed:", str(e))
        raise
 
 
if __name__ == "__main__":
    main()