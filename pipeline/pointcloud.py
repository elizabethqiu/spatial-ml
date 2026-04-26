"""
2D → 3D lifting and Z-axis canonicalization.

Ported from Spatial_Reasoning_with_Point_Clouds.ipynb and aligned with
the SpatialVLM paper (gravity-aligned coordinate frame via flat-surface RANSAC).

Public API used by inference/run.py:
    backproject(depth, intrinsics, rgb=None) -> open3d.PointCloud
    canonicalize_point_cloud(pcd, threshold=0.3) -> (pcd, bool, transform | None)
    default_intrinsics(W, H) -> dict
    gravity_align_rotation(plane_model) -> np.ndarray (3x3)
    find_ground_normal(pcd, threshold=0.3) -> np.ndarray | None
    estimate_normals(pcd) -> open3d.PointCloud
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Camera intrinsics
# ---------------------------------------------------------------------------

def default_intrinsics(W: int, H: int) -> dict:
    """
    Estimate a pinhole camera model assuming 1.5× focal length heuristic
    (matches SpatialVLM paper / notebook default).
    """
    fx = fy = 1.5 * W
    return {"width": W, "height": H, "fx": fx, "fy": fy, "cx": W / 2, "cy": H / 2}


# ---------------------------------------------------------------------------
# Backprojection
# ---------------------------------------------------------------------------

def backproject(
    depth: np.ndarray,
    intrinsics: dict,
    rgb: np.ndarray | None = None,
) -> "open3d.geometry.PointCloud":
    """
    Lift a metric depth map (H×W float32, metres) into a 3-D point cloud.

    Parameters
    ----------
    depth     : H×W float32 array of metric depth values in metres.
    intrinsics: dict with keys fx, fy, cx, cy (pixels).
    rgb       : optional H×W×3 uint8 array; colours the point cloud when given.

    Returns
    -------
    open3d.geometry.PointCloud
    """
    import open3d as o3d

    H, W = depth.shape
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    valid = depth > 0

    Z = depth[valid].astype(np.float64)
    X = (us[valid] - cx) * Z / fx
    Y = (vs[valid] - cy) * Z / fy

    pts = np.stack([X, Y, Z], axis=-1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    if rgb is not None:
        colors = rgb[valid].astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


# ---------------------------------------------------------------------------
# Plane detection / Z-axis canonicalization
# ---------------------------------------------------------------------------

def find_ground_normal(
    pcd: "open3d.geometry.PointCloud",
    threshold: float = 0.3,
) -> np.ndarray | None:
    """
    Segment the dominant plane with RANSAC.  Returns the outward unit normal
    (pointing away from camera / upward) if the plane covers at least
    `threshold` fraction of all points, else None.
    """
    if len(pcd.points) < 10:
        return None

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.01, ransac_n=3, num_iterations=1000
    )
    if len(inliers) / len(pcd.points) < threshold:
        return None

    normal = np.array(plane_model[:3])
    # Ensure it points "upward" (toward positive Y in camera space)
    if np.dot(normal, [0, 1, 0]) < 0:
        normal = -normal
    return normal / np.linalg.norm(normal)


def gravity_align_rotation(plane_model: np.ndarray) -> np.ndarray:
    """
    Build a 3×3 rotation matrix that maps `plane_model`'s normal to +Y
    (gravity-aligned Z becomes world-up after the transform).

    Parameters
    ----------
    plane_model : length-4 array [a, b, c, d] from open3d segment_plane,
                  OR a length-3 unit normal vector.
    """
    normal = np.array(plane_model[:3], dtype=np.float64)
    if np.dot(normal, [0, 1, 0]) < 0:
        normal = -normal
    normal /= np.linalg.norm(normal)

    new_y = normal
    # Pick an arbitrary vector not parallel to new_y
    ref = np.array([0, 0, -1.0]) if abs(new_y[2]) < 0.9 else np.array([1, 0, 0.0])
    new_x = np.cross(new_y, ref)
    new_x /= np.linalg.norm(new_x)
    new_z = np.cross(new_x, new_y)

    return np.vstack([new_x, new_y, new_z]).T  # 3×3


def canonicalize_point_cloud(
    pcd: "open3d.geometry.PointCloud",
    canonicalize_threshold: float = 0.3,
) -> tuple["open3d.geometry.PointCloud", bool, np.ndarray | None]:
    """
    Detect the largest planar surface (assumed to be the ground / flat slab)
    and rotate the point cloud so that Z points up, matching the SpatialVLM
    gravity-aligned coordinate convention.

    Returns
    -------
    (transformed_pcd, was_canonicalized, 4×4_transform_or_None)
    """
    import open3d as o3d

    if len(pcd.points) < 10:
        return pcd, False, None

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.01, ransac_n=3, num_iterations=1000
    )

    if len(inliers) / len(pcd.points) < canonicalize_threshold:
        return pcd, False, None

    # Build 4×4 transform from rotation + translation anchored at first inlier
    R = gravity_align_rotation(plane_model)
    anchor = np.asarray(pcd.points)[inliers[0]]

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ anchor

    pcd.transform(T)

    # Additional 180° rotation around Z so the scene faces forward
    rot_z180 = np.array([
        [np.cos(np.pi), -np.sin(np.pi), 0],
        [np.sin(np.pi),  np.cos(np.pi), 0],
        [0,              0,             1],
    ])
    pcd.rotate(rot_z180, center=(0, 0, 0))

    return pcd, True, T


# ---------------------------------------------------------------------------
# Normal estimation (used by some downstream steps)
# ---------------------------------------------------------------------------

def estimate_normals(
    pcd: "open3d.geometry.PointCloud",
    radius: float = 0.1,
    max_nn: int = 30,
) -> "open3d.geometry.PointCloud":
    """Estimate per-point normals in-place and return the cloud."""
    pcd.estimate_normals(
        search_param=__import__("open3d").geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=max_nn
        )
    )
    return pcd


# ---------------------------------------------------------------------------
# Convenience: build coloured point cloud from RGB + depth images on disk
# ---------------------------------------------------------------------------

def point_cloud_from_images(
    rgb_path: str,
    depth_path: str,
    intrinsics: dict | None = None,
    depth_scale: float = 1000.0,
    depth_trunc: float = 10.0,
) -> "open3d.geometry.PointCloud":
    """
    Load an RGB image + a 16-bit depth PNG and return a coloured point cloud
    using open3d's native RGBD pipeline (mirrors the notebook approach).

    depth_scale: divisor to convert raw depth values to metres (1000 for mm).
    """
    import open3d as o3d

    color_o3d = o3d.io.read_image(rgb_path)
    depth_o3d = o3d.io.read_image(depth_path)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )

    import cv2
    img = cv2.imread(rgb_path)
    H, W = img.shape[:2]
    intr = intrinsics or default_intrinsics(W, H)

    cam = o3d.camera.PinholeCameraIntrinsic()
    cam.set_intrinsics(intr["width"], intr["height"],
                       intr["fx"], intr["fy"],
                       intr["cx"], intr["cy"])

    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam)
