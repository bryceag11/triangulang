"""ScanNet++ 3D geometry and mesh rasterization utilities.

Provides mesh rasterization, vertex-to-object mapping, and GT mask generation
from 3D scene meshes for ScanNet++ multi-view training.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


def _normalize_label_local(label: str, label_fixes: dict) -> str:
    """Local copy of normalize_label to avoid circular imports."""
    label = label.strip()
    while '  ' in label:
        label = label.replace('  ', ' ')
    label = label.rstrip(']').rstrip('[').strip()
    return label_fixes.get(label, label)


def load_vertex_object_ids(scene_path: Path) -> Tuple[np.ndarray, Dict]:
    """
    Load vertex-to-object-ID mapping from annotations.

    Returns:
        vertex_obj_ids: (N_vertices,) array mapping each vertex to object ID (0 = background)
        objects: Dict mapping obj_id -> {label, segments, obb, ...}
    """
    from triangulang.utils.scannetpp_loader import normalize_label

    # Load segments.json (vertex -> segment)
    segments_file = scene_path / "scans" / "segments.json"
    with open(segments_file) as f:
        segments_data = json.load(f)
    seg_indices = np.array(segments_data['segIndices'], dtype=np.int32)

    # Load segments_anno.json (segment -> object)
    anno_file = scene_path / "scans" / "segments_anno.json"
    with open(anno_file) as f:
        anno_data = json.load(f)

    # Build segment -> object ID mapping
    seg_to_obj = {}
    objects = {}
    for group in anno_data.get('segGroups', []):
        obj_id = group['objectId']  # Use objectId, not id
        label = normalize_label(group.get('label', 'unknown'))
        segments = group.get('segments', [])

        for seg_id in segments:
            seg_to_obj[seg_id] = obj_id

        objects[obj_id] = {
            'label': label,
            'segments': segments,
            'obb': group.get('obb', {})
        }

    # Map vertices to object IDs (using segment indices)
    n_vertices = len(seg_indices)
    vertex_obj_ids = np.zeros(n_vertices, dtype=np.int32)

    for vtx_idx, seg_id in enumerate(seg_indices):
        if seg_id in seg_to_obj:
            vertex_obj_ids[vtx_idx] = seg_to_obj[seg_id]

    return vertex_obj_ids, objects


def get_object_centroid_3d(
    mesh_vertices: np.ndarray,
    vertex_obj_ids: np.ndarray,
    target_obj_id: int
) -> Optional[np.ndarray]:
    """
    Get 3D centroid of an object from mesh vertices.

    Args:
        mesh_vertices: (N_vertices, 3) mesh vertices in metric world coordinates
        vertex_obj_ids: (N_vertices,) object ID per vertex
        target_obj_id: Object ID to get centroid for

    Returns:
        (3,) centroid in world coordinates (meters), or None if object not found
    """
    mask = vertex_obj_ids == target_obj_id
    if mask.sum() == 0:
        return None

    object_vertices = mesh_vertices[mask]  # (K, 3)

    # Use mean - matches what triangulation computes from unprojected mask pixels
    centroid = object_vertices.mean(axis=0)

    return centroid


def get_vtx_prop_on_2d(pix_to_face: np.ndarray, vtx_prop: np.ndarray,
                       mesh_faces: np.ndarray) -> np.ndarray:
    """
    Map vertex property to 2D image using face indices.

    Args:
        pix_to_face: (H, W) face indices from rasterization (-1 = no hit)
        vtx_prop: (N_vertices,) property per vertex (e.g., object IDs)
        mesh_faces: (N_faces, 3) face definitions

    Returns:
        (H, W) property map (using first vertex of each face)
    """
    valid = pix_to_face >= 0
    pix_prop = np.zeros_like(pix_to_face, dtype=vtx_prop.dtype)

    valid_faces = pix_to_face[valid]
    first_vertices = mesh_faces[valid_faces, 0]
    pix_prop[valid] = vtx_prop[first_vertices]

    return pix_prop


def rasterize_mesh_o3d(
    mesh: 'o3d.geometry.TriangleMesh',
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,  # camera-to-world (c2w)
    height: int,
    width: int
) -> Dict[str, np.ndarray]:
    """
    Rasterize mesh using Open3D ray casting.

    Uses OpenGL/nerfstudio camera convention:
    - Camera looks down -Z axis in camera space
    - +X is right, +Y is up

    Returns:
        pix_to_face: (H, W) face indices (-1 for misses)
        zbuf: (H, W) depth values
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d required for mesh rasterization")

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Create pixel grid
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)

    # Ray directions in camera space (OpenGL convention: -Z is forward)
    # For pixel (u, v), ray direction is ((u-cx)/fx, -(v-cy)/fy, -1) normalized
    # The Y is negated because image Y grows down but camera Y grows up
    x = (u - cx) / fx
    y = -(v - cy) / fy  # Negate for OpenGL convention
    z = -np.ones_like(x)  # -Z is forward in OpenGL
    dirs_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    dirs_cam = dirs_cam / np.linalg.norm(dirs_cam, axis=-1, keepdims=True)

    # Transform to world space
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    dirs_world = dirs_cam @ R.T
    dirs_world = dirs_world / np.linalg.norm(dirs_world, axis=-1, keepdims=True)
    origins = np.tile(t, (height * width, 1))

    # Cast rays
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)

    rays = np.concatenate([origins, dirs_world], axis=1).astype(np.float32)
    result = scene.cast_rays(o3d.core.Tensor(rays))

    t_hit = result['t_hit'].numpy().reshape(height, width)
    prim_ids = result['primitive_ids'].numpy().reshape(height, width).astype(np.int64)

    # Mark misses
    misses = np.isinf(t_hit)
    pix_to_face = prim_ids.copy()
    pix_to_face[misses] = -1
    zbuf = t_hit.copy()
    zbuf[misses] = 0

    return {'pix_to_face': pix_to_face, 'zbuf': zbuf}


def get_objects_in_image(
    pix_obj_ids: np.ndarray,
    min_pixel_fraction: float = 0.001
) -> List[Tuple[int, float]]:
    """
    Get objects visible in image, filtered by pixel coverage.

    Returns:
        List of (obj_id, pixel_fraction) sorted by coverage descending
    """
    H, W = pix_obj_ids.shape
    total_pixels = H * W

    unique_ids, counts = np.unique(pix_obj_ids, return_counts=True)

    objects = []
    for obj_id, count in zip(unique_ids, counts):
        if obj_id <= 0:  # Skip background
            continue
        fraction = count / total_pixels
        if fraction >= min_pixel_fraction:
            objects.append((int(obj_id), float(fraction)))

    objects.sort(key=lambda x: x[1], reverse=True)
    return objects


class SceneRasterizer:
    """
    Handles mesh rasterization for a scene with caching.
    Uses Open3D ray casting for mesh-to-image rasterization.
    """

    def __init__(
        self,
        scene_path: Path,
        cache_dir: Optional[Path] = None,
        use_undistorted: bool = True
    ):
        self.scene_path = Path(scene_path)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_undistorted = use_undistorted

        # Load mesh
        mesh_path = self.scene_path / "scans" / "mesh_aligned_0.05.ply"
        if HAS_OPEN3D and mesh_path.exists():
            self.mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            self.mesh.compute_vertex_normals()
            self.mesh_faces = np.asarray(self.mesh.triangles)
            self.mesh_vertices = np.asarray(self.mesh.vertices)  # (N, 3) metric coords
        else:
            self.mesh = None
            self.mesh_faces = None
            self.mesh_vertices = None

        # Load annotations
        try:
            self.vertex_obj_ids, self.objects = load_vertex_object_ids(self.scene_path)
        except Exception as e:
            print(f"Warning: Failed to load annotations: {e}")
            self.vertex_obj_ids = None
            self.objects = {}

        # Load transforms for camera params
        if use_undistorted:
            transforms_path = self.scene_path / "dslr" / "nerfstudio" / "transforms_undistorted.json"
        else:
            transforms_path = self.scene_path / "dslr" / "nerfstudio" / "transforms.json"

        from triangulang.utils.scannetpp_io import load_nerfstudio_transforms
        self.transforms = load_nerfstudio_transforms(transforms_path)

        # Build frame lookup
        self.frame_lookup = {}
        if self.transforms:
            for frame in self.transforms.get('frames', []):
                fname = Path(frame.get('file_path', '')).name
                self.frame_lookup[fname] = frame

    def get_gt_mask(
        self,
        image_name: str,
        target_obj_id: int,
        output_size: Tuple[int, int] = None
    ) -> Optional[np.ndarray]:
        """
        Get GT binary mask for a specific object in an image.

        Args:
            image_name: Image filename
            target_obj_id: Object ID to extract
            output_size: Optional (H, W) to resize mask

        Returns:
            (H, W) binary mask as float32
        """
        if self.mesh is None or self.vertex_obj_ids is None:
            return None

        raster = self._get_rasterization(image_name)
        if raster is None:
            return None

        pix_to_face = raster['pix_to_face']
        pix_obj_ids = get_vtx_prop_on_2d(pix_to_face, self.vertex_obj_ids, self.mesh_faces)

        mask = (pix_obj_ids == target_obj_id).astype(np.float32)

        if output_size is not None and mask.shape != output_size:
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((output_size[1], output_size[0]), Image.NEAREST)
            mask = np.array(mask_pil).astype(np.float32) / 255.0

        return mask

    def get_visible_objects(
        self,
        image_name: str,
        min_coverage: float = 0.001
    ) -> List[Tuple[int, str, float]]:
        """
        Get objects visible in an image with their labels and coverage.

        Returns:
            List of (obj_id, label, coverage_fraction)
        """
        if self.mesh is None or self.vertex_obj_ids is None:
            return []

        raster = self._get_rasterization(image_name)
        if raster is None:
            return []

        pix_obj_ids = get_vtx_prop_on_2d(
            raster['pix_to_face'], self.vertex_obj_ids, self.mesh_faces
        )

        visible = get_objects_in_image(pix_obj_ids, min_coverage)

        from triangulang.utils.scannetpp_loader import normalize_label
        result = []
        for obj_id, coverage in visible:
            label = normalize_label(self.objects.get(obj_id, {}).get('label', 'unknown'))
            result.append((obj_id, label, coverage))

        return result

    def get_object_centroid(self, obj_id: int) -> Optional[np.ndarray]:
        """
        Get 3D centroid of an object in world coordinates (meters).

        Args:
            obj_id: Object ID

        Returns:
            (3,) centroid in world coordinates, or None if not found
        """
        if self.mesh_vertices is None or self.vertex_obj_ids is None:
            return None

        return get_object_centroid_3d(
            self.mesh_vertices,
            self.vertex_obj_ids,
            obj_id
        )

    def _get_rasterization(self, image_name: str) -> Optional[Dict]:
        """Get rasterization for an image, using cache if available."""
        # Check cache - support both toolkit format (dslr/scene_id/) and simple format (scene_id/)
        if self.cache_dir:
            # Try toolkit format first: cache_dir/dslr/{scene_id}/{image}.pth
            cache_paths = [
                self.cache_dir / "dslr" / self.scene_path.name / f"{image_name}.pth",
                self.cache_dir / self.scene_path.name / f"{image_name}.pth"
            ]
            for cache_path in cache_paths:
                if cache_path.exists():
                    data = torch.load(cache_path, weights_only=True)
                    return {
                        'pix_to_face': data['pix_to_face'].numpy() if torch.is_tensor(data['pix_to_face']) else data['pix_to_face'],
                        'zbuf': data['zbuf'].numpy() if torch.is_tensor(data['zbuf']) else data['zbuf']
                    }

        # Compute rasterization
        if image_name not in self.frame_lookup:
            return None

        frame = self.frame_lookup[image_name]
        c2w = np.array(frame['transform_matrix'], dtype=np.float64)

        W = int(self.transforms.get('w', 1752))
        H = int(self.transforms.get('h', 1168))
        fx = self.transforms.get('fl_x', 1000)
        fy = self.transforms.get('fl_y', 1000)
        cx = self.transforms.get('cx', W / 2)
        cy = self.transforms.get('cy', H / 2)

        intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        try:
            raster = rasterize_mesh_o3d(self.mesh, intrinsics, c2w, H, W)
        except Exception as e:
            print(f"Rasterization failed for {image_name}: {e}")
            return None

        # Cache result
        if self.cache_dir:
            cache_path = self.cache_dir / self.scene_path.name / f"{image_name}.pth"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'pix_to_face': torch.from_numpy(raster['pix_to_face']),
                'zbuf': torch.from_numpy(raster['zbuf'].astype(np.float32))
            }, cache_path)

        return raster


def get_scene_3d_annotations(scene_path: Path) -> Optional[Dict]:
    """
    Load 3D semantic mesh annotations for a scene.
    Returns vertex colors and semantic labels.

    Requires trimesh library.
    """
    if not HAS_TRIMESH:
        print("Warning: trimesh not installed. Cannot load 3D annotations.")
        return None

    mesh_path = scene_path / "scans" / "mesh_aligned_0.05_semantic.ply"
    if not mesh_path.exists():
        return None

    mesh = trimesh.load(mesh_path)

    return {
        'vertices': np.array(mesh.vertices),  # (N, 3) in metric coordinates
        'vertex_colors': np.array(mesh.visual.vertex_colors)[:, :3],  # RGB
        'faces': np.array(mesh.faces)
    }


def project_mesh_to_mask(
    vertices: np.ndarray,
    faces: np.ndarray,
    segment_indices: List[int],
    segments: Dict,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    image_size: Tuple[int, int]
) -> np.ndarray:
    """
    Project 3D mesh segments to 2D mask.

    Args:
        vertices: (N, 3) mesh vertices in world coordinates
        faces: (F, 3) face indices
        segment_indices: List of segment IDs for the target object
        segments: Dict mapping segment_id -> face indices
        intrinsics: (3, 3) camera intrinsics
        extrinsics: (4, 4) camera extrinsics (world to camera)
        image_size: (H, W) output mask size

    Returns:
        (H, W) binary mask
    """
    H, W = image_size
    mask = np.zeros((H, W), dtype=np.float32)

    # Get faces belonging to target segments
    target_faces = []
    for seg_id in segment_indices:
        seg_key = str(seg_id)
        if seg_key in segments:
            target_faces.extend(segments[seg_key])

    if not target_faces:
        return mask

    # Get vertices of target faces
    target_face_indices = np.array(target_faces)
    target_verts = vertices[faces[target_face_indices].flatten()]  # (F*3, 3)

    # Transform to camera coordinates
    # extrinsics is c2w, we need w2c
    w2c = np.linalg.inv(extrinsics)
    R = w2c[:3, :3]
    t = w2c[:3, 3]

    verts_cam = (R @ target_verts.T).T + t  # (N, 3)

    # Filter points behind camera
    valid = verts_cam[:, 2] > 0.01
    if not valid.any():
        return mask

    verts_cam = verts_cam[valid]

    # Project to image
    verts_proj = (intrinsics @ verts_cam.T).T  # (N, 3)
    verts_2d = verts_proj[:, :2] / verts_proj[:, 2:3]  # (N, 2)

    # Draw points on mask (simple scatter - could use rasterization for better quality)
    x = np.clip(verts_2d[:, 0].astype(int), 0, W - 1)
    y = np.clip(verts_2d[:, 1].astype(int), 0, H - 1)
    mask[y, x] = 1.0

    # Dilate to fill gaps
    from scipy import ndimage
    mask = ndimage.binary_dilation(mask, iterations=3).astype(np.float32)

    return mask


def load_segments_mapping(scene_path: Path) -> Dict[str, List[int]]:
    """Load segment ID to face indices mapping."""
    segments_file = scene_path / "scans" / "segments.json"
    if not segments_file.exists():
        return {}

    with open(segments_file) as f:
        data = json.load(f)

    # segments.json has {"segIndices": [seg_id_for_each_face]}
    seg_indices = data.get('segIndices', [])

    # Build reverse mapping: segment_id -> list of face indices
    segments = {}
    for face_idx, seg_id in enumerate(seg_indices):
        seg_key = str(seg_id)
        if seg_key not in segments:
            segments[seg_key] = []
        segments[seg_key].append(face_idx)

    return segments
