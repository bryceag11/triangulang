"""
Cache GT 3D centroids for all objects in all scenes.

This pre-computes the ground truth 3D centroid (mesh vertex median) for each
object in each scene, eliminating the need to load trimesh during training.

Usage:
    python scripts/cache_gt_centroids.py --max-scenes 100

Output:
    data/scannetpp/centroid_cache.json
    Format: {scene_id: {obj_id: [x, y, z], ...}, ...}
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    logger.warning("trimesh not installed. Run: pip install trimesh")

# Constants
SCANNETPP_ROOT = Path(__file__).parent.parent / "data" / "scannetpp"

# Load vertex-to-object ID mapping
def load_vertex_object_ids(scene_path: Path):
    # Try segments file
    segments_path = scene_path / "scans" / "segments.json"
    if segments_path.exists():
        with open(segments_path) as f:
            segments_data = json.load(f)
        return np.array(segments_data.get('segIndices', []), dtype=np.int32), 'segments'

    # Try mesh_aligned_0.05_semantic.ply vertex colors as fallback
    semantic_mesh_path = scene_path / "scans" / "mesh_aligned_0.05_semantic.ply"
    if semantic_mesh_path.exists():
        mesh = trimesh.load(str(semantic_mesh_path), process=False)
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
            # Decode object IDs from vertex colors (R + G*256 + B*65536)
            colors = mesh.visual.vertex_colors[:, :3].astype(np.int32)
            obj_ids = colors[:, 0] + colors[:, 1] * 256 + colors[:, 2] * 65536
            return obj_ids, 'semantic_mesh'

    return None, None

# Get list of object IDs from annotations
def get_object_ids_from_annotations(scene_path: Path):
    anno_path = scene_path / "scans" / "segments_anno.json"
    if not anno_path.exists():
        return {}

    with open(anno_path) as f:
        anno_data = json.load(f)

    obj_id_to_label = {}
    for group in anno_data.get('segGroups', []):
        obj_id = group.get('objectId')
        label = group.get('label', 'unknown')
        if obj_id is not None:
            obj_id_to_label[obj_id] = label

    return obj_id_to_label

# Compute 3D centroid (median) for an object
def compute_centroid_for_object(vertices: np.ndarray, vertex_obj_ids: np.ndarray, obj_id: int):
    mask = vertex_obj_ids == obj_id
    if not np.any(mask):
        return None

    obj_vertices = vertices[mask]
    # Use median for robustness (matches triangulation approach)
    centroid = np.median(obj_vertices, axis=0)
    return centroid.tolist()

# Process single scene, return dict of obj_id -> centroid
def process_scene(scene_id: str, data_root: Path):
    scene_path = data_root / "data" / scene_id

    # Load mesh
    mesh_path = scene_path / "scans" / "mesh_aligned_0.05.ply"
    if not mesh_path.exists():
        return None, f"Mesh not found: {mesh_path}"

    try:
        mesh = trimesh.load(str(mesh_path), process=False)
        vertices = np.array(mesh.vertices, dtype=np.float32)
    except Exception as e:
        return None, f"Error loading mesh: {e}"

    # Load vertex-to-object mapping
    vertex_obj_ids, source = load_vertex_object_ids(scene_path)
    if vertex_obj_ids is None:
        return None, "No vertex-to-object mapping found"

    if len(vertex_obj_ids) != len(vertices):
        return None, f"Vertex count mismatch: {len(vertices)} vs {len(vertex_obj_ids)}"

    # Get object IDs from annotations
    obj_id_to_label = get_object_ids_from_annotations(scene_path)
    if not obj_id_to_label:
        # Fallback: get unique object IDs from vertex mapping
        unique_ids = np.unique(vertex_obj_ids)
        obj_id_to_label = {int(oid): 'unknown' for oid in unique_ids if oid > 0}

    # Compute centroid for each object
    centroids = {}
    for obj_id in obj_id_to_label.keys():
        centroid = compute_centroid_for_object(vertices, vertex_obj_ids, obj_id)
        if centroid is not None:
            centroids[obj_id] = centroid

    return centroids, None

# Get list of scenes from data directory
def get_all_scenes(data_root: Path, split: str = "train", max_scenes: int = None):
    data_dir = data_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    scenes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])

    if max_scenes:
        scenes = scenes[:max_scenes]

    return scenes

# Wrapper for multiprocessing - unpacks args tuple
def process_scene_wrapper(args):
    scene_id, data_root = args
    return scene_id, process_scene(scene_id, data_root)

def main():
    parser = argparse.ArgumentParser(description="Cache GT 3D centroids")
    parser.add_argument('--data-root', type=str, default=str(SCANNETPP_ROOT),
                        help='Path to ScanNet++ data')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path (default: data_root/centroid_cache.json)')
    parser.add_argument('--split', type=str, default='train',
                        help='Split to process (train/val/both)')
    parser.add_argument('--max-scenes', type=int, default=None,
                        help='Max scenes to process')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of parallel workers (default: cpu_count()). '
                             'Uses multiprocessing for faster caching.')
    args = parser.parse_args()

    if not HAS_TRIMESH:
        print("Error: trimesh required. Install with: pip install trimesh")
        sys.exit(1)

    data_root = Path(args.data_root)
    output_path = Path(args.output) if args.output else data_root / "centroid_cache.json"
    num_workers = args.num_workers if args.num_workers else cpu_count()

    logger.info(f"Data root: {data_root}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Workers: {num_workers}")

    # Get scenes
    if args.split == 'both':
        scenes = get_all_scenes(data_root, 'train', args.max_scenes)
        scenes += get_all_scenes(data_root, 'val', args.max_scenes)
        scenes = list(set(scenes))  # Remove duplicates
    else:
        scenes = get_all_scenes(data_root, args.split, args.max_scenes)

    logger.info(f"Processing {len(scenes)} scenes")

    # Load existing cache if present
    all_centroids = {}
    if output_path.exists():
        with open(output_path) as f:
            all_centroids = json.load(f)
        logger.info(f"Loaded existing cache with {len(all_centroids)} scenes")

    # Filter out already cached scenes
    scenes_to_process = [s for s in scenes if s not in all_centroids]
    already_cached = len(scenes) - len(scenes_to_process)
    if already_cached > 0:
        logger.info(f"Skipping {already_cached} already-cached scenes")

    if not scenes_to_process:
        logger.info("All scenes already cached!")
        return

    success = already_cached
    errors = []

    # Use multiprocessing for parallel processing
    if num_workers > 1 and len(scenes_to_process) > 1:
        logger.info(f"Processing {len(scenes_to_process)} scenes with {num_workers} workers...")
        work_items = [(scene_id, data_root) for scene_id in scenes_to_process]

        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(process_scene_wrapper, work_items),
                total=len(work_items),
                desc="Processing scenes"
            ))

        for scene_id, (centroids, error) in results:
            if centroids is not None:
                all_centroids[scene_id] = centroids
                success += 1
            else:
                errors.append((scene_id, error))
    else:
        # Single-threaded fallback
        for scene_id in tqdm(scenes_to_process, desc="Processing scenes"):
            centroids, error = process_scene(scene_id, data_root)

            if centroids is not None:
                all_centroids[scene_id] = centroids
                success += 1
            else:
                errors.append((scene_id, error))

    # Save cache
    with open(output_path, 'w') as f:
        json.dump(all_centroids, f)

    print(f"\nCaching complete!")
    print(f"  Scenes processed: {success}/{len(scenes)}")
    print(f"  Total objects: {sum(len(v) for v in all_centroids.values())}")
    print(f"  Output: {output_path}")

    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for scene_id, error in errors[:10]:
            print(f"    {scene_id}: {error}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
