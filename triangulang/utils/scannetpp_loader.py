"""ScanNet++ multi-view dataset loader.

Loads RGB images, depth, masks, and camera parameters for training and evaluation.
Includes label normalization (LABEL_FIXES) and cached sample discovery.
"""
import json
import pickle
import hashlib
import fcntl
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, OrderedDict, defaultdict
import random
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


def _is_main_process():
    """Check if this is the main process (rank 0) for DDP-safe printing."""
    return not dist.is_initialized() or dist.get_rank() == 0

# Import spatial context types (for GT-aware spatial augmentation)
try:
    from triangulang.utils.spatial_reasoning import (
        SpatialContext,
        InstanceSpatialInfo,
        build_spatial_context,
        compute_instance_spatial_info
    )
    HAS_SPATIAL_CONTEXT = True
except ImportError:
    HAS_SPATIAL_CONTEXT = False
    SpatialContext = None

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


# Label normalization: Fix typos and inconsistencies in ScanNet++ annotations
# Updated 2026-01-11 based on training run analysis (full_200s_8v_mix_ramp)
LABEL_FIXES = {
    # Typos
    'copmuter tower': 'computer tower',
    'moitor': 'monitor',
    'ceilng light': 'ceiling light',
    'ceiliing': 'ceiling',
    # Ceiling lamp typos moved to consolidation section below
    'exhaustive fan': 'exhaust fan',
    'cupoard': 'cupboard',
    'vaccum cleaner': 'vacuum cleaner',
    'vacum cleaner': 'vacuum cleaner',
    'robot vaccuum cleaner': 'robot vacuum cleaner',
    'bagpack': 'backpack',
    'backpag': 'backpack',
    'bakcpack': 'backpack',
    'dumbell': 'dumbbell',
    'chrismas tree': 'christmas tree',
    'folder oragnizer': 'folder organizer',
    'file orginizer': 'file organizer',
    'filer organizer': 'file organizer',
    'oragnizer': 'organizer',
    'bedisde table': 'bedside table',
    'botttle': 'bottle',
    'joga mat': 'yoga mat',
    'hangbag': 'handbag',
    'picutre': 'picture',
    'water boittle': 'water bottle',
    'piilow': 'pillow',
    'balnket': 'blanket',
    'keyabord': 'keyboard',
    'office chiar': 'office chair',
    'ofice chair': 'office chair',
    'whtieboard eraser': 'whiteboard eraser',
    'whteboard eraser': 'whiteboard eraser',
    'fire extuingisher': 'fire extinguisher',
    'fire extinquisher': 'fire extinguisher',
    'fire destinguisher': 'fire extinguisher',
    'fire estinguisher': 'fire extinguisher',
    'refirgerator': 'refrigerator',
    'wasching machine': 'washing machine',
    'toliet paper': 'toilet paper',
    'toliet paper rolls': 'toilet paper roll',
    'tissu box': 'tissue box',
    'carboard box': 'cardboard box',
    'smoke detecotr': 'smoke detector',
    'smoke dector': 'smoke detector',
    'smok alarm': 'smoke detector',  # typo found in eval analysis
    'suticase': 'suitcase',
    'remote controle': 'remote control',
    'slipers': 'slipper',
    'matress': 'mattress',
    'termostat': 'thermostat',
    'savon': 'soap',
    'savon container': 'soap dispenser',
    'powre socket': 'power outlet',
    'power socker': 'power outlet',  # typo found in eval analysis
    'covered power socket': 'power outlet',  # merge with power outlet

    # More typos
    'office  chair': 'office chair',  # double space
    'whitwboard eraser': 'whiteboard eraser',
    'kitachen cabinet': 'kitchen cabinet',
    'doorfraome': 'door frame',
    'intercome': 'intercom',
    'pi\u0307cture': 'picture',  # unicode dot above i
    'bukcet': 'bucket',
    'buket': 'bucket',  # typo found in eval analysis
    'teleephone': 'telephone',
    'bullettin boards': 'bulletin board',
    'shue rack': 'shoe rack',
    'door sper': 'door stopper',

    # Truncated labels
    'f': 'object',  # truncated
    't': 'object',  # truncated
    'tware': 'object',  # truncated
    'f pot': 'flower pot',  # likely truncated
    'lap': 'laptop',  # truncated

    # Combined labels
    'shower glass/delete this?': 'shower glass',
    'object/toiletry': 'toiletry',
    'bag/lap bag': 'laptop bag',
    'bathroom cabinet/bathroom counter': 'bathroom cabinet',
    'exhaust fan/vent': 'exhaust fan',
    'exhaust fan/ceiling vent': 'exhaust fan',
    'exhaust fan/ceiling ventilation valve': 'exhaust fan',
    'shower tub/shower floor': 'shower floor',

    # Class normalization
    # TV variants -> tv
    'television': 'tv',
    'tv screen': 'tv',
    'flat panel display': 'tv',
    'flat screen display': 'tv',

    # Trash variants -> trash can
    'trash bin': 'trash can',
    'trashcan': 'trash can',
    'trashbin': 'trash can',
    'dustbin': 'trash can',
    'wastebin': 'trash can',
    'garbage bin': 'trash can',
    'bin': 'trash can',
    'paper bin': 'trash can',

    # Sofa variants -> sofa
    'couch': 'sofa',

    # Whiteboard variants -> whiteboard
    'white board': 'whiteboard',

    # Bookshelf variants -> bookshelf
    'book shelf': 'bookshelf',

    # Nightstand variants -> nightstand
    'night stand': 'nightstand',
    'bedside table': 'nightstand',

    # Power outlet variants -> power outlet
    'power socket': 'power outlet',
    'power sockets': 'power outlet',
    'electrical outlet': 'power outlet',
    'electric outlet': 'power outlet',
    'wall outlet': 'power outlet',
    'electric socket': 'power outlet',
    'electrical socket': 'power outlet',
    'outlet': 'power outlet',
    'socket': 'power outlet',
    'multi socket': 'power outlet',

    # Flip flop variants -> flip flops
    'flip flop': 'flip flops',
    'flipflop': 'flip flops',
    'flipflops': 'flip flops',

    # Hair dryer variants -> hair dryer
    'hairdryer': 'hair dryer',

    # Cutting board variants -> cutting board
    'chopping board': 'cutting board',
    'cutboard': 'cutting board',

    # Door mat variants -> doormat
    'door mat': 'doormat',

    # Bean bag variants -> bean bag
    'beanbag': 'bean bag',
    'bean bag chair': 'bean bag',

    # Power strip variants -> power strip
    'powerstrip': 'power strip',
    'extension board': 'power strip',
    'power board': 'power strip',

    # Webcam variants
    'web cam': 'webcam',
    'computer camera': 'webcam',

    # Door frame variants
    'doorframe': 'door frame',

    # Window sill variants
    'windowsill': 'window sill',
    'window frame': 'window sill',
    'windowframe': 'window sill',

    # Mouse pad variants
    'mousepad': 'mouse pad',

    # Bathtub variants
    'bath tub': 'bathtub',

    # Briefcase variants
    'brief case': 'briefcase',

    # Duct variants -> electrical duct
    'electric duct': 'electrical duct',
    'e-duct': 'electrical duct',

    # Blind rail variants -> blind rail (singular)
    'blind rails': 'blind rail',

    # Switch board variants -> switchboard
    'switch board': 'switchboard',
    'brief': 'briefcase',

    # Tool box variants
    'toolbox': 'tool box',

    # Dust pan variants
    'dustpan': 'dust pan',

    # Cooking utensils variants
    'vat': 'wok pan',  # vat is common colloquial term for wok
    'wok': 'wok pan',  # normalize to full name

    # Joined tables variants
    'jioned tables': 'joined tables',

    # Bed sheet variants
    'bed  sheet': 'bedsheet',

    # Plural to singular
    # This ensures all instances of the same object type share a label
    # NOTE: Some plurals are INTENTIONALLY kept separate because they represent
    # grouped annotations (multiple objects in one mask) vs singular individual objects.
    # Analysis showed these have >1.5x larger coverage when plural:
    #   - 'books' (1.8x) - kept separate from 'book'
    #   - 'blinds' (31x) - kept separate from 'blind'
    #   - 'cables' (6.9x) - kept separate from 'cable'
    #   - 'cardboards' (3.0x) - kept separate from 'cardboard'
    #   - 'clothes' (2.5x) - kept separate from 'cloth'
    #   - 'headphones' (2.7x) - kept separate from 'headphone'
    #   - 'objects' (2.2x) - kept separate from 'object'
    #   - 'wall hooks' (3.5x) - kept separate from 'wall hook'
    # 'books': 'book',  # SEPARATED - plural annotations are grouped (1.8x larger)
    'bottles': 'bottle',
    'chairs': 'chair',
    'papers': 'paper',
    # 'objects': 'object',  # SEPARATED - plural annotations are grouped (2.2x larger)
    'blankets': 'blanket',
    'curtains': 'curtain',
    'pillows': 'pillow',
    'towels': 'towel',
    'sheets': 'sheet',
    'shoes': 'shoe',
    'slippers': 'slipper',
    'bags': 'bag',
    'bananas': 'banana',
    # 'blinds': 'blind',  # SEPARATED - plural annotations are grouped (31x larger)
    'boots': 'boot',
    # 'cables': 'cable',  # SEPARATED - plural annotations are grouped (6.9x larger)
    'cards': 'card',
    # 'cardboards': 'cardboard',  # SEPARATED - plural annotations are grouped (3.0x larger)
    'ceiling lamps': 'ceiling light',
    'ceiling lights': 'ceiling light',
    'cloth hangers': 'cloth hanger',
    # 'clothes': 'cloth',  # SEPARATED - plural annotations are grouped (2.5x larger)
    'clothes hanger': 'cloth hanger',
    'dumbbells': 'dumbbell',
    'earbuds': 'earbud',
    'files': 'file',
    'flowers': 'flower',
    'folders': 'folder',
    'fruits': 'fruit',
    'hangers': 'hanger',
    # 'headphones': 'headphone',  # SEPARATED - plural annotations are grouped (2.7x larger)
    'keyboards': 'keyboard',
    'lights': 'light',
    'magazines': 'magazine',
    'milk cartons': 'milk carton',
    'notes': 'note',
    'notebooks': 'notebook',
    'oven gloves': 'oven glove',
    'paintings': 'painting',
    'paper rolls': 'paper roll',
    'paper towels': 'paper towel',
    'paper trays': 'paper tray',
    'pens': 'pen',
    'pencils cup': 'pencil cup',
    'pipes': 'pipe',
    'plates': 'plate',
    'posters': 'poster',
    'scissors': 'scissor',
    'stairs': 'stair',
    'sticky notes': 'sticky note',
    'tissues': 'tissue',
    'toilet papers': 'toilet paper',
    'toilet paper rolls': 'toilet paper roll',
    # 'tools': 'tool',  # SEPARATED - plural annotations are grouped (1.8x larger)

    # Ambiguous labels
    # These are marked for review - use visualize_label.py to check
    # 'power switch': '???',  # Could be light switch or actual power switch
    # 'paper rolls': '???',  # Could be paper towel roll or toilet paper roll
    # 'power adapter': '???',  # Could be charger

    # More plural to singular
    'switches': 'switch',
    'sockets': 'socket',
    'coats': 'coat',
    'shirts': 'shirt',
    'jackets': 'jacket',
    'cushions': 'cushion',
    'baskets': 'basket',
    'buckets': 'bucket',
    'drawers': 'drawer',
    'crates': 'crate',
    'boxes': 'box',
    'cups': 'cup',
    'mugs': 'mug',
    'bowls': 'bowl',
    'jars': 'jar',
    'pots': 'pot',
    'pans': 'pan',
    'knives': 'knife',
    'spoons': 'spoon',
    'plants': 'plant',
    'pictures': 'picture',
    'mirrors': 'mirror',
    'tables': 'table',
    'desks': 'desk',
    'lockers': 'locker',
    'speakers': 'speaker',
    'candles': 'candle',
    'lamps': 'lamp',
    'rugs': 'rug',
    'mats': 'mat',
    'rods': 'rod',
    'weights': 'weight',
    'wires': 'wire',
    'plugs': 'plug',
    'hooks': 'hook',
    'knobs': 'knob',
    'handles': 'handle',
    'brushes': 'brush',
    'sponges': 'sponge',
    'towel racks': 'towel rack',
    'coat hangers': 'coat hanger',
    'clothes hangers': 'cloth hanger',
    'light switches': 'light switch',
    'light switchs': 'light switch',
    'power switches': 'power switch',
    'power outlets': 'power outlet',
    'smoke detectors': 'smoke detector',
    'fire extinguishers': 'fire extinguisher',
    'ceiling fans': 'ceiling fan',
    'trash cans': 'trash can',
    'trash bins': 'trash can',
    'trash bags': 'trash bag',
    'cardboard boxes': 'cardboard box',

    # More typos
    'covred power socket': 'power outlet',
    'toilet paper dispensor': 'toilet paper dispenser',
    # ceiling lamp typos consolidated in section below
    'ceiliing': 'ceiling',
    'liight switch': 'light switch',
    'light swich': 'light switch',
    'light swicth': 'light switch',
    'light swtich': 'light switch',
    'swtich': 'switch',
    'botlle': 'bottle',
    'vottle': 'bottle',
    'boox': 'book',
    'kettlle': 'kettle',
    'cettel': 'kettle',
    'intercorm': 'intercom',
    'orginizer': 'organizer',
    'foset': 'faucet',
    'fosit': 'faucet',
    'facuet': 'faucet',
    'wal': 'wall',
    'flor': 'floor',
    'helf': 'shelf',
    'shelve': 'shelf',
    'shelves': 'shelf',
    'loth': 'cloth',
    'clothng rail': 'clothes rail',
    'clithes': 'clothes',

    # Class consolidation
    # Remote variants -> remote control
    'remote': 'remote control',
    'remote controller': 'remote control',
    'tv remote': 'remote control',
    'tv controller': 'remote control',

    # Fridge variants -> refrigerator
    'fridge': 'refrigerator',
    'mini fridge': 'refrigerator',
    'lab fridge': 'refrigerator',

    # Keyboard variants (computer)
    'keyabord': 'keyboard',

    # Monitor variants -> monitor
    'computer monitor': 'monitor',
    'computer screen': 'monitor',

    # PC variants -> computer
    'pc': 'computer',
    'pc tower': 'computer',
    'tower pc': 'computer',
    'cpu': 'computer',

    # Blinds variants -> blind (all map directly to 'blind', no chaining!)
    # 'blinds': 'blind',  # SEPARATED - plural annotations are grouped (31x larger coverage)
    'window blind': 'blind',
    'roller blinds': 'blind',
    'rolling blinds': 'blind',
    'vertical blinds': 'blind',
    'window blinds': 'blind',

    # Filter out invalid labels
    'REMOVE': 'object',
    'remove': 'object',
    'SPLIT': 'object',
    'split': 'object',
    'bottommost tware': 'object',  # Garbage annotation
    'f': 'object',  # Single letter - garbage
    'fs': 'object',  # Garbage
    't': 'object',  # Single letter - garbage
    'c': 'object',  # Single letter - garbage
    'o': 'object',  # Single letter - garbage
    'datashow socket': 'object',  # Too specific/rare

    # More typos
    'storage ecabinet': 'storage cabinet',
    'recessed shelve': 'recessed shelf',
    'garbage bin cover': 'trash can lid',

    # More typos
    'cabel': 'cable',
    'curboard': 'cupboard',
    'ceilng lamp': 'ceiling light',
    'ballon': 'balloon',
    'electrucal duct': 'electrical duct',
    'ligth switch': 'light switch',
    'deodrant': 'deodorant',
    'show curtain rod': 'shower curtain rod',
    'computer towerl': 'computer tower',
    'lounge pug': 'lounge rug',
    'phone charger]': 'phone charger',
    'smoke alarme': 'smoke alarm',
    'coarser': 'coaster',

    # Duplicate consolidation
    # Tube light variants -> tube light
    'tubelight': 'tube light',
    'tube lights': 'tube light',

    # Copier -> copy machine (user preference)
    'copier': 'copy machine',

    # Flush button -> toilet flush button
    'flush button': 'toilet flush button',

    # Keyboard consolidation (computer keyboard -> keyboard)
    'computer keyboard': 'keyboard',

    # Projector consolidation -> projector
    'overhead projector': 'projector',

    # Ceiling light consolidation -> ceiling light
    'ceiling lamp': 'ceiling light',
    'ceilng lamp bar': 'ceiling light',
    'ceiilng lamp': 'ceiling light',
    'celing lamp': 'ceiling light',
    'ceiling lmap': 'ceiling light',
    'ceiling lapm': 'ceiling light',

    # Book consolidation -> book
    'textbook': 'book',
    'novel': 'book',

    # Cardboard box consolidation
    'cardbox': 'cardboard box',

    # Storage -> storage cabinet
    'storage': 'storage cabinet',

    # Stapler consolidation
    'paper stapler': 'stapler',

    # Table variants (keep tabletop separate - different concept)
    # 'tabletop': 'table',  # NOT merging - tabletop is surface, table is furniture

    # More typos
    'cardboads': 'cardboard',       # typo for 'cardboards'
    'cardboards': 'cardboard',      # plural -> singular
    'kitchen shelve': 'kitchen shelf',  # grammar error
    'fusebox': 'fuse box',          # missing space
    'stovetop': 'stove top',        # missing space
    'tooth brush': 'toothbrush',    # extra space
}

# Bad annotations to exclude: (scene_id, object_id) pairs with incorrect masks
# Detected via segment count outlier analysis (objects with >100x median segment count)
BAD_ANNOTATIONS = {
    ('3db0a1c8f3', 86),   # 'light switch' has 24371 segs (median=142) - clearly wrong
    ('3db0a1c8f3', 54),   # 'remove' with 164922 segs
    ('ab11145646', 96),   # 'remove' with 119930 segs
    ('cf1ffd871d', 26),   # 'remove' with 116861 segs
}

# Frames to exclude per scene: {scene_id: {frame_stem, ...}}
# These frames have catastrophic DA3 depth errors (e.g. 609cm Procrustes error)
EXCLUDE_FRAMES = {
    'c4c04e6d6c': {'DSC03071'},  # 609cm Procrustes error, rest of scene is 3.7cm
}


def is_excluded_frame(scene_id: str, frame_stem: str) -> bool:
    """Check if a frame should be excluded due to known DA3 depth errors."""
    excluded = EXCLUDE_FRAMES.get(scene_id)
    return excluded is not None and frame_stem in excluded


def is_bad_annotation(scene_id: str, obj_id: int) -> bool:
    """Check if an annotation should be excluded due to known errors."""
    return (scene_id, obj_id) in BAD_ANNOTATIONS


def normalize_label(label: str) -> str:
    """Fix typos and normalize labels from ScanNet++ annotations."""
    label = label.strip()
    # Collapse double (or more) spaces to single space
    while '  ' in label:
        label = label.replace('  ', ' ')
    # Strip trailing brackets/artifacts
    label = label.rstrip(']').rstrip('[').strip()
    return LABEL_FIXES.get(label, label)

# Add scannetpp_toolkit to path for utilities
_toolkit_path = Path(__file__).parent.parent.parent / "scannetpp_toolkit"
if _toolkit_path.exists() and str(_toolkit_path) not in sys.path:
    sys.path.insert(0, str(_toolkit_path))


def get_scenes_dir(data_root: Path) -> Path:
    """Get the directory containing scene folders (handles nested 'data' folder)."""
    # ScanNet++ download creates: data_root/data/<scene_id>/
    nested = data_root / "data"
    if nested.exists() and nested.is_dir():
        return nested
    return data_root


def load_scene_list(data_root: Path, split: str) -> List[str]:
    """
    Load scene IDs from split file.

    Args:
        data_root: Path to scannetpp folder
        split: One of 'nvs_sem_train', 'nvs_sem_val', 'nvs_test', 'sem_test'

    Returns:
        List of scene IDs
    """
    split_file = data_root / "splits" / f"{split}.txt"
    if not split_file.exists():
        # Try alternative locations
        for alt_path in [
            data_root / f"{split}.txt",
            data_root / "metadata" / f"{split}.txt"
        ]:
            if alt_path.exists():
                split_file = alt_path
                break

    if not split_file.exists():
        print(f"Warning: Split file not found: {split_file}")
        return []

    with open(split_file) as f:
        scenes = [line.strip() for line in f if line.strip()]
    return scenes


def load_semantic_classes(data_root: Path) -> Dict[int, str]:
    """Load semantic class mapping from metadata."""
    classes_file = data_root / "metadata" / "semantic_classes.txt"
    if not classes_file.exists():
        return {}

    classes = {}
    with open(classes_file) as f:
        for i, line in enumerate(f):
            name = line.strip()
            if name:
                classes[i] = name
    return classes


def load_nerfstudio_transforms(transforms_path: Path) -> Dict:
    """
    Load camera transforms from nerfstudio format.

    Returns dict with:
        - frames: list of {file_path, transform_matrix, ...}
        - camera intrinsics (fl_x, fl_y, cx, cy, w, h)
    """
    if not transforms_path.exists():
        return None

    with open(transforms_path) as f:
        data = json.load(f)
    return data


def load_train_test_split(scene_path: Path) -> Tuple[List[str], List[str]]:
    """Load image train/test split for a scene."""
    split_file = scene_path / "dslr" / "train_test_lists.json"
    if not split_file.exists():
        return [], []

    with open(split_file) as f:
        data = json.load(f)

    train_images = data.get('train', [])
    test_images = data.get('test', [])
    return train_images, test_images


def get_available_scenes(data_root: Path, split: str = None) -> List[str]:
    """
    Get available scene IDs with valid data.

    Args:
        data_root: Path to scannetpp folder
        split: Optional split to filter by

    Returns:
        List of valid scene IDs
    """
    data_root = Path(data_root)
    scenes_dir = get_scenes_dir(data_root)

    # Get scenes from split file if specified
    if split:
        scene_ids = load_scene_list(data_root, split)
    else:
        # Find all scene directories
        scene_ids = [d.name for d in scenes_dir.iterdir()
                     if d.is_dir() and not d.name.startswith('.')
                     and d.name not in ['splits', 'metadata', 'data']]

    # Filter to scenes with valid DSLR data
    valid_scenes = []
    for scene_id in scene_ids:
        scene_path = scenes_dir / scene_id
        dslr_dir = scene_path / "dslr"

        # Check for required data
        if not dslr_dir.exists():
            continue

        # Check for images
        images_dir = dslr_dir / "resized_images"
        if not images_dir.exists():
            images_dir = dslr_dir / "resized_undistorted_images"

        if images_dir.exists() and len(list(images_dir.glob("*.JPG"))) > 0:
            valid_scenes.append(scene_id)

    return valid_scenes


def load_semantic_annotations(scene_path: Path) -> Dict[str, List[int]]:
    """
    Load semantic annotations for a scene.

    Returns:
        Dict mapping object label -> list of segment indices
    """
    anno_file = scene_path / "scans" / "segments_anno.json"
    if not anno_file.exists():
        return {}

    with open(anno_file) as f:
        data = json.load(f)

    # Parse annotations
    annotations = {}
    for item in data.get('segGroups', []):
        label = normalize_label(item.get('label', 'unknown'))
        segments = item.get('segments', [])
        if label not in annotations:
            annotations[label] = []
        annotations[label].extend(segments)

    return annotations


# Common indoor object prompts for ScanNet++
SCANNETPP_PROMPTS = {
    # Furniture
    'chair': ['chair', 'office chair', 'desk chair', 'armchair', 'seat'],
    'table': ['table', 'desk', 'dining table', 'coffee table', 'work table'],
    'sofa': ['sofa', 'couch', 'loveseat', 'settee'],
    'bed': ['bed', 'mattress', 'bedroom furniture'],
    'cabinet': ['cabinet', 'cupboard', 'storage cabinet', 'kitchen cabinet'],
    'shelf': ['shelf', 'bookshelf', 'shelving', 'rack'],
    'door': ['door', 'doorway', 'entrance'],
    'window': ['window', 'glass window', 'window frame'],
    # Electronics
    'monitor': ['monitor', 'computer screen', 'display', 'TV screen'],
    'keyboard': ['keyboard', 'computer keyboard'],
    'lamp': ['lamp', 'light', 'desk lamp', 'floor lamp', 'lighting fixture'],
    'tv': ['TV', 'television', 'flat screen', 'display'],
    # Kitchen
    'refrigerator': ['refrigerator', 'fridge', 'freezer'],
    'microwave': ['microwave', 'microwave oven'],
    'sink': ['sink', 'kitchen sink', 'bathroom sink', 'basin'],
    'toilet': ['toilet', 'bathroom toilet', 'lavatory'],
    # Objects
    'book': ['book', 'books', 'textbook', 'notebook'],
    'plant': ['plant', 'potted plant', 'houseplant', 'flower pot'],
    'bottle': ['bottle', 'water bottle', 'container'],
    'box': ['box', 'cardboard box', 'storage box'],
    'bag': ['bag', 'backpack', 'handbag', 'shopping bag'],
    'pillow': ['pillow', 'cushion', 'throw pillow'],
    'blanket': ['blanket', 'throw', 'bedding'],
    'curtain': ['curtain', 'drape', 'window covering'],
    'picture': ['picture', 'painting', 'artwork', 'photo frame'],
    'mirror': ['mirror', 'wall mirror', 'bathroom mirror'],
    # Structure
    'wall': ['wall', 'room wall'],
    'floor': ['floor', 'flooring', 'ground'],
    'ceiling': ['ceiling', 'room ceiling'],
}


class ScanNetPPDataset(Dataset):
    """
    Dataset for ScanNet++ single-view training.
    Each sample is a single image with semantic prompts.

    Since ScanNet++ doesn't have per-image masks, we use:
    - SAM3's text prompting for segmentation
    - 3D mesh for ground truth (project to 2D if needed)
    """

    def __init__(
        self,
        data_root: Path,
        split: str = 'nvs_sem_train',
        images_per_scene: int = 10,
        image_size: Tuple[int, int] = (518, 518),
        mask_size: Tuple[int, int] = (128, 128),
        use_undistorted: bool = True,
        max_scenes: int = None
    ):
        self.data_root = Path(data_root)
        self.scenes_dir = get_scenes_dir(self.data_root)
        self.image_size = image_size
        self.mask_size = mask_size
        self.use_undistorted = use_undistorted

        # Get valid scenes
        scenes = get_available_scenes(self.data_root, split)
        if max_scenes:
            scenes = scenes[:max_scenes]

        # Build samples: (image_path, scene_id, prompt)
        self.samples = []
        self.scene_transforms = {}  # Cache transforms per scene

        print(f"\nLoading ScanNet++ dataset ({split})...")

        from tqdm import tqdm

        for scene_id in tqdm(scenes, desc="Loading scenes"):
            scene_path = self.scenes_dir / scene_id

            # Get image directory
            if use_undistorted:
                images_dir = scene_path / "dslr" / "resized_undistorted_images"
            else:
                images_dir = scene_path / "dslr" / "resized_images"

            if not images_dir.exists():
                continue

            # Get train images only
            train_images, _ = load_train_test_split(scene_path)
            if not train_images:
                # Fall back to all images
                train_images = [f.name for f in images_dir.glob("*.JPG")]

            # Load transforms for camera parameters
            transforms_path = scene_path / "dslr" / "nerfstudio" / "transforms.json"
            if use_undistorted:
                transforms_path = scene_path / "dslr" / "nerfstudio" / "transforms_undistorted.json"

            transforms = load_nerfstudio_transforms(transforms_path)
            if transforms:
                self.scene_transforms[scene_id] = transforms

            # Load semantic annotations for prompts
            annotations = load_semantic_annotations(scene_path)
            object_labels = list(annotations.keys())

            # Sample images from scene
            if len(train_images) > images_per_scene:
                selected = random.sample(train_images, images_per_scene)
            else:
                selected = train_images

            for img_name in selected:
                img_path = images_dir / img_name
                if img_path.exists():
                    # Choose a random object prompt from scene annotations
                    if object_labels:
                        label = random.choice(object_labels)
                        if label in SCANNETPP_PROMPTS:
                            prompt = SCANNETPP_PROMPTS[label][0]
                        else:
                            prompt = label
                    else:
                        # Fallback to common indoor prompts
                        prompt = random.choice(['chair', 'table', 'monitor', 'lamp'])

                    self.samples.append((img_path, scene_id, prompt))

        print(f"Loaded {len(self.samples)} samples from {len(self.scene_transforms)} scenes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, scene_id, prompt = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')
        orig_size = image.size  # (W, H)
        image = image.resize(self.image_size, Image.BILINEAR)
        image = torch.from_numpy(np.array(image)).float() / 255.0
        image = image.permute(2, 0, 1)  # HWC -> CHW

        # Get camera parameters if available
        intrinsics = None
        extrinsics = None

        if scene_id in self.scene_transforms:
            transforms = self.scene_transforms[scene_id]

            # Find this frame's transform
            img_name = img_path.name
            for frame in transforms.get('frames', []):
                if img_name in frame.get('file_path', ''):
                    extrinsics = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
                    break

            # Intrinsics from transforms
            intrinsics = torch.tensor([
                [transforms.get('fl_x', 500), 0, transforms.get('cx', 256)],
                [0, transforms.get('fl_y', 500), transforms.get('cy', 256)],
                [0, 0, 1]
            ], dtype=torch.float32)

        return {
            'image': image,
            'prompt': prompt,
            'scene_id': scene_id,
            'image_name': img_path.name,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'orig_size': orig_size,
            'has_metric_scale': True,  # ScanNet++ has metric GT
            'has_gt_mask': False  # No per-image masks, use SAM3 predictions
        }


# GT Mask Generation from Mesh Rasterization (MV-SAM style)

def load_vertex_object_ids(scene_path: Path) -> Tuple[np.ndarray, Dict]:
    """
    Load vertex-to-object-ID mapping from annotations.

    Returns:
        vertex_obj_ids: (N_vertices,) array mapping each vertex to object ID (0 = background)
        objects: Dict mapping obj_id -> {label, segments, obb, ...}
    """
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


# Multi-View Dataset with Supervised Training Support

class ScanNetPPMultiViewDataset(Dataset):
    """
    Dataset for ScanNet++ multi-view training.
    Each sample is multiple views of the same indoor scene.

    Key advantage: METRIC SCALE ground truth from LiDAR.

    With supervised=True, uses pre-computed obj_ids from ScanNet++ toolkit:
    - Dense GT masks from mesh rasterization (pre-computed)
    - Objects filtered by min visibility (0.1% of image pixels)
    - Ready for focal + dice loss training

    Pre-computed obj_ids are loaded from:
        data_root/semantics_2d_train/obj_ids/{scene_id}/{image_name}.pth
    """

    def __init__(
        self,
        data_root: Path,
        split: str = 'nvs_sem_train',
        views_per_sample: int = 8,
        image_size: Tuple[int, int] = (518, 518),
        mask_size: Tuple[int, int] = (128, 128),
        use_undistorted: bool = True,
        max_scenes: int = None,
        sampling_strategy: str = 'stratified',  # 'random', 'stratified', 'sequential', 'chunk_aware'
        da3_chunk_size: int = 8,  # Chunk size for DA3-NESTED (used with 'chunk_aware' sampling)
        supervised: bool = True,  # Use pre-computed GT masks
        min_object_pixels: float = 0.001,  # Min fraction of image (0.1%)
        raster_cache_dir: Path = None,  # Legacy - kept for compatibility
        obj_ids_dir: str = None,  # Dir with pre-computed obj_ids (e.g., 'semantics_2d_train')
        samples_per_scene: int = 10,  # Number of 8-view samples per scene per epoch (only used if enumerate_all_objects=False)
        enumerate_all_objects: bool = True,  # Iterate through ALL objects deterministically (default)
        semantic_union: bool = True,  # NEW: Use union of all instances with same label for GT
        use_cached_depth: bool = False,  # Load pre-computed DA3 depth from da3_cache/
        da3_cache_name: str = 'da3_cache',  # Cache directory name: 'da3_cache' or 'da3_nested_cache'
        min_category_samples: int = 1,  # Filter out categories with fewer samples (1 = no filtering)
        exclude_categories: List[str] = None,  # Categories to exclude from training (e.g., ['wall', 'floor', 'ceiling'])
        include_categories: List[str] = None,  # Whitelist specific categories (None = all)
        num_objects_per_sample: int = 1,  # Number of GT objects per sample (K>1 = multi-object training)
    ):
        self.data_root = Path(data_root)
        self.min_category_samples = min_category_samples
        self.exclude_categories = set(exclude_categories) if exclude_categories else set()
        self.include_categories = set(include_categories) if include_categories else None
        self.scenes_dir = get_scenes_dir(self.data_root)
        self.image_size = image_size
        self.mask_size = mask_size
        self.views_per_sample = views_per_sample
        self.use_undistorted = use_undistorted
        self.sampling_strategy = sampling_strategy
        self.da3_chunk_size = da3_chunk_size  # For 'chunk_aware' sampling with DA3-NESTED
        self.enumerate_all_objects = enumerate_all_objects
        self.supervised = supervised
        self.min_object_pixels = min_object_pixels
        self.raster_cache_dir = Path(raster_cache_dir) if raster_cache_dir else None
        self.samples_per_scene = samples_per_scene
        self.semantic_union = semantic_union  # NEW: Union all instances of same class
        self.num_objects_per_sample = num_objects_per_sample  # Multi-object training: K objects per sample
        self.use_cached_depth = use_cached_depth  # Load pre-computed DA3 depth
        self.da3_cache_name = da3_cache_name  # Support da3_cache or da3_nested_cache

        # Set up DA3 cache directory
        self.da3_cache_dir = self.data_root / da3_cache_name if use_cached_depth else None
        if use_cached_depth and self.da3_cache_dir and not self.da3_cache_dir.exists():
            if _is_main_process():
                print(f"Warning: DA3 cache directory not found: {self.da3_cache_dir}")
                print(f"  Run scripts/preprocess_da3.py or preprocess_da3_nested.py first.")

        # Load pre-computed GT centroids (avoids trimesh loading during training)
        self.centroid_cache = {}
        centroid_cache_path = self.data_root / "centroid_cache.json"
        if centroid_cache_path.exists():
            with open(centroid_cache_path) as f:
                self.centroid_cache = json.load(f)
            if _is_main_process():
                print(f"Loaded GT centroid cache: {len(self.centroid_cache)} scenes")

        # Set up pre-computed 2D mask directory
        # Structure: semantics_2d_train/{scene_id}/{frame}.JPG.pth
        if obj_ids_dir:
            self.obj_ids_root = self.data_root / obj_ids_dir
        elif 'train' in split:
            self.obj_ids_root = self.data_root / "semantics_2d_train"
        else:
            self.obj_ids_root = self.data_root / "semantics_2d_val"

        # Get valid scenes
        scenes = get_available_scenes(self.data_root, split)
        if max_scenes:
            scenes = scenes[:max_scenes]

        # Build scene list with metadata
        self.scenes = []  # (scene_id, scene_path, train_images, transforms, annotations)
        self.rasterizers = OrderedDict()  # Legacy - kept for compatibility

        if _is_main_process():
            print(f"\nLoading ScanNet++ multi-view dataset ({split}, supervised={supervised})...")

        from tqdm import tqdm
        skipped = 0

        for scene_id in tqdm(scenes, desc="Loading scenes", disable=not _is_main_process()):
            scene_path = self.scenes_dir / scene_id

            # Get image directory
            if use_undistorted:
                images_dir = scene_path / "dslr" / "resized_undistorted_images"
            else:
                images_dir = scene_path / "dslr" / "resized_images"

            if not images_dir.exists():
                skipped += 1
                continue

            # Get train images only
            train_images, _ = load_train_test_split(scene_path)
            if not train_images:
                train_images = sorted([f.name for f in images_dir.glob("*.JPG")])

            # Need enough views
            if len(train_images) < views_per_sample:
                skipped += 1
                continue

            # Load transforms
            transforms_path = scene_path / "dslr" / "nerfstudio" / "transforms.json"
            if use_undistorted:
                transforms_path = scene_path / "dslr" / "nerfstudio" / "transforms_undistorted.json"
            transforms = load_nerfstudio_transforms(transforms_path)

            # Load annotations
            annotations = load_semantic_annotations(scene_path)

            # For supervised mode, check we have pre-computed obj_ids
            if supervised:
                scene_obj_ids_dir = self.obj_ids_root / scene_id
                if not scene_obj_ids_dir.exists():
                    skipped += 1
                    continue
                # Filter train_images to only those with pre-computed obj_ids
                available_obj_ids = {p.stem for p in scene_obj_ids_dir.glob("*.pth")}
                train_images = [img for img in train_images if img in available_obj_ids]
                if len(train_images) < views_per_sample:
                    skipped += 1
                    continue

            # Filter out excluded frames (DA3 depth errors)
            if scene_id in EXCLUDE_FRAMES:
                before = len(train_images)
                train_images = [img for img in train_images
                                if (Path(img).stem if '.' in img else img) not in EXCLUDE_FRAMES[scene_id]]
                if len(train_images) < before and _is_main_process():
                    print(f"  {scene_id}: excluded {before - len(train_images)} bad frames")

            # Build obj_id -> label and label -> [obj_ids] mappings for semantic union
            obj_to_label = {}
            label_to_obj_ids = defaultdict(list)
            anno_path = scene_path / "scans" / "segments_anno.json"
            if anno_path.exists():
                with open(anno_path) as f:
                    anno_data = json.load(f)
                for group in anno_data.get('segGroups', []):
                    obj_id = group.get('objectId') or group.get('id')
                    label = normalize_label(group.get('label', 'object'))
                    if obj_id is not None:
                        obj_to_label[obj_id] = label
                        label_to_obj_ids[label].append(obj_id)

            self.scenes.append({
                'scene_id': scene_id,
                'scene_path': scene_path,
                'images_dir': images_dir,
                'train_images': train_images,
                'transforms': transforms,
                'annotations': annotations,
                'obj_to_label': obj_to_label,  # NEW: for semantic union
                'label_to_obj_ids': dict(label_to_obj_ids),  # NEW: for semantic union
            })

        if _is_main_process():
            print(f"Loaded {len(self.scenes)} scenes ({skipped} skipped)")

        # Build chunk membership map for DA3 nested cache (used by _get_chunk_groups at runtime)
        # This avoids loading .pt files on every __getitem__ call, which caused DDP deadlocks
        self._scene_chunk_map = {}  # scene_id -> {stem -> chunk_idx}
        if self.use_cached_depth and self.da3_cache_dir and self.da3_cache_dir.exists():
            for scene in self.scenes:
                scene_id = scene['scene_id']
                da3_scene_dir = self.da3_cache_dir / scene_id
                if not da3_scene_dir.exists():
                    continue
                da3_cached_stems = sorted(f.stem for f in da3_scene_dir.glob("*.pt"))
                if not da3_cached_stems:
                    continue
                # Read one file to get actual chunk size
                sample_data = torch.load(
                    da3_scene_dir / f"{da3_cached_stems[0]}.pt",
                    map_location='cpu', weights_only=True, mmap=False
                )
                cframes = sample_data.get('chunk_frames', [])
                actual_chunk_size = len(cframes) if cframes else 16
                # Reconstruct chunk membership from sorted order
                stem_to_chunk = {}
                for stem_idx, stem in enumerate(da3_cached_stems):
                    stem_to_chunk[stem] = stem_idx // actual_chunk_size
                self._scene_chunk_map[scene_id] = stem_to_chunk
            if self._scene_chunk_map and _is_main_process():
                print(f"Built DA3 chunk map for {len(self._scene_chunk_map)} scenes")

        # NOTE: We don't pre-load SAM3 cache into RAM to avoid memory issues with DDP
        # (each rank would duplicate the cache, exceeding available RAM)
        # Instead, we load from disk on-demand and rely on OS page cache for subsequent accesses

        # Scene-grouped mode: when num_objects_per_sample == 0 (dynamic/all), iterate per-scene.
        # Each __getitem__ discovers ALL visible objects dynamically → one forward pass per scene.
        # Dataset length = len(scenes) * samples_per_scene for proper batching.
        self.scene_grouped = (num_objects_per_sample == 0)

        # If enumerate_all_objects, pre-compute all (scene_idx, obj_id, label) tuples
        # OPTIMIZATION: Only include objects visible in >= views_per_sample images
        # and store which images contain each object for smart view sampling
        self.object_samples = []
        if enumerate_all_objects and supervised and not self.scene_grouped:
            # Generate cache key based on config
            # Include sampling_strategy so chunk-filtered caches don't get reused for non-chunk strategies
            needs_chunk_filter = sampling_strategy in ('chunk_aware', 'overlap_30', 'overlap_50')
            chunk_tag = f"chunk_{da3_cache_name}" if (use_cached_depth and needs_chunk_filter) else "nochunk"
            cache_key_data = f"{split}_{max_scenes}_{views_per_sample}_{min_object_pixels}_{len(self.scenes)}_{chunk_tag}"
            cache_key = hashlib.md5(cache_key_data.encode()).hexdigest()[:12]
            cache_path = self.data_root / f".object_samples_cache_{cache_key}.pkl"

            # Try to load from cache (with file locking for multi-process safety)
            LOADER_DEBUG = os.environ.get('LOADER_DEBUG', '0') == '1'
            lock_path = cache_path.with_suffix('.lock')

            if cache_path.exists():
                if LOADER_DEBUG:
                    print(f"[LOADER_DEBUG] Attempting to load cache: {cache_path.name}", flush=True)
                if _is_main_process():
                    print(f"Loading object samples from cache: {cache_path.name}")
                try:
                    # Use shared lock for reading
                    with open(lock_path, 'w') as lock_file:
                        if LOADER_DEBUG:
                            print(f"[LOADER_DEBUG] Acquiring shared lock for read...", flush=True)
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
                        try:
                            with open(cache_path, 'rb') as f:
                                cached_data = pickle.load(f)
                            self.object_samples = cached_data['object_samples']
                            if _is_main_process():
                                print(f"  Loaded {len(self.object_samples)} object samples from cache")
                        finally:
                            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                            if LOADER_DEBUG:
                                print(f"[LOADER_DEBUG] Released shared lock", flush=True)
                except Exception as e:
                    print(f"  Cache load failed: {e}, re-enumerating...")
                    self.object_samples = []

            # Enumerate if not cached (with lock to prevent race condition)
            if not self.object_samples:
                # Acquire exclusive lock before checking again (another process may have created cache)
                with open(lock_path, 'w') as lock_file:
                    if LOADER_DEBUG:
                        print(f"[LOADER_DEBUG] Acquiring exclusive lock to check/build cache...", flush=True)
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                    try:
                        # Re-check if cache was created while we waited for the lock
                        if cache_path.exists():
                            if LOADER_DEBUG:
                                print(f"[LOADER_DEBUG] Cache appeared while waiting, loading it...", flush=True)
                            try:
                                with open(cache_path, 'rb') as f:
                                    cached_data = pickle.load(f)
                                self.object_samples = cached_data['object_samples']
                                if _is_main_process():
                                    print(f"  Loaded {len(self.object_samples)} object samples from cache (created by another process)")
                            except Exception as e:
                                print(f"  Cache load failed: {e}, will enumerate...")
                    finally:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

            # Now enumerate if still not loaded
            if not self.object_samples:
                if _is_main_process():
                    print("Enumerating all objects across scenes (will cache for future runs)...")
                total_objects_found = 0
                filtered_out = 0
                total_chunk_filtered = 0

                for scene_idx, scene in enumerate(tqdm(self.scenes, desc="Enumerating objects", disable=not _is_main_process())):
                    scene_id = scene['scene_id']
                    scene_path = scene['scene_path']
                    train_images_set = set(scene['train_images'])  # Only consider train images

                    # Load annotations to get labels
                    anno_path = scene_path / "scans" / "segments_anno.json"
                    obj_to_label = {}
                    if anno_path.exists():
                        with open(anno_path) as f:
                            anno_data = json.load(f)
                        for group in anno_data.get('segGroups', []):
                            obj_id = group.get('objectId') or group.get('id')
                            label = normalize_label(group.get('label', 'object'))
                            if obj_id is not None:
                                obj_to_label[obj_id] = label

                    # Track per-object visibility: obj_id -> set of image names where it's visible
                    obj_visibility = defaultdict(set)

                    scene_obj_ids_dir = self.obj_ids_root / scene_id
                    for pth_file in scene_obj_ids_dir.glob("*.pth"):
                        img_name = pth_file.stem  # e.g., "DSC00001.JPG"
                        # Only consider images in train split
                        if img_name not in train_images_set:
                            continue
                        try:
                            pix_obj_ids = torch.load(pth_file, weights_only=False)
                            unique_ids = np.unique(pix_obj_ids)
                            for obj_id in unique_ids:
                                if obj_id > 0 and obj_id in obj_to_label:
                                    label = obj_to_label[obj_id]
                                    # Skip invalid/generic labels + structural elements (matches evaluate_gasa.py)
                                    skip_labels = [
                                        # Generic/invalid annotations
                                        'remove', 'split', 'object', 'objects', 'stuff', 'unknown',
                                        # Structural elements (too easy, not interesting for segmentation)
                                        'wall', 'floor', 'ceiling', 'door', 'window', 'doorframe', 'window frame',
                                        # Annotation artifacts
                                        'reflection', 'mirror', 'structure',
                                    ]
                                    if label.lower() not in skip_labels:
                                        obj_visibility[obj_id].add(img_name)
                        except:
                            continue

                    # Add objects that are visible in at least views_per_sample images
                    # Use pre-built chunk membership map (built once in __init__)
                    img_to_chunk_idx = self._scene_chunk_map.get(scene_id, {})

                    bad_annotation_count = 0
                    chunk_filtered_out = 0
                    for obj_id, visible_images in obj_visibility.items():
                        total_objects_found += 1
                        # Skip known bad annotations
                        if is_bad_annotation(scene_id, obj_id):
                            bad_annotation_count += 1
                            continue

                        if len(visible_images) < views_per_sample:
                            filtered_out += 1
                            continue

                        # Check chunk feasibility: only needed for chunk_aware/overlap strategies
                        # stratified/random/sequential sample from ALL visible images, so they
                        # don't need any single chunk to have enough views
                        if img_to_chunk_idx and self.sampling_strategy in ('chunk_aware', 'overlap_30', 'overlap_50'):
                            chunk_counts = Counter()
                            for img in visible_images:
                                stem = Path(img).stem if '.' in img else img
                                if stem in img_to_chunk_idx:
                                    chunk_counts[img_to_chunk_idx[stem]] += 1
                            max_in_chunk = max(chunk_counts.values()) if chunk_counts else 0
                            if max_in_chunk < views_per_sample:
                                chunk_filtered_out += 1
                                filtered_out += 1
                                continue

                        label = obj_to_label.get(obj_id, 'object')
                        self.object_samples.append({
                            'scene_idx': scene_idx,
                            'obj_id': obj_id,
                            'label': label,
                            'visible_images': list(visible_images)  # For smart view sampling (already filtered to cached)
                        })
                    total_chunk_filtered += chunk_filtered_out
                    if bad_annotation_count > 0:
                        print(f"    Skipped {bad_annotation_count} bad annotations in {scene_id}")

                if _is_main_process():
                    print(f"  Total objects found: {total_objects_found}")
                    print(f"  Filtered out (visible in <{views_per_sample} views): {filtered_out}")
                    if total_chunk_filtered > 0:
                        print(f"    (of which {total_chunk_filtered} had enough views overall but no single DA3 chunk with >={views_per_sample})")
                    print(f"  Valid object samples: {len(self.object_samples)}")

                # Save to cache (with exclusive lock for multi-process safety)
                if _is_main_process():
                    print(f"  Saving to cache: {cache_path.name}")
                if LOADER_DEBUG:
                    print(f"[LOADER_DEBUG] Acquiring exclusive lock for write...", flush=True)
                with open(lock_path, 'w') as lock_file:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                    try:
                        with open(cache_path, 'wb') as f:
                            pickle.dump({'object_samples': self.object_samples}, f)
                        if LOADER_DEBUG:
                            print(f"[LOADER_DEBUG] Cache saved successfully", flush=True)
                    finally:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                        if LOADER_DEBUG:
                            print(f"[LOADER_DEBUG] Released exclusive lock", flush=True)
        else:
            if _is_main_process():
                print(f"  {self.samples_per_scene} samples/scene = {len(self.scenes) * self.samples_per_scene} samples/epoch")

        if self.scene_grouped and _is_main_process():
            print(f"  Scene-grouped mode: {len(self.scenes)} scenes/epoch "
                  f"(all visible objects per scene, ~10-30× fewer iterations)")

        # Filter out excluded categories (e.g., wall, floor, ceiling)
        if self.exclude_categories and self.object_samples:
            before_count = len(self.object_samples)
            self.object_samples = [s for s in self.object_samples
                                   if s['label'] not in self.exclude_categories]
            excluded_count = before_count - len(self.object_samples)
            if excluded_count > 0 and _is_main_process():
                print(f"  Excluded {excluded_count} samples from {len(self.exclude_categories)} categories: "
                      f"{sorted(self.exclude_categories)}")
                print(f"    {before_count} -> {len(self.object_samples)} samples")

        # Whitelist specific categories if include_categories is set
        if self.include_categories and self.object_samples:
            before_count = len(self.object_samples)
            self.object_samples = [s for s in self.object_samples
                                   if s['label'] in self.include_categories]
            included_count = len(self.object_samples)
            if included_count < before_count and _is_main_process():
                print(f"  Whitelist: kept {included_count}/{before_count} samples from {len(self.include_categories)} categories: "
                      f"{sorted(self.include_categories)}")

        # Filter out rare categories if min_category_samples > 1
        if self.min_category_samples > 1 and self.object_samples:
            category_counts = Counter(s['label'] for s in self.object_samples)
            rare_categories = {cat for cat, count in category_counts.items()
                              if count < self.min_category_samples}
            if rare_categories:
                before_count = len(self.object_samples)
                self.object_samples = [s for s in self.object_samples
                                       if s['label'] not in rare_categories]
                if _is_main_process():
                    print(f"  Filtered {len(rare_categories)} rare categories "
                          f"(< {self.min_category_samples} samples): {before_count} -> {len(self.object_samples)} samples")
                    examples = sorted(rare_categories)[:5]
                    print(f"    Examples: {examples}")

        # LRU cache for mesh data (vertices + object IDs) - much smaller than full rasterizers
        self._mesh_cache = OrderedDict()
        self._mesh_cache_max = 20  # ~50MB per scene max

    def _get_rasterizer(self, scene_id: str, scene_path: Path) -> SceneRasterizer:
        """Get or create rasterizer for a scene (LRU cached to limit memory)."""
        MAX_CACHED_RASTERIZERS = 10  # Limit to ~500MB RAM for meshes

        if scene_id in self.rasterizers:
            # Move to end (most recently used)
            self.rasterizers.move_to_end(scene_id)
        else:
            # Evict oldest if at capacity
            while len(self.rasterizers) >= MAX_CACHED_RASTERIZERS:
                oldest_id, oldest_raster = self.rasterizers.popitem(last=False)
                # Free memory
                del oldest_raster

            self.rasterizers[scene_id] = SceneRasterizer(
                scene_path,
                cache_dir=self.raster_cache_dir,
                use_undistorted=self.use_undistorted
            )
        return self.rasterizers[scene_id]

    def _get_object_centroid_lightweight(self, scene_id: str, scene_path: Path, target_obj_id: int) -> Optional[np.ndarray]:
        """
        Get 3D centroid from mesh vertex median (NOT OBB centroid).

        OBB centroid is the geometric center of the bounding box, which for
        large planar objects (walls, floors) can be meters away from the
        visible surface. Using mesh vertex median matches what triangulation
        computes from predicted masks.
        """
        # Check cache first
        if scene_id in self._mesh_cache:
            self._mesh_cache.move_to_end(scene_id)
            vertices, vertex_obj_ids = self._mesh_cache[scene_id]
        else:
            # Load mesh vertices and object mapping
            mesh_path = scene_path / "scans" / "mesh_aligned_0.05.ply"
            if not mesh_path.exists() or not HAS_TRIMESH:
                return None

            try:
                # Load mesh (just vertices, not full rasterization data)
                mesh = trimesh.load(str(mesh_path), process=False)
                vertices = np.array(mesh.vertices, dtype=np.float32)

                # Load vertex-to-object mapping
                vertex_obj_ids, _ = load_vertex_object_ids(scene_path)

                # Evict oldest if at capacity
                while len(self._mesh_cache) >= self._mesh_cache_max:
                    self._mesh_cache.popitem(last=False)

                self._mesh_cache[scene_id] = (vertices, vertex_obj_ids)
            except Exception as e:
                print(f"Warning: Failed to load mesh for {scene_id}: {e}")
                return None

        # Compute centroid using vertex median (matches triangulation output)
        return get_object_centroid_3d(vertices, vertex_obj_ids, target_obj_id)

    def _build_spatial_context(
        self,
        target_obj_id: int,
        target_label: str,
        pix_obj_ids: np.ndarray,
        depth: np.ndarray,
        obj_to_label: Dict[int, str],
        min_coverage: float = 0.001,
        max_other_objects: int = 15
    ) -> Optional['SpatialContext']:
        """Build spatial context for GT-aware spatial augmentation.

        Computes spatial relationships between the target object and:
        1. Other instances of the same label (for "nearest chair", "leftmost chair")
        2. Nearby objects of different labels (for "chair next to table")

        Args:
            target_obj_id: Object ID of the target
            target_label: Label of the target
            pix_obj_ids: Per-pixel object IDs [H, W]
            depth: Depth map [H, W]
            obj_to_label: Dict mapping obj_id -> label
            min_coverage: Min fraction of image for an object to be considered
            max_other_objects: Max other objects to include

        Returns:
            SpatialContext or None if spatial context building is not available
        """
        if not HAS_SPATIAL_CONTEXT:
            return None

        H, W = pix_obj_ids.shape
        total_pixels = H * W

        # Get target mask
        target_mask = (pix_obj_ids == target_obj_id).astype(np.float32)
        if target_mask.sum() == 0:
            return None

        # Find all visible objects with sufficient coverage
        unique_ids, counts = np.unique(pix_obj_ids, return_counts=True)
        scene_obj_masks = {}

        for obj_id, count in zip(unique_ids, counts):
            if obj_id <= 0:  # Skip background
                continue
            coverage = count / total_pixels
            if coverage < min_coverage:
                continue
            if obj_id not in obj_to_label:
                continue
            # Create mask for this object
            scene_obj_masks[obj_id] = (pix_obj_ids == obj_id).astype(np.float32)

        # Convert depth to numpy if needed
        if isinstance(depth, torch.Tensor):
            depth_np = depth.squeeze().cpu().numpy()
        else:
            depth_np = depth.squeeze() if hasattr(depth, 'squeeze') else depth

        # Build spatial context using the imported function
        return build_spatial_context(
            target_mask=target_mask,
            target_obj_id=target_obj_id,
            target_label=target_label,
            depth=depth_np,
            scene_obj_masks=scene_obj_masks,
            obj_to_label=obj_to_label,
            max_nearby_objects=max_other_objects
        )

    def __len__(self):
        if self.scene_grouped:
            # Multiply by samples_per_scene so batch_size>1 works with DDP
            # Each sample picks different random views + objects from the same scene
            return len(self.scenes) * self.samples_per_scene
        if self.enumerate_all_objects and self.object_samples:
            return len(self.object_samples)
        return len(self.scenes) * self.samples_per_scene

    def _estimate_view_overlap(self, pos1: np.ndarray, dir1: np.ndarray,
                                pos2: np.ndarray, dir2: np.ndarray,
                                baseline_scale: float = 1.5) -> float:
        """Estimate visual overlap between two camera views.

        Heuristic based on:
        1. Camera baseline distance (closer = more overlap)
        2. Viewing direction similarity (similar directions = more overlap)

        Args:
            pos1, dir1: Camera 1 position and viewing direction (unit vector)
            pos2, dir2: Camera 2 position and viewing direction (unit vector)
            baseline_scale: Distance at which overlap decays to ~37% (meters)

        Returns:
            Overlap score in [0, 1]. Higher = more expected visual overlap.
        """
        # Distance between cameras
        baseline = np.linalg.norm(pos1 - pos2)

        # Angle between viewing directions (cosine similarity)
        cos_angle = np.clip(np.dot(dir1, dir2), -1, 1)

        # Distance score: exponential decay (closer cameras = higher overlap)
        dist_score = np.exp(-baseline / baseline_scale)

        # Angle score: cameras pointing same direction have more overlap
        # cos_angle ranges from -1 (opposite) to 1 (same direction)
        # Map to [0, 1]: opposite = 0, perpendicular = 0.5, same = 1
        angle_score = (cos_angle + 1) / 2

        # For good overlap, both cameras should be close AND looking similarly
        # Use geometric mean to require both conditions
        return np.sqrt(dist_score * angle_score)

    def _get_chunk_groups(self, scene_id: str, images: List[str]) -> Dict[int, List[str]]:
        """Group images by their DA3 processing chunk using pre-built chunk map.

        DA3-NESTED processes images in chunks of 16. Images within a chunk share
        consistent world coordinates. Uses self._scene_chunk_map built during __init__
        to avoid loading .pt files at runtime (which caused DDP deadlocks).

        Args:
            scene_id: Scene identifier
            images: List of candidate image names

        Returns:
            Dict mapping chunk_idx -> list of image names in that chunk.
            Empty dict if cache not available.
        """
        if not self.use_cached_depth or not self._scene_chunk_map:
            return {}

        stem_to_chunk = self._scene_chunk_map.get(scene_id)
        if not stem_to_chunk:
            return {}

        chunk_groups = {}  # chunk_idx -> [image names]
        for img_name in images:
            stem = Path(img_name).stem
            chunk_idx = stem_to_chunk.get(stem)
            if chunk_idx is not None:
                if chunk_idx not in chunk_groups:
                    chunk_groups[chunk_idx] = []
                chunk_groups[chunk_idx].append(img_name)

        return chunk_groups

    def _select_chunk_group(self, chunk_groups: Dict, n_views: int,
                            prefer_overlap: bool = False, transforms: dict = None) -> List[str]:
        """Select a chunk group with enough views for sampling.

        Args:
            chunk_groups: Dict from _get_chunk_groups
            n_views: Minimum number of views needed
            prefer_overlap: If True, prefer chunks with high visual overlap
            transforms: Scene transforms (used if prefer_overlap=True)

        Returns:
            List of images from the selected chunk group.
        """
        if not chunk_groups:
            return []

        # Filter to chunks with enough views
        valid_chunks = [(k, v) for k, v in chunk_groups.items() if len(v) >= n_views]

        if not valid_chunks:
            # No single chunk has enough - find the largest chunk
            largest = max(chunk_groups.items(), key=lambda x: len(x[1]))
            return largest[1]

        if prefer_overlap and transforms:
            # Score chunks by average pairwise overlap
            frame_lookup = {}
            for frame in transforms.get('frames', []):
                fname = Path(frame.get('file_path', '')).name
                frame_lookup[fname] = frame

            best_chunk = None
            best_score = -1

            for key, imgs in valid_chunks:
                # Compute average overlap within chunk
                positions = []
                for img in imgs:
                    if img in frame_lookup:
                        c2w = np.array(frame_lookup[img]['transform_matrix'], dtype=np.float64)
                        pos = c2w[:3, 3]
                        view_dir = -c2w[:3, 2]
                        norm = np.linalg.norm(view_dir)
                        if norm > 1e-6:
                            view_dir = view_dir / norm
                        positions.append((pos, view_dir))

                if len(positions) >= 2:
                    total_overlap = 0
                    count = 0
                    for i in range(len(positions)):
                        for j in range(i + 1, len(positions)):
                            overlap = self._estimate_view_overlap(
                                positions[i][0], positions[i][1],
                                positions[j][0], positions[j][1]
                            )
                            total_overlap += overlap
                            count += 1
                    avg_overlap = total_overlap / count if count > 0 else 0

                    if avg_overlap > best_score:
                        best_score = avg_overlap
                        best_chunk = imgs

            if best_chunk:
                return best_chunk

        # Random selection among valid chunks
        return random.choice(valid_chunks)[1]

    def _sample_views_overlap(self, images: List[str], n_views: int,
                               transforms: dict, min_overlap: float = 0.3) -> List[str]:
        """Sample views ensuring sufficient pairwise visual overlap.

        Uses greedy selection: starts with a random seed view, then iteratively
        adds views that have maximum overlap with already selected views.
        This ensures views share visual content, which is required for sheaf
        loss to be meaningful.

        Args:
            images: List of candidate image names
            n_views: Number of views to select
            transforms: Scene transforms dict with 'frames' containing camera poses
            min_overlap: Minimum overlap threshold (0.3 = 30%)

        Returns:
            List of selected image names with sufficient overlap.
        """
        if not transforms or len(images) <= n_views:
            return random.sample(images, min(n_views, len(images)))

        # Build frame lookup
        frame_lookup = {}
        for frame in transforms.get('frames', []):
            fname = Path(frame.get('file_path', '')).name
            frame_lookup[fname] = frame

        # Extract camera positions and viewing directions
        camera_info = {}
        for img in images:
            if img in frame_lookup:
                c2w = np.array(frame_lookup[img]['transform_matrix'], dtype=np.float64)
                pos = c2w[:3, 3]  # Translation = camera position
                # Viewing direction is -Z axis in camera frame, transformed to world
                view_dir = -c2w[:3, 2]
                norm = np.linalg.norm(view_dir)
                if norm > 1e-6:
                    view_dir = view_dir / norm
                camera_info[img] = (pos, view_dir)

        # Filter to images with valid camera info
        available = [img for img in images if img in camera_info]
        if len(available) < n_views:
            # Not enough images with pose info - fall back to random
            return random.sample(images, min(n_views, len(images)))

        # Greedy selection starting from random seed
        selected = [random.choice(available)]
        remaining = set(available) - set(selected)

        while len(selected) < n_views and remaining:
            best_img = None
            best_score = -1

            for img in remaining:
                pos, view_dir = camera_info[img]
                # Compute max overlap with any already-selected view
                max_overlap = 0
                for sel_img in selected:
                    sel_pos, sel_dir = camera_info[sel_img]
                    overlap = self._estimate_view_overlap(pos, view_dir, sel_pos, sel_dir)
                    max_overlap = max(max_overlap, overlap)

                # Prefer views that meet overlap threshold and maximize it
                if max_overlap >= min_overlap and max_overlap > best_score:
                    best_score = max_overlap
                    best_img = img

            if best_img is None:
                # No view meets threshold - pick the one with maximum overlap anyway
                for img in remaining:
                    pos, view_dir = camera_info[img]
                    max_overlap = 0
                    for sel_img in selected:
                        sel_pos, sel_dir = camera_info[sel_img]
                        overlap = self._estimate_view_overlap(pos, view_dir, sel_pos, sel_dir)
                        max_overlap = max(max_overlap, overlap)
                    if max_overlap > best_score:
                        best_score = max_overlap
                        best_img = img

            if best_img:
                selected.append(best_img)
                remaining.remove(best_img)
            else:
                break

        # Fill remaining slots with random if needed
        if len(selected) < n_views:
            remaining_pool = [img for img in images if img not in selected]
            needed = n_views - len(selected)
            selected.extend(random.sample(remaining_pool, min(needed, len(remaining_pool))))

        return selected

    def _sample_views(self, images: List[str], n_views: int, transforms: dict = None,
                       scene_id: str = None) -> List[str]:
        """Sample views using specified strategy.

        Only chunk_aware and overlap strategies restrict to a single DA3 chunk.
        stratified, random, and sequential sample from ALL candidate images
        (the original behavior), which gives better scene coverage.

        Strategies:
        - 'random': Random views from all candidates
        - 'stratified': Evenly spaced views from all candidates (full scene coverage)
        - 'sequential': Consecutive views from all candidates
        - 'chunk_aware': Random views from a single DA3 chunk (consistent world coords)
        - 'overlap_30': High-overlap views from a single DA3 chunk (for sheaf loss)
        - 'overlap_50': Very high-overlap views from a single DA3 chunk

        Args:
            images: List of candidate image names
            n_views: Number of views to select
            transforms: Scene transforms dict (required for overlap strategies)
            scene_id: Scene identifier (required for chunk-aware sampling with DA3 cache)

        Returns:
            List of selected image names.
        """
        if len(images) <= n_views:
            return images

        # Only apply chunk filtering for strategies that need consistent world coordinates
        needs_chunk = self.sampling_strategy in ('chunk_aware', 'overlap_30', 'overlap_50')

        if needs_chunk:
            chunk_groups = {}
            if scene_id and self.use_cached_depth:
                chunk_groups = self._get_chunk_groups(scene_id, images)

            needs_overlap = self.sampling_strategy in ('overlap_30', 'overlap_50')

            if chunk_groups:
                chunk_images = self._select_chunk_group(
                    chunk_groups, n_views,
                    prefer_overlap=needs_overlap,
                    transforms=transforms
                )
                if len(chunk_images) >= n_views:
                    images = chunk_images
                else:
                    if not hasattr(self, '_chunk_warning_printed'):
                        print(f"[Sampling] Warning: No DA3 chunk has {n_views} views. "
                              f"Using largest chunk ({len(chunk_images)} views). "
                              f"Max views per chunk is 16.")
                        self._chunk_warning_printed = True
                    images = chunk_images

            if len(images) <= n_views:
                return images

            if self.sampling_strategy == 'overlap_30':
                return self._sample_views_overlap(images, n_views, transforms, min_overlap=0.3)
            elif self.sampling_strategy == 'overlap_50':
                return self._sample_views_overlap(images, n_views, transforms, min_overlap=0.5)
            else:  # chunk_aware
                return random.sample(images, n_views)

        # Non-chunk strategies: sample from ALL candidate images
        if self.sampling_strategy == 'random':
            return random.sample(images, n_views)
        elif self.sampling_strategy == 'stratified':
            # Evenly spaced views across the full scene
            step = len(images) / n_views
            indices = [int(i * step) for i in range(n_views)]
            return [images[i] for i in indices]
        else:  # sequential
            if len(images) == n_views:
                return images
            start = random.randint(0, len(images) - n_views)
            return images[start:start + n_views]

    def _sample_covisible_objects(self, scene, all_obj_ids, exclude_obj_id, exclude_label, num_extra,
                                   deduplicate_labels=True):
        """Sample additional objects visible in the selected views for multi-object training.

        Args:
            scene: Scene dict from self.scenes
            all_obj_ids: List of per-view pixel obj_id arrays (already loaded)
            exclude_obj_id: Object ID of the primary object (exclude from sampling)
            exclude_label: Label of the primary object
            num_extra: Number of additional objects to sample (0 = unlimited/all)
            deduplicate_labels: If True, only one object per semantic label (default).
                If False, include multiple instances of the same label (SAM3-style).

        Returns:
            List of (obj_id, label) tuples (may be shorter than num_extra if scene has fewer objects)
        """
        if num_extra == 0:
            return []

        obj_to_label = scene.get('obj_to_label', {})
        label_to_obj_ids = scene.get('label_to_obj_ids', {})

        skip_labels = {
            'remove', 'split', 'object', 'objects', 'stuff', 'unknown',
            'wall', 'floor', 'ceiling', 'door', 'window', 'doorframe', 'window frame',
            'reflection', 'mirror', 'structure',
        }

        # Find all objects visible across the selected views
        visible_obj_ids = set()
        for pix_obj_ids in all_obj_ids:
            if pix_obj_ids is not None:
                unique_ids = np.unique(pix_obj_ids)
                for oid in unique_ids:
                    oid = int(oid)
                    if oid > 0:
                        visible_obj_ids.add(oid)

        # Filter candidates: different from primary, not in skip_labels, has valid label
        used_labels = {exclude_label} if (exclude_label and deduplicate_labels) else set()
        candidates = []
        for oid in visible_obj_ids:
            if oid == exclude_obj_id:
                continue
            label = obj_to_label.get(oid)
            if label is None:
                continue
            label = normalize_label(label)
            if label.lower() in skip_labels:
                continue
            if label in self.exclude_categories:
                continue
            if self.include_categories and label not in self.include_categories:
                continue
            # Deduplicate by label when semantic_union is on (same label = same mask)
            if deduplicate_labels and label in used_labels:
                continue
            # Check minimum visibility (at least 1 view with > min_object_pixels coverage)
            has_coverage = False
            for pix_obj_ids in all_obj_ids:
                if pix_obj_ids is not None:
                    coverage = (pix_obj_ids == oid).sum() / pix_obj_ids.size
                    if coverage >= self.min_object_pixels:
                        has_coverage = True
                        break
            if has_coverage:
                candidates.append((oid, label))
                if deduplicate_labels:
                    used_labels.add(label)

        # Shuffle and return (unlimited if num_extra < 0)
        random.shuffle(candidates)
        if num_extra < 0:
            return candidates
        return candidates[:num_extra]

    def __getitem__(self, idx):
        # Handle enumerate_all_objects mode
        forced_obj_id = None
        forced_label = None
        forced_visible_images = None  # For smart view sampling

        if self.scene_grouped:
            # Scene-grouped mode: iterate per-scene, discover all objects dynamically
            scene_idx = idx % len(self.scenes)
        elif self.enumerate_all_objects and self.object_samples:
            # Use pre-computed object sample
            obj_sample = self.object_samples[idx]
            scene_idx = obj_sample['scene_idx']
            forced_obj_id = obj_sample['obj_id']
            forced_label = obj_sample['label']
            forced_visible_images = obj_sample.get('visible_images')  # SMART SAMPLING
        else:
            # Map idx to scene (multiple samples per scene)
            scene_idx = idx % len(self.scenes)
            sample_idx = idx // len(self.scenes)  # Which sample from this scene

        scene = self.scenes[scene_idx]
        scene_id = scene['scene_id']
        scene_path = scene['scene_path']
        images_dir = scene['images_dir']

        # SMART VIEW SAMPLING: Only sample from images where the object is visible
        if forced_visible_images is not None:
            candidate_images = forced_visible_images
        else:
            candidate_images = scene['train_images']

        selected_images = self._sample_views(
            candidate_images, self.views_per_sample, transforms=scene['transforms'],
            scene_id=scene_id
        )

        images = []
        intrinsics_list = []
        extrinsics_list = []
        image_names = []

        # Get intrinsics from transforms (shared across all images)
        transforms = scene['transforms']
        if transforms:
            fx = transforms.get('fl_x', 500)
            fy = transforms.get('fl_y', 500)
            cx = transforms.get('cx', 256)
            cy = transforms.get('cy', 256)
            shared_intrinsics = torch.tensor([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=torch.float32)

            # Build frame lookup
            frame_lookup = {}
            for frame in transforms.get('frames', []):
                fname = Path(frame.get('file_path', '')).name
                frame_lookup[fname] = frame
        else:
            shared_intrinsics = torch.eye(3)
            frame_lookup = {}

        for img_name in selected_images:
            img_path = images_dir / img_name

            # Load image (always needed for now, but could skip if all caches available)
            image = Image.open(img_path).convert('RGB')
            image = image.resize(self.image_size, Image.BILINEAR)
            image = torch.from_numpy(np.array(image)).float() / 255.0
            image = image.permute(2, 0, 1)
            images.append(image)

            # Get extrinsics
            if img_name in frame_lookup:
                extrinsics = torch.tensor(
                    frame_lookup[img_name]['transform_matrix'],
                    dtype=torch.float32
                )
            else:
                extrinsics = torch.eye(4)

            intrinsics_list.append(shared_intrinsics)
            extrinsics_list.append(extrinsics)
            image_names.append(img_name)

        # For supervised mode: get GT masks from pre-computed obj_ids
        gt_masks = None
        target_obj_id = None
        prompt = None

        if self.supervised:
            # Load pre-computed obj_ids for all views
            all_obj_ids = []
            for img_name in selected_images:
                obj_ids_path = self.obj_ids_root / scene_id / f"{img_name}.pth"
                if obj_ids_path.exists():
                    pix_obj_ids = torch.load(obj_ids_path, weights_only=False)
                    all_obj_ids.append(pix_obj_ids)
                else:
                    all_obj_ids.append(None)

            # Get visible objects from views (first image for per-object, any for scene-grouped)
            has_any_obj_ids = any(x is not None for x in all_obj_ids) if self.scene_grouped else (all_obj_ids[0] is not None)
            if has_any_obj_ids:
                # OPTIMIZATION: Skip expensive np.unique if we already have forced_obj_id
                if forced_obj_id is not None:
                    target_obj_id = forced_obj_id
                    prompt = forced_label
                else:
                    # Only compute visible objects when needed (random object selection)
                    obj_to_label = scene.get('obj_to_label', {})
                    skip_labels = {
                        'remove', 'split', 'object', 'objects', 'stuff', 'unknown',
                        'wall', 'floor', 'ceiling', 'door', 'window', 'doorframe', 'window frame',
                        'reflection', 'mirror', 'structure',
                    }

                    # Find visible objects across ALL views (not just first) for scene-grouped mode
                    scan_views = all_obj_ids if self.scene_grouped else [all_obj_ids[0]]
                    visible_objects = []
                    seen_obj_ids = set()
                    for pix_obj_ids in scan_views:
                        if pix_obj_ids is None:
                            continue
                        H, W = pix_obj_ids.shape
                        total_pixels = H * W
                        unique_ids, counts = np.unique(pix_obj_ids, return_counts=True)
                        for obj_id, count in zip(unique_ids, counts):
                            obj_id = int(obj_id)
                            if obj_id <= 0 or obj_id in seen_obj_ids:
                                continue
                            fraction = count / total_pixels
                            if fraction >= self.min_object_pixels:
                                label = obj_to_label.get(obj_id)
                                if label and label.lower() not in skip_labels:
                                    if not self.exclude_categories or label not in self.exclude_categories:
                                        if not self.include_categories or label in self.include_categories:
                                            visible_objects.append((obj_id, label))
                                            seen_obj_ids.add(obj_id)

                    if visible_objects:
                        target_obj_id, prompt = random.choice(visible_objects)

                if target_obj_id is not None:
                    # Build list of all target objects: primary + additional for multi-object training
                    all_target_objects = [(target_obj_id, prompt)]
                    if self.num_objects_per_sample != 1:
                        # 0 = dynamic (ALL visible objects), >1 = fixed K
                        if self.num_objects_per_sample == 0:
                            num_extra = -1  # unlimited
                        else:
                            num_extra = self.num_objects_per_sample - 1
                        # Deduplicate labels when semantic_union is on (same label = same mask)
                        extra_objects = self._sample_covisible_objects(
                            scene, all_obj_ids, target_obj_id, prompt,
                            num_extra=num_extra,
                            deduplicate_labels=self.semantic_union,
                        )
                        all_target_objects.extend(extra_objects)

                    # Generate GT masks for each object across all views
                    all_object_gt_masks = []  # List of [N, H, W] tensors, one per object
                    all_object_gt_coverages = []  # List of [N] coverage arrays
                    all_object_prompts = []
                    label_to_obj_ids_map = scene.get('label_to_obj_ids', {})

                    for obj_id, obj_label in all_target_objects:
                        # Get all obj_ids with the same semantic label (for semantic union)
                        if self.semantic_union and obj_label is not None:
                            matching_obj_ids = label_to_obj_ids_map.get(obj_label, [obj_id])
                        else:
                            matching_obj_ids = [obj_id]

                        obj_gt_masks = []
                        obj_gt_coverages = []
                        for pix_obj_ids in all_obj_ids:
                            if pix_obj_ids is not None:
                                # SEMANTIC UNION: mask = union of ALL instances with same label
                                mask = np.zeros_like(pix_obj_ids, dtype=np.float32)
                                for oid in matching_obj_ids:
                                    mask += (pix_obj_ids == oid).astype(np.float32)
                                mask = (mask > 0).astype(np.float32)  # Clamp to binary
                                orig_coverage = mask.sum() / mask.size
                                obj_gt_coverages.append(orig_coverage)
                                # Resize to mask_size
                                mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
                                mask_pil = mask_pil.resize(
                                    (self.mask_size[1], self.mask_size[0]), Image.NEAREST
                                )
                                mask = np.array(mask_pil).astype(np.float32) / 255.0
                                obj_gt_masks.append(torch.from_numpy(mask))
                            else:
                                obj_gt_masks.append(torch.zeros(self.mask_size))
                                obj_gt_coverages.append(0.0)

                        all_object_gt_masks.append(torch.stack(obj_gt_masks))  # [N, H, W]
                        all_object_gt_coverages.append(obj_gt_coverages)
                        all_object_prompts.append(obj_label)

                    K_actual = len(all_target_objects)
                    if K_actual == 1 and self.num_objects_per_sample == 1:
                        # Single object mode: keep original shape [N, H, W] for backward compat
                        gt_masks = all_object_gt_masks[0]  # (N, H, W)
                        gt_mask_coverages = all_object_gt_coverages[0]
                    else:
                        # Multi-object: always stack to [K, N, H, W] (even K=1)
                        # This ensures consistent tensor dims for collate padding
                        gt_masks = torch.stack(all_object_gt_masks)  # (K, N, H, W)
                        # Coverage for primary object (used for view validity checks)
                        gt_mask_coverages = all_object_gt_coverages[0]

        # Fallback prompt if not set
        if prompt is None:
            annotations = scene['annotations']
            object_labels = list(annotations.keys())
            if object_labels:
                label = random.choice(object_labels)
                if label in SCANNETPP_PROMPTS:
                    prompt = SCANNETPP_PROMPTS[label][0]
                else:
                    prompt = label
            else:
                prompt = random.choice(['chair', 'table', 'monitor', 'lamp'])

        # Get original image resolution from transforms (for intrinsics scaling)
        orig_h = transforms.get('h', 1168) if transforms else 1168
        orig_w = transforms.get('w', 1752) if transforms else 1752

        result = {
            'images': torch.stack(images),              # (N, 3, H, W)
            'intrinsics': torch.stack(intrinsics_list), # (N, 3, 3) - at orig_hw resolution
            'extrinsics': torch.stack(extrinsics_list), # (N, 4, 4)
            'orig_hw': (orig_h, orig_w),                # Original resolution for intrinsics
            'scene_id': scene_id,
            'prompt': prompt,
            'image_names': image_names,
            'has_metric_scale': True,
            'has_gt_mask': gt_masks is not None
        }

        if gt_masks is not None:
            result['gt_masks'] = gt_masks  # (N, H, W) if K=1, (K, N, H, W) if K>1
            result['target_obj_id'] = target_obj_id
            result['gt_mask_coverage'] = torch.tensor(gt_mask_coverages, dtype=torch.float32)  # (N,) coverage at original res
            # Multi-object fields (only when K > 1)
            if gt_masks.dim() == 4:  # [K, N, H, W]
                result['num_objects'] = gt_masks.shape[0]
                result['multi_object_prompts'] = all_object_prompts  # List[str] length K

            # Get GT 3D centroid - try cache first, fallback to trimesh
            centroid_3d = None
            if scene_id in self.centroid_cache and str(target_obj_id) in self.centroid_cache[scene_id]:
                # Use cached centroid (fast path - no trimesh needed)
                centroid_3d = self.centroid_cache[scene_id][str(target_obj_id)]
                result['centroid_3d'] = torch.tensor(centroid_3d, dtype=torch.float32)  # (3,) meters
            else:
                # Fallback to computing from mesh (slow path)
                centroid_3d = self._get_object_centroid_lightweight(scene_id, scene_path, target_obj_id)
                if centroid_3d is not None:
                    result['centroid_3d'] = torch.from_numpy(centroid_3d).float()  # (3,) meters

        # Load cached DA3 depth if available (2-4x training speedup)
        # Supports both formats:
        #   - da3_cache: depth is [1, 1, H, W] (old format from preprocess_da3.py)
        #   - da3_nested_cache: depth is [H, W], includes extrinsics/intrinsics (from preprocess_da3_nested.py)
        if self.use_cached_depth and self.da3_cache_dir is not None:
            cached_depths = []
            cached_extrinsics = []
            cached_intrinsics = []
            all_cached = True
            has_poses = True  # Track if cache includes pose info

            for img_name in image_names:
                cache_path = self.da3_cache_dir / scene_id / f"{Path(img_name).stem}.pt"
                if cache_path.exists():
                    try:
                        # NOTE: mmap=True can cause DDP hangs when multiple ranks access same file
                        # Disable for debugging - may use more RAM but avoids file lock contention
                        cache_data = torch.load(cache_path, map_location='cpu', weights_only=True, mmap=False)
                        depth = cache_data['depth'].float()  # Convert from fp16

                        # Handle both formats: [1, 1, H, W] (old) or [H, W] (new)
                        if depth.dim() == 4:
                            depth = depth.squeeze(0)  # [1, 1, H, W] -> [1, H, W]
                        elif depth.dim() == 2:
                            depth = depth.unsqueeze(0)  # [H, W] -> [1, H, W]
                        # depth.dim() == 3 means [1, H, W] already

                        cached_depths.append(depth)  # [1, H, W]

                        # Load poses if available (da3_nested_cache format)
                        if 'extrinsics' in cache_data:
                            cached_extrinsics.append(cache_data['extrinsics'].float())  # [4, 4]
                        else:
                            has_poses = False
                        if 'intrinsics' in cache_data:
                            cached_intrinsics.append(cache_data['intrinsics'].float())  # [3, 3]

                    except Exception as e:
                        if not hasattr(self, '_cache_warning_printed'):
                            print(f"[DA3 Cache] Error loading {cache_path}: {e}")
                            self._cache_warning_printed = True
                        all_cached = False
                        break
                else:
                    if not hasattr(self, '_cache_warning_printed'):
                        print(f"[DA3 Cache] Missing: {cache_path}")
                        print(f"  scene_id={scene_id}, img_name={img_name}")
                        self._cache_warning_printed = True
                    all_cached = False
                    break

            if all_cached and len(cached_depths) == len(image_names):
                cached_depth_tensor = torch.stack(cached_depths)  # (N, 1, H, W)

                # Resize cached depth to target resolution if needed
                # Cache may be at different resolution (e.g., 336×504) than target (e.g., 504×504)
                cache_h, cache_w = cached_depth_tensor.shape[-2:]
                target_h, target_w = self.image_size

                # Track scale factors for intrinsics adjustment
                scale_h, scale_w = 1.0, 1.0

                if cache_h != target_h or cache_w != target_w:
                    # Compute scale factors BEFORE resize
                    scale_h = target_h / cache_h
                    scale_w = target_w / cache_w

                    # Use bilinear interpolation for depth resize
                    cached_depth_tensor = F.interpolate(
                        cached_depth_tensor,  # (N, 1, H, W)
                        size=(target_h, target_w),
                        mode='bilinear',
                        align_corners=False
                    )
                    # Note: depth values are preserved, just spatially resampled

                result['cached_depth'] = cached_depth_tensor  # (N, 1, target_H, target_W)

                # Include cached poses if available (from da3_nested_cache)
                if has_poses and len(cached_extrinsics) == len(image_names):
                    result['cached_da3_extrinsics'] = torch.stack(cached_extrinsics)  # (N, 4, 4)
                if len(cached_intrinsics) == len(image_names):
                    intrinsics_tensor = torch.stack(cached_intrinsics)  # (N, 3, 3)

                    # Scale intrinsics to match resized depth
                    # Intrinsics were calibrated for cache resolution (e.g., 336×504)
                    # After resize to target (e.g., 504×504), must scale fx, fy, cx, cy
                    if scale_h != 1.0 or scale_w != 1.0:
                        orig_fy = intrinsics_tensor[0, 1, 1].item()
                        orig_cy = intrinsics_tensor[0, 1, 2].item()
                        intrinsics_tensor = intrinsics_tensor.clone()
                        intrinsics_tensor[:, 0, 0] *= scale_w  # fx scales with width
                        intrinsics_tensor[:, 1, 1] *= scale_h  # fy scales with height
                        intrinsics_tensor[:, 0, 2] *= scale_w  # cx scales with width
                        intrinsics_tensor[:, 1, 2] *= scale_h  # cy scales with height
                        # Log once per dataset
                        if not hasattr(self, '_intrinsics_scale_logged'):
                            self._intrinsics_scale_logged = True
                            print(f"[Dataloader] Intrinsics scaled for depth resize: "
                                  f"{cache_h}×{cache_w} → {target_h}×{target_w}, "
                                  f"scale_h={scale_h:.2f}, scale_w={scale_w:.2f}")
                            print(f"  fy: {orig_fy:.1f} → {intrinsics_tensor[0, 1, 1].item():.1f}, "
                                  f"cy: {orig_cy:.1f} → {intrinsics_tensor[0, 1, 2].item():.1f}")

                    result['cached_da3_intrinsics'] = intrinsics_tensor  # (N, 3, 3)

                # Build spatial context for GT-aware augmentation
                # Requires: cached depth, gt_masks, and obj_to_label mapping
                if (HAS_SPATIAL_CONTEXT and
                    gt_masks is not None and
                    target_obj_id is not None and
                    prompt is not None and
                    self.supervised):
                    try:
                        # Get first view's data for spatial context
                        scene_obj_ids_dir = self.obj_ids_root / scene_id
                        first_img_name = image_names[0]
                        obj_ids_path = scene_obj_ids_dir / f"{first_img_name}.pth"

                        if obj_ids_path.exists():
                            first_view_obj_ids = torch.load(obj_ids_path, weights_only=False)
                            first_view_depth = cached_depth_tensor[0, 0].numpy()  # [H, W]

                            # Resize obj_ids to match depth resolution if needed
                            if first_view_obj_ids.shape != first_view_depth.shape:
                                # Use nearest neighbor for integer IDs
                                obj_ids_pil = Image.fromarray(first_view_obj_ids.astype(np.int32), mode='I')
                                obj_ids_pil = obj_ids_pil.resize(
                                    (first_view_depth.shape[1], first_view_depth.shape[0]),
                                    Image.NEAREST
                                )
                                first_view_obj_ids = np.array(obj_ids_pil)

                            # Build spatial context
                            spatial_context = self._build_spatial_context(
                                target_obj_id=target_obj_id,
                                target_label=prompt,
                                pix_obj_ids=first_view_obj_ids,
                                depth=first_view_depth,
                                obj_to_label=scene.get('obj_to_label', {})
                            )

                            if spatial_context is not None:
                                result['spatial_context'] = spatial_context
                    except Exception as e:
                        # Don't fail the whole sample if spatial context building fails
                        if not hasattr(self, '_spatial_context_warning_printed'):
                            print(f"[Spatial Context] Warning: Failed to build spatial context: {e}")
                            self._spatial_context_warning_printed = True

        return result


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


if __name__ == "__main__":
    # Test the loader
    data_root = Path(__file__).parent.parent.parent / "data" / "scannetpp"

    print("Testing ScanNet++ loader...")

    # Check available scenes
    scenes = get_available_scenes(data_root, split='nvs_sem_train')
    print(f"\nAvailable scenes (train): {len(scenes)}")

    if scenes:
        # Test single-view dataset
        dataset = ScanNetPPDataset(
            data_root,
            split='nvs_sem_train',
            images_per_scene=5,
            max_scenes=3
        )

        print(f"\nSingle-view dataset: {len(dataset)} samples")

        # Test multi-view dataset
        mv_dataset = ScanNetPPMultiViewDataset(
            data_root,
            split='nvs_sem_train',
            views_per_sample=8,
            max_scenes=3
        )

        print(f"Multi-view dataset: {len(mv_dataset)} scenes")

        if len(mv_dataset) > 0:
            sample = mv_dataset[0]
            print(f"\nSample keys: {list(sample.keys())}")
            print(f"Images shape: {sample['images'].shape}")
            print(f"Prompt: {sample['prompt']}")
            print(f"Has metric scale: {sample['has_metric_scale']}")
    else:
        print("\nNo scenes found. Please download ScanNet++ data first.")
        print("Run: python download_scannetpp.py download_focused.yml")
