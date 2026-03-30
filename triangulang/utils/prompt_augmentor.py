"""
Prompt Augmentation for Robust Segmentation Training

Based on SAM2-Video training strategies:
- Dense mask: drop 80%, perturb 20% to simulate errors
- Points: sample from GT mask with jitter
- BBox: from mask bounds with jitter and expansion
- Language: category names with synonyms

Also includes sprinkle removal for cleaner predictions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Union
import random


# Reverse synonyms: Map SPECIFIC labels to their GENERAL form
# This helps when dataset has very specific labels like "pedestal fan" but model knows "fan" better
REVERSE_SYNONYMS = {
    # Fan variants → fan
    'pedestal fan': 'fan',
    'desk fan': 'fan',
    'portable fan': 'fan',
    'table fan': 'fan',
    'ceiling fan': 'fan',
    'exhaust fan': 'fan',

    # Chair variants → chair
    'office chair': 'chair',
    'dining chair': 'chair',
    'lounge chair': 'chair',
    'desk chair': 'chair',
    'high stool': 'stool',

    # Table variants → table
    'office table': 'table',
    'office desk': 'desk',
    'coffee table': 'table',
    'dining table': 'table',

    # Lid/cover variants
    'pot lid': 'lid',
    'container lid': 'lid',

    # Paper variants
    'tissue paper': 'tissue',
    'toilet paper': 'toilet paper',
    'paper towel': 'towel',

    # Computer variants
    'computer tower': 'computer',
    'copmuter tower': 'computer',  # Fix typo in dataset
    'tower pc': 'computer',
    'desktop computer': 'computer',

    # Cabinet variants
    'storage cabinet': 'cabinet',
    'file cabinet': 'cabinet',
    'kitchen cabinet': 'cabinet',
    'office cabinet': 'cabinet',

    # Socket/outlet variants
    'power socket': 'outlet',
    'wall outlet': 'outlet',
    'electrical plug': 'plug',
    'network socket': 'socket',
    'datashow socket': 'socket',

    # Lamp variants
    'ceiling lamp': 'lamp',
    'table lamp': 'lamp',
    'desk lamp': 'lamp',
    'standing lamp': 'lamp',
    'floor lamp': 'lamp',

    # Bin/trash variants
    'trash bin': 'trash can',
    'trash can': 'trash can',
    'dustbin': 'trash can',
    'wastebin': 'trash can',
    'trashcan': 'trash can',

    # Board variants
    'whiteboard': 'board',
    'white board': 'board',
    'bulletin board': 'board',

    # Bag variants
    'paper bag': 'bag',
    'laptop bag': 'bag',
    'tote bag': 'bag',

    # Hanger variants
    'clothes hanger': 'hanger',
    'coat hanger': 'hanger',
    'wall hanger': 'hanger',
    'standing clothes hanger': 'hanger',

    # Monitor variants
    'moitor': 'monitor',  # Fix typo

    # Remote/controller variants
    'remote controller': 'remote',
    'remote control': 'remote',
    'tv remote': 'remote',

    # Closet variants
    'foldable closet': 'closet',
    'wardrobe closet': 'closet',

    # Specific items that CLIP knows better by general name
    'window sill': 'windowsill',
    'window frame': 'window',
    'door frame': 'doorframe',
    'blind rail': 'blinds',
    'ceiling light': 'light',
    'ceiling lights': 'lights',
}

# Category synonyms for language augmentation
# Maps category names to lists of synonyms/paraphrases
CATEGORY_SYNONYMS = {
    # Common objects
    'chair': ['seat', 'seating', 'armchair', 'office chair', 'dining chair'],
    'table': ['desk', 'surface', 'tabletop', 'dining table', 'work table'],
    'sofa': ['couch', 'settee', 'loveseat', 'sectional', 'divan'],
    'bed': ['mattress', 'sleeping surface', 'bedframe', 'cot'],
    'lamp': ['light', 'lighting', 'light fixture', 'floor lamp', 'table lamp'],
    'tv': ['television', 'screen', 'monitor', 'display', 'tv screen'],
    'plant': ['houseplant', 'potted plant', 'greenery', 'foliage', 'indoor plant'],
    'book': ['volume', 'novel', 'textbook', 'reading material'],
    'bottle': ['container', 'flask', 'water bottle', 'beverage container'],
    'cup': ['mug', 'glass', 'drinking vessel', 'coffee cup', 'teacup'],
    'bowl': ['dish', 'container', 'serving bowl', 'mixing bowl'],
    'keyboard': ['computer keyboard', 'typing device', 'input device'],
    'mouse': ['computer mouse', 'pointing device', 'input device'],
    'laptop': ['notebook', 'portable computer', 'laptop computer'],
    'phone': ['telephone', 'mobile phone', 'smartphone', 'cellphone', 'cell phone'],
    'clock': ['timepiece', 'wall clock', 'timer', 'watch'],
    'vase': ['flower vase', 'container', 'decorative vase', 'urn'],
    'pillow': ['cushion', 'throw pillow', 'head rest', 'decorative pillow'],
    'blanket': ['throw', 'cover', 'bedding', 'comforter'],
    'curtain': ['drape', 'window covering', 'window treatment'],
    'blinds': ['window blinds', 'venetian blinds', 'window shades', 'shades', 'roller blinds'],
    'door': ['doorway', 'entrance', 'entry', 'portal'],
    'window': ['glass pane', 'window pane', 'opening'],
    'cabinet': ['cupboard', 'storage', 'shelving', 'wardrobe'],
    'shelf': ['shelving', 'ledge', 'rack', 'bookshelf'],
    'mirror': ['looking glass', 'reflective surface', 'wall mirror'],
    'picture': ['painting', 'artwork', 'photo', 'photograph', 'image', 'wall art'],
    'rug': ['carpet', 'floor covering', 'mat', 'area rug'],
    'refrigerator': ['fridge', 'cooler', 'icebox', 'appliance'],
    'sink': ['basin', 'wash basin', 'washbasin'],
    'toilet': ['commode', 'lavatory', 'bathroom fixture'],
    'bathtub': ['tub', 'bath', 'soaking tub'],
    'shower': ['shower stall', 'shower enclosure', 'bathroom shower'],
    'shower head': ['showerhead', 'overhead shower', 'hand shower', 'shower', 'shower faucet'],
    'shower tap': ['shower faucet', 'shower control', 'shower handle', 'tap', 'faucet'],
    'shower curtain': ['shower curtain rod', 'curtain', 'bathroom curtain'],
    'shower door': ['shower glass', 'shower partition', 'shower screen'],
    'tap': ['faucet', 'water tap', 'spigot', 'valve'],
    'faucet': ['tap', 'water tap', 'spigot', 'fixture'],
    'kitchen tap': ['kitchen faucet', 'sink faucet', 'tap', 'faucet'],
    'bathtub tap': ['tub faucet', 'bath faucet', 'tap', 'faucet'],

    # Furniture
    'dresser': ['chest of drawers', 'bureau', 'wardrobe'],
    'nightstand': ['bedside table', 'night table', 'end table'],
    'bookshelf': ['bookcase', 'shelving unit', 'book rack'],
    'ottoman': ['footstool', 'hassock', 'pouf'],

    # Kitchen items
    'microwave': ['microwave oven', 'appliance'],
    'oven': ['stove', 'range', 'cooker'],
    'toaster': ['toaster oven', 'bread toaster'],
    'blender': ['mixer', 'food processor'],
    'dishwasher': ['dish washer', 'appliance'],
    'stove': ['oven', 'range', 'cooker', 'cooking appliance'],
    'countertop': ['counter', 'kitchen counter', 'work surface'],

    # Electronics
    'speaker': ['audio speaker', 'sound system', 'loudspeaker'],
    'headphones': ['earphones', 'headset', 'earbuds'],
    'camera': ['photo camera', 'digital camera'],
    'printer': ['computer printer', 'inkjet', 'laser printer'],

    # Outdoor/misc
    'car': ['automobile', 'vehicle', 'auto'],
    'bicycle': ['bike', 'cycle', 'two-wheeler'],
    'motorcycle': ['motorbike', 'bike', 'scooter'],
    'bus': ['coach', 'transit bus', 'vehicle'],
    'train': ['railway', 'locomotive', 'rail car'],
    'airplane': ['plane', 'aircraft', 'jet'],
    'boat': ['vessel', 'watercraft', 'ship'],

    # People
    'person': ['human', 'individual', 'figure', 'someone', 'somebody'],
    'man': ['male', 'guy', 'gentleman', 'male person'],
    'woman': ['female', 'lady', 'female person'],
    'child': ['kid', 'young person', 'minor', 'youngster'],

    # Animals
    'dog': ['canine', 'puppy', 'hound', 'pet dog'],
    'cat': ['feline', 'kitten', 'kitty', 'pet cat'],
    'bird': ['avian', 'fowl', 'feathered creature'],
    'horse': ['equine', 'steed', 'pony'],

    # Food
    'apple': ['fruit', 'red apple', 'green apple'],
    'banana': ['fruit', 'yellow banana'],
    'orange': ['citrus', 'citrus fruit'],
    'pizza': ['pie', 'pizza pie', 'flatbread'],
    'sandwich': ['sub', 'hoagie', 'wrap'],
    'cake': ['dessert', 'pastry', 'baked good'],

    # Sports
    'ball': ['sphere', 'sports ball'],
    'bat': ['baseball bat', 'cricket bat'],
    'racket': ['tennis racket', 'badminton racket'],
    'skateboard': ['board', 'skate'],
    'surfboard': ['board', 'surf'],
    'skis': ['skiing equipment', 'snow skis'],
}

# Prompt templates for more natural language
PROMPT_TEMPLATES = [
    "{category}",
]


class PromptAugmentor:
    """
    Augments prompts (masks, points, boxes, language) for robust training.

    Based on SAM2-Video training strategies for handling imperfect prompts.
    """

    def __init__(
        self,
        # Mask augmentation
        mask_drop_ratio: float = 0.8,
        mask_perturb_ratio: float = 0.2,
        mask_perturb_kernel_range: Tuple[int, int] = (3, 7),
        # Point augmentation
        point_jitter_px: int = 5,
        num_points_range: Tuple[int, int] = (1, 5),
        # BBox augmentation
        bbox_jitter_ratio: float = 0.05,
        bbox_expand_ratio: float = 0.1,
        # Language augmentation
        use_synonyms: bool = True,
        use_templates: bool = True,
        synonym_prob: float = 0.3,
        template_prob: float = 0.5,
        # Sprinkle removal
        min_area_ratio: float = 0.001,  # 0.1% of image
        # General
        seed: Optional[int] = None,
    ):
        self.mask_drop_ratio = mask_drop_ratio
        self.mask_perturb_ratio = mask_perturb_ratio
        self.mask_perturb_kernel_range = mask_perturb_kernel_range

        self.point_jitter_px = point_jitter_px
        self.num_points_range = num_points_range

        self.bbox_jitter_ratio = bbox_jitter_ratio
        self.bbox_expand_ratio = bbox_expand_ratio

        self.use_synonyms = use_synonyms
        self.use_templates = use_templates
        self.synonym_prob = synonym_prob
        self.template_prob = template_prob

        self.min_area_ratio = min_area_ratio

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def augment_mask(
        self,
        mask: torch.Tensor,
        drop_ratio: Optional[float] = None,
        perturb_ratio: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Augment dense mask prompt by dropping and perturbing pixels.

        Simulates imperfect mask prompts from:
        - Previous frame predictions (video)
        - User scribbles
        - Coarse annotations

        Args:
            mask: Binary mask [H, W] or [1, H, W] or [B, 1, H, W]
            drop_ratio: Fraction of mask pixels to drop (default: 0.8)
            perturb_ratio: Fraction of mask pixels to perturb (default: 0.2)

        Returns:
            Augmented mask with same shape
        """
        drop_ratio = drop_ratio if drop_ratio is not None else self.mask_drop_ratio
        perturb_ratio = perturb_ratio if perturb_ratio is not None else self.mask_perturb_ratio

        # Handle different input shapes
        squeeze_dims = []
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
            squeeze_dims = [0, 1]
        elif mask.dim() == 3:
            mask = mask.unsqueeze(0)
            squeeze_dims = [0]

        B, C, H, W = mask.shape
        device = mask.device
        augmented = mask.clone().float()

        for b in range(B):
            m = augmented[b, 0]

            # Get foreground pixels
            fg_mask = m > 0.5
            fg_indices = fg_mask.nonzero(as_tuple=False)

            if len(fg_indices) == 0:
                continue

            num_fg = len(fg_indices)

            # 1. Drop pixels (set to 0)
            num_drop = int(num_fg * drop_ratio)
            if num_drop > 0:
                drop_perm = torch.randperm(num_fg, device=device)[:num_drop]
                drop_coords = fg_indices[drop_perm]
                m[drop_coords[:, 0], drop_coords[:, 1]] = 0

            # 2. Perturb remaining pixels (morphological ops)
            # Re-get foreground after dropping
            fg_mask = m > 0.5
            if fg_mask.sum() > 0:
                # Random choice: dilate or erode
                kernel_size = random.randint(*self.mask_perturb_kernel_range)
                if kernel_size % 2 == 0:
                    kernel_size += 1

                perturb_mask = fg_mask.float().unsqueeze(0).unsqueeze(0)

                if random.random() < 0.5:
                    # Dilate (expand mask)
                    perturbed = F.max_pool2d(
                        perturb_mask,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2
                    )
                else:
                    # Erode (shrink mask)
                    perturbed = -F.max_pool2d(
                        -perturb_mask,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2
                    )

                # Blend original and perturbed based on perturb_ratio
                blend_mask = torch.rand_like(m) < perturb_ratio
                m[blend_mask] = perturbed[0, 0, blend_mask]

            augmented[b, 0] = m

        # Restore original shape
        for dim in reversed(squeeze_dims):
            augmented = augmented.squeeze(dim)

        return (augmented > 0.5).float()

    def augment_points(
        self,
        mask: torch.Tensor,
        num_points: Optional[int] = None,
        jitter_px: Optional[int] = None,
        include_negative: bool = False,
        negative_ratio: float = 0.2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample point prompts from ground truth mask with jitter.

        Args:
            mask: Binary mask [H, W] or [1, H, W]
            num_points: Number of points to sample (default: random from range)
            jitter_px: Max jitter in pixels (default: self.point_jitter_px)
            include_negative: Whether to include negative points from background
            negative_ratio: Ratio of negative to positive points

        Returns:
            points: [N, 2] tensor of (x, y) coordinates
            labels: [N] tensor of labels (1=positive, 0=negative)
        """
        jitter_px = jitter_px if jitter_px is not None else self.point_jitter_px

        if num_points is None:
            num_points = random.randint(*self.num_points_range)

        # Handle input shape
        if mask.dim() == 3:
            mask = mask.squeeze(0)

        H, W = mask.shape
        device = mask.device

        # Get foreground pixels
        fg_mask = mask > 0.5
        fg_indices = fg_mask.nonzero(as_tuple=False)  # [N, 2] as (row, col) = (y, x)

        if len(fg_indices) == 0:
            # No foreground, return center point as fallback
            return (
                torch.tensor([[W // 2, H // 2]], device=device, dtype=torch.float),
                torch.tensor([1], device=device, dtype=torch.long)
            )

        # Sample positive points
        num_pos = min(num_points, len(fg_indices))
        sample_idx = torch.randperm(len(fg_indices), device=device)[:num_pos]
        pos_coords = fg_indices[sample_idx]  # [N, 2] as (y, x)

        # Convert to (x, y) and add jitter
        pos_points = pos_coords[:, [1, 0]].float()  # [N, 2] as (x, y)

        if jitter_px > 0:
            jitter = torch.randint(-jitter_px, jitter_px + 1, pos_points.shape, device=device).float()
            pos_points = pos_points + jitter
            # Clamp to valid range
            pos_points[:, 0] = pos_points[:, 0].clamp(0, W - 1)
            pos_points[:, 1] = pos_points[:, 1].clamp(0, H - 1)

        pos_labels = torch.ones(num_pos, device=device, dtype=torch.long)

        if include_negative:
            # Sample negative points from background
            bg_mask = ~fg_mask
            bg_indices = bg_mask.nonzero(as_tuple=False)

            num_neg = max(1, int(num_pos * negative_ratio))

            if len(bg_indices) > 0:
                num_neg = min(num_neg, len(bg_indices))
                sample_idx = torch.randperm(len(bg_indices), device=device)[:num_neg]
                neg_coords = bg_indices[sample_idx]
                neg_points = neg_coords[:, [1, 0]].float()

                if jitter_px > 0:
                    jitter = torch.randint(-jitter_px, jitter_px + 1, neg_points.shape, device=device).float()
                    neg_points = neg_points + jitter
                    neg_points[:, 0] = neg_points[:, 0].clamp(0, W - 1)
                    neg_points[:, 1] = neg_points[:, 1].clamp(0, H - 1)

                neg_labels = torch.zeros(num_neg, device=device, dtype=torch.long)

                pos_points = torch.cat([pos_points, neg_points], dim=0)
                pos_labels = torch.cat([pos_labels, neg_labels], dim=0)

        return pos_points, pos_labels

    def augment_bbox(
        self,
        mask: torch.Tensor,
        jitter_ratio: Optional[float] = None,
        expand_ratio: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Get bounding box from mask with jitter and expansion.

        Args:
            mask: Binary mask [H, W] or [1, H, W]
            jitter_ratio: Max jitter as fraction of box size (default: 0.05)
            expand_ratio: Expansion as fraction of box size (default: 0.1)

        Returns:
            bbox: [4] tensor as (x1, y1, x2, y2)
        """
        jitter_ratio = jitter_ratio if jitter_ratio is not None else self.bbox_jitter_ratio
        expand_ratio = expand_ratio if expand_ratio is not None else self.bbox_expand_ratio

        # Handle input shape
        if mask.dim() == 3:
            mask = mask.squeeze(0)

        H, W = mask.shape
        device = mask.device

        # Get foreground pixels
        fg_mask = mask > 0.5
        fg_indices = fg_mask.nonzero(as_tuple=False)  # [N, 2] as (y, x)

        if len(fg_indices) == 0:
            # No foreground, return full image bbox
            return torch.tensor([0, 0, W, H], device=device, dtype=torch.float)

        # Get bounding box
        y_min, x_min = fg_indices.min(dim=0)[0]
        y_max, x_max = fg_indices.max(dim=0)[0]

        # Box dimensions
        box_w = (x_max - x_min + 1).float()
        box_h = (y_max - y_min + 1).float()

        # Convert to float
        x1, y1, x2, y2 = x_min.float(), y_min.float(), x_max.float(), y_max.float()

        # Add expansion
        if expand_ratio > 0:
            expand_x = box_w * expand_ratio
            expand_y = box_h * expand_ratio
            x1 = x1 - expand_x
            y1 = y1 - expand_y
            x2 = x2 + expand_x
            y2 = y2 + expand_y

        # Add jitter
        if jitter_ratio > 0:
            jitter_x = box_w * jitter_ratio
            jitter_y = box_h * jitter_ratio
            x1 = x1 + (random.random() * 2 - 1) * jitter_x
            y1 = y1 + (random.random() * 2 - 1) * jitter_y
            x2 = x2 + (random.random() * 2 - 1) * jitter_x
            y2 = y2 + (random.random() * 2 - 1) * jitter_y

        # Clamp to valid range
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W - 1, x2)
        y2 = min(H - 1, y2)

        return torch.tensor([x1, y1, x2, y2], device=device, dtype=torch.float)

    def augment_language(
        self,
        category: str,
        use_synonym: Optional[bool] = None,
        use_template: Optional[bool] = None,
        use_reverse_synonym: bool = True,
        reverse_synonym_prob: float = 0.5,
    ) -> str:
        """
        Augment language prompt with synonyms and templates.

        Args:
            category: Category name (e.g., "chair", "table")
            use_synonym: Whether to potentially use a synonym
            use_template: Whether to wrap in a template
            use_reverse_synonym: Whether to simplify specific labels to general form
            reverse_synonym_prob: Probability of using reverse synonym (default: 0.5)

        Returns:
            Augmented language prompt
        """
        use_synonym = use_synonym if use_synonym is not None else self.use_synonyms
        use_template = use_template if use_template is not None else self.use_templates

        result = category.lower().strip()

        # First: Maybe simplify specific label to general form (e.g., "pedestal fan" → "fan")
        # This helps when CLIP/SAM3 knows the general concept better than the specific variant
        if use_reverse_synonym and random.random() < reverse_synonym_prob:
            general_form = REVERSE_SYNONYMS.get(result)
            if general_form:
                result = general_form

        # Then: Maybe replace with forward synonym (e.g., "fan" → "ventilator")
        if use_synonym and random.random() < self.synonym_prob:
            synonyms = CATEGORY_SYNONYMS.get(result, [])
            if synonyms:
                result = random.choice(synonyms)

        # Maybe wrap in template
        if use_template and random.random() < self.template_prob:
            template = random.choice(PROMPT_TEMPLATES)
            result = template.format(category=result)

        return result

    def remove_sprinkles(
        self,
        mask: torch.Tensor,
        min_area_ratio: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Remove small disconnected regions (sprinkles) from prediction.

        Args:
            mask: Predicted mask [H, W] or [1, H, W] or [B, 1, H, W]
            min_area_ratio: Minimum area as fraction of total pixels

        Returns:
            Cleaned mask with small regions removed
        """
        min_area_ratio = min_area_ratio if min_area_ratio is not None else self.min_area_ratio

        # Handle different input shapes
        squeeze_dims = []
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
            squeeze_dims = [0, 1]
        elif mask.dim() == 3:
            mask = mask.unsqueeze(0)
            squeeze_dims = [0]

        B, C, H, W = mask.shape
        total_pixels = H * W
        min_area = int(total_pixels * min_area_ratio)

        cleaned = mask.clone()

        for b in range(B):
            m = cleaned[b, 0]
            binary = (m > 0.5).cpu().numpy().astype(np.uint8)

            # Find connected components
            try:
                import cv2
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

                # Remove small components
                for i in range(1, num_labels):  # Skip background (label 0)
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area < min_area:
                        m[torch.from_numpy(labels == i).to(m.device)] = 0
            except ImportError:
                # Fallback without cv2: just threshold by total area
                if m.sum() < min_area:
                    m.zero_()

            cleaned[b, 0] = m

        # Restore original shape
        for dim in reversed(squeeze_dims):
            cleaned = cleaned.squeeze(dim)

        return cleaned

    def augment_all(
        self,
        mask: torch.Tensor,
        category: Optional[str] = None,
        return_dict: bool = True,
    ) -> Union[Dict, Tuple]:
        """
        Generate all augmented prompt types from a ground truth mask.

        Args:
            mask: Ground truth binary mask
            category: Category name for language prompt
            return_dict: Whether to return as dict or tuple

        Returns:
            Dict or tuple with augmented prompts:
            - mask_prompt: Augmented dense mask
            - points: Point coordinates
            - point_labels: Point labels (1=pos, 0=neg)
            - bbox: Bounding box
            - language: Language prompt (if category provided)
        """
        mask_prompt = self.augment_mask(mask)
        points, point_labels = self.augment_points(mask, include_negative=True)
        bbox = self.augment_bbox(mask)

        if return_dict:
            result = {
                'mask_prompt': mask_prompt,
                'points': points,
                'point_labels': point_labels,
                'bbox': bbox,
            }
            if category is not None:
                result['language'] = self.augment_language(category)
            return result
        else:
            language = self.augment_language(category) if category else None
            return mask_prompt, points, point_labels, bbox, language


def get_category_synonyms(category: str) -> List[str]:
    """Get all synonyms for a category."""
    category = category.lower().strip()
    return CATEGORY_SYNONYMS.get(category, [category])


def add_category_synonyms(category: str, synonyms: List[str]):
    """Add custom synonyms for a category."""
    category = category.lower().strip()
    if category in CATEGORY_SYNONYMS:
        CATEGORY_SYNONYMS[category].extend(synonyms)
    else:
        CATEGORY_SYNONYMS[category] = synonyms
