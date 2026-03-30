"""
uCO3D Dataset Loader for Multi-View Training.

uCO3D (UnCommon Objects in 3D) provides ~170k turntable videos of objects
from ~1000 LVIS categories. Each video has:
- RGB frames (from rgb_video.mp4)
- Segmentation masks (from mask_video.mkv)
- Depth maps (from depth_maps.h5, aligned with VGGSfM)
- Camera poses (from metadata.sqlite)
- 3D Gaussian splats and point clouds

Key differences from ScanNet++:
- Object-centric (single foreground object) vs scene-level (multiple objects)
- Turntable videos (small baseline) vs wide-baseline indoor scenes
- Category name as text prompt (no instance-level naming)

Reference: Liu et al., "UnCommon Objects in 3D" (2025)
https://arxiv.org/abs/2501.07574

Usage:
    from triangulang.data.uco3d_dataset import UCO3DMultiViewDataset

    dataset = UCO3DMultiViewDataset(
        data_root='data/uco3d',
        split='train',
        num_views=8,
        num_sequences=50,  # Sample 50 representative sequences
        frames_per_sequence=50,  # Sample 50 frames uniformly
    )
"""

import os
import sys
import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

# Try to import uCO3D package
try:
    from uco3d import UCO3DDataset as UCO3DDatasetBase, UCO3DFrameDataBuilder
    from uco3d.dataset_utils.utils import get_dataset_root
    HAS_UCO3D = True
except ImportError:
    HAS_UCO3D = False
    print("[uCO3D] Warning: uco3d package not installed. Using standalone loader.")

# Try to import h5py for depth loading
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# Categories to Skip (bad mask quality in uCO3D)
# These categories have annotation issues (e.g., hollow objects with filled centers)
# and should be excluded from training/evaluation.

SKIP_CATEGORIES = {
    'coil',              # Hollow center incorrectly filled
    'table-tennis_table',  # Poor mask quality
    'pickle',            # Invalid masks'
    # 'vat'
}

# Specific sequences to skip (mislabeled or broken in uCO3D)
SKIP_SEQUENCES = {
    '1-25422-49542',  # bow_weapon/train - actually a helmet, not a bow
    '3-17635-8218',   # power_shovel - actually a tea kettle
    '5-92833-30767',  # power_shovel - actually a statue
    # knocker_on_a_door - not actually door knockers
    '1-6744-21685',
    '1-68750-93047',
    '1-6349-53277',
    '1-69528-19789',
    '1-86156-2813',
    '1-93493-94147',
    '1-56299-42327',
    '1-38709-12493',
    '1-86510-12828',
    '1-93786-71166',
}


# Prompt Normalization for LVIS Categories
# LVIS categories often have compound words that text encoders struggle with.
# This maps them to simpler, more common terms.
# Generated from all 1067 uCO3D categories (373 compound words).

PROMPT_SIMPLIFICATIONS = {
    'aerosol_can': 'spray can',
    'air_conditioner': 'air conditioner',
    'alarm_clock': 'alarm clock',
    'arctic_type_of_shoe': 'shoe',
    'army_tank': 'tank',
    'automatic_washer': 'washing machine',
    'bullhorn': 'megaphone',
    'baby_buggy': 'stroller',
    'ballet_skirt': 'tutu',
    'band_aid': 'bandaid',
    'baseball_base': 'baseball diamond',
    'baseball_bat': 'baseball bat',
    'baseball_cap': 'cap',
    'baseball_glove': 'baseball glove',
    'basketball_backboard': 'backboard',
    'bass_horn': 'tuba',
    'bath_towel': 'towel',
    'batter_food': 'batter',
    'bean_curd': 'tofu',
    'beef_food': 'beef',
    'beer_bottle': 'beer bottle',
    'beer_can': 'beer can',
    'bell_pepper': 'pepper',
    'belt_buckle': 'buckle',
    'birthday_cake': 'birthday cake',
    'birthday_card': 'card',
    'black_sheep': 'sheep',
    'blinder_for_horses': 'horse blinder',
    'bobby_pin': 'hair pin',
    'boiled_egg': 'egg',
    'bolo_tie': 'tie',
    'boom_microphone': 'microphone',
    'bottle_cap': 'bottle cap',
    'bottle_opener': 'bottle opener',
    'bow-tie': 'bow tie',
    'bow_decorative_ribbons': 'ribbon bow',
    'bow_weapon': 'bow',
    'bowling_ball': 'bowling ball',
    'boxing_glove': 'boxing glove',
    'brake_light': 'brake light',
    'brass_plaque': 'plaque',
    'bread-bin': 'bread box',
    'bridal_gown': 'wedding dress',
    'brussels_sprouts': 'brussels sprouts',
    'bubble_gum': 'gum',
    'bullet_train': 'train',
    'bulletproof_vest': 'vest',
    'bunk_bed': 'bunk bed',
    'bus_vehicle': 'bus',
    'business_card': 'business card',
    'cab_taxi': 'taxi',
    'cabin_car': 'cabin',
    'camera_lens': 'lens',
    'camper_vehicle': 'camper',
    'can_opener': 'can opener',
    'candle_holder': 'candle holder',
    'candy_bar': 'candy bar',
    'candy_cane': 'candy cane',
    'cantaloup': 'cantaloupe',  # Fix LVIS spelling error
    'cap_headwear': 'cap',
    'car_automobile': 'car',
    'car_battery': 'car battery',
    'cargo_ship': 'cargo ship',
    'cash_register': 'cash register',
    'cayenne_spice': 'cayenne',
    'cd_player': 'cd player',
    'cellular_telephone': 'phone',
    'chain_mail': 'chainmail',
    'chaise_longue': 'lounge chair',
    'chicken_animal': 'chicken',
    'chili_vegetable': 'chili pepper',
    'chocolate_bar': 'chocolate bar',
    'chocolate_cake': 'chocolate cake',
    'chocolate_milk': 'chocolate milk',
    'chocolate_mousse': 'mousse',
    'chopping_board': 'cutting board',
    'christmas_tree': 'christmas tree',
    'cigar_box': 'cigar box',
    'cigarette_case': 'cigarette case',
    'cleansing_agent': 'cleaner',
    'cleaner': 'cleaning bag',
    'cleat_for_securing_rope': 'cleat',
    'clippers_for_plants': 'garden shears',
    'clock_tower': 'clock tower',
    'clothes_hamper': 'hamper',
    'clutch_bag': 'clutch purse',
    'coat_hanger': 'hanger',
    'cocoa_beverage': 'cocoa',
    'coffee_maker': 'coffee maker',
    'coffee_table': 'coffee table',
    'coloring_material': 'coloring',
    'combination_lock': 'lock',
    'comic_book': 'comic book',
    'computer_keyboard': 'keyboard',
    'convertible_automobile': 'convertible',
    'cooking_utensil': 'utensil',
    'cooler_for_food': 'cooler',
    'cornice': 'decorative wall',  # Architectural term - decorative wall trim
    'cork_bottle_plug': 'cork',
    'cowboy_hat': 'cowboy hat',
    'crab_animal': 'crab',
    'cream_pitcher': 'pitcher',
    'crescent_roll': 'croissant',
    'crisp_potato_chip': 'potato chip',
    'crock_pot': 'slow cooker',
    'cruise_ship': 'cruise ship',
    'cub_animal': 'cub',
    'curling_iron': 'curling iron',
    'date_fruit': 'date',
    'deck_chair': 'deck chair',
    'dental_floss': 'floss',
    'dining_table': 'dining table',
    'dirt_bike': 'dirt bike',
    'dish_antenna': 'satellite dish',
    'dishwasher_detergent': 'bottle',
    'diving_board': 'diving board',
    'dixie_cup': 'paper cup',
    'dog_collar': 'collar',
    'domestic_ass': 'donkey',
    'dress_hat': 'hat',
    'drum_musical_instrument': 'drum',
    'duct_tape': 'duct tape',
    'duffel_bag': 'duffel bag',
    'edible_corn': 'corn',
    'egg_roll': 'egg roll',
    'egg_yolk': 'yolk',
    'elevator_car': 'elevator',
    'fig_fruit': 'fig',
    'fighter_jet': 'jet',
    'file_cabinet': 'filing cabinet',
    'file_tool': 'file',
    'fire_alarm': 'fire alarm',
    'fire_engine': 'fire truck',
    'fire_extinguisher': 'fire extinguisher',
    'first-aid_kit': 'first aid kit',
    'fish_food': 'fish',
    'flip-flop_sandal': 'flip flops',
    'flipper_footwear': 'flippers',
    'flower_arrangement': 'flower arrangement',
    'flute_glass': 'champagne glass',
    'folding_chair': 'folding chair',
    'food_processor': 'food processor',
    'football_american': 'football',
    'football_helmet': 'football helmet',
    'freight_car': 'freight car',
    'french_toast': 'french toast',
    'fruit_juice': 'juice',
    'frying_pan': 'frying pan',
    'garbage_truck': 'garbage truck',
    'garden_hose': 'hose',
    'giant_panda': 'panda',
    'gift_wrap': 'wrapping paper',
    'glass_drink_container': 'glass',
    'golf_club': 'golf club',
    'gravy_boat': 'gravy boat',
    'green_bean': 'green bean',
    'green_onion': 'green onion',
    'grocery_bag': 'grocery bag',
    'hair_curler': 'hair curler',
    'hair_dryer': 'hair dryer',
    'halter_top': 'halter top',
    'hand_glass': 'magnifying glass',
    'hand_towel': 'hand towel',
    'hardback_book': 'hardcover book',
    'horse_buggy': 'buggy',
    'hot_sauce': 'hot sauce',
    'ice_maker': 'ice maker',
    'ice_pack': 'ice pack',
    'ice_skate': 'ice skate',
    'identity_card': 'id card',
    'iron_for_clothing': 'iron',
    'ironing_board': 'ironing board',
    'jelly_bean': 'jelly bean',
    'jet_plane': 'jet',
    'grape': 'grapes',
    'kitchen_sink': 'sink',
    'kitchen_table': 'kitchen table',
    'kiwi_fruit': 'kiwi',
    'knitting_needle': 'knitting needle',
    'knocker_on_a_door': 'door knocker',
    'lab_coat': 'lab coat',
    'lamb-chop': 'lamb chop',
    'lamb_animal': 'lamb',
    'laptop_computer': 'laptop',
    'lawn_mower': 'lawn mower',
    'legging_clothing': 'yoga pants',
    'life_buoy': 'life ring',
    'lightning_rod': 'lightning rod',
    'lip_balm': 'lip balm',
    'machine_gun': 'machine gun',
    'mailbox_at_home': 'mailbox',
    'mandarin_orange': 'mandarin',
    'mashed_potato': 'mashed potatoes',
    'mat_gym_equipment': 'gym mat',
    'measuring_cup': 'measuring cup',
    'microwave_oven': 'microwave',
    'milk_can': 'milk jug',
    'mint_candy': 'mint',
    'mixer_kitchen_tool': 'mixer',
    'motor_scooter': 'scooter',
    'motor_vehicle': 'vehicle',
    'mouse_computer_equipment': 'computer mouse',
    'music_stool': 'stool',
    'musical_instrument': 'instrument',
    'nosebag_for_animals': 'feed bag',
    'noseband_for_animals': 'bridle',
    'octopus_food': 'octopus',
    'oil_lamp': 'oil lamp',
    'olive_oil': 'olive oil',
    'orange_fruit': 'orange',
    'orange_juice': 'orange juice',
    'overalls_clothing': 'overalls',
    'coil': 'spring',
    'pan_for_cooking': 'pan',
    'pan_metal_container': 'pan',
    'paper_plate': 'paper plate',
    'paper_towel': 'paper towel',
    'paperback_book': 'paperback',
    'parking_meter': 'parking meter',
    'passenger_car_part_of_a_train': 'train car',
    'passenger_ship': 'ship',
    'patty_food': 'patty',
    'peanut_butter': 'peanut butter',
    'peeler_tool_for_fruit_and_vegetables': 'peeler',
    'pencil_box': 'pencil case',
    'pencil_sharpener': 'sharpener',
    'penny_coin': 'penny',
    'pepper_mill': 'pepper grinder',
    'phonograph_record': 'vinyl record',
    'pickup_truck': 'pickup truck',
    'piggy_bank': 'piggy bank',
    'pin_non_jewelry': 'pin',
    'ping-pong_ball': 'ping pong ball',
    'pipe_bowl': 'pipe',
    'pita_bread': 'pita',
    'pitcher_vessel_for_liquid': 'pitcher',
    'place_mat': 'placemat',
    'plastic_bag': 'plastic bag',
    'plow_farm_equipment': 'plow',
    'pocket_watch': 'pocket watch',
    'poker_chip': 'poker chip',
    'polar_bear': 'polar bear',
    'police_cruiser': 'police car',
    'pool_table': 'pool table',
    'pop_soda': 'soda',
    'postbox_public': 'mailbox',
    'postcard': 'postcard',
    'power_shovel': 'excavator',
    'projectile_weapon': 'bow',
    'pug-dog': 'pug',
    'race_car': 'race car',
    'radio_receiver': 'radio',
    'rag_doll': 'doll',
    'railcar_part_of_a_train': 'train car',
    'ram_animal': 'ram',
    'reamer_juicer': 'juicer',
    'rearview_mirror': 'mirror',
    'record_player': 'record player',
    'remote_control': 'remote',
    'rib_food': 'ribs',
    'river_boat': 'boat',
    'road_map': 'map',
    'rocking_chair': 'rocking chair',
    'roller_skate': 'roller skate',
    'rolling_pin': 'rolling pin',
    'root_beer': 'root beer',
    'router_computer_equipment': 'router',
    'rubber_band': 'rubber band',
    'runner_carpet': 'rug',
    'saddle_blanket': 'saddle pad',
    'safety_pin': 'safety pin',
    'salad_plate': 'plate',
    'salmon_fish': 'salmon',
    'salmon_food': 'salmon',
    'sandal_type_of_shoe': 'sandal',
    'scale_measuring_instrument': 'scale',
    'school_bus': 'school bus',
    'scrubbing_brush': 'scrub brush',
    'sewing_machine': 'sewing machine',
    'shampoo': 'label',
    'chinaware': 'dishes',
    'shaver_electric': 'electric razor',
    'shaving_cream': 'shaving cream',
    'shepherd_dog': 'sheepdog',
    'shopping_bag': 'shopping bag',
    'shopping_cart': 'shopping cart',
    'shot_glass': 'shot glass',
    'shoulder_bag': 'shoulder bag',
    'shower_cap': 'shower cap',
    'shower_curtain': 'shower curtain',
    'shower_head': 'shower head',
    'shredder_for_paper': 'paper shredder',
    'ski_boot': 'ski boot',
    'ski_parka': 'ski jacket',
    'sleeping_bag': 'sleeping bag',
    'sling_bandage': 'sling',
    'slipper_footwear': 'slippers',
    'soccer_ball': 'soccer ball',
    'sofa_bed': 'sofa bed',
    'solar_array': 'solar panel',
    'soup_bowl': 'soup bowl',
    'sour_cream': 'sour cream',
    'soya_milk': 'soy milk',
    'space_shuttle': 'space shuttle',
    'sparkler_fireworks': 'sparkler',
    'speaker_stero_equipment': 'speaker',
    'spice_rack': 'spice rack',
    'squid_food': 'squid',
    'stapler_stapling_machine': 'stapler',
    'statue_sculpture': 'statue',
    'steak_food': 'steak',
    'steak_knife': 'steak knife',
    'steering_wheel': 'steering wheel',
    'step_stool': 'step stool',
    'stereo_sound_system': 'stereo',
    'stop_sign': 'stop sign',
    'straw_for_drinking': 'straw',
    'street_sign': 'street sign',
    'string_cheese': 'string cheese',
    'sugar_bowl': 'sugar bowl',
    'sugarcane_plant': 'sugarcane',
    'suit_clothing': 'suit',
    'sweet_potato': 'sweet potato',
    'tabasco_sauce': 'hot sauce',
    'table-tennis_table': 'ping pong table',
    'table_lamp': 'lamp',
    'tank_storage_vessel': 'tank',
    'tape_measure': 'tape measure',
    'tape_sticky_cloth_or_paper': 'tape',
    'tea_bag': 'tea bag',
    'teddy_bear': 'teddy bear',
    'telephone_booth': 'phone booth',
    'telephone_pole': 'telephone pole',
    'telephoto_lens': 'telephoto lens',
    'television_camera': 'tv camera',
    'television_set': 'television',
    'tennis_ball': 'tennis ball',
    'tennis_racket': 'tennis racket',
    'thermos_bottle': 'thermos',
    'tights_clothing': 'tights',
    'toast_food': 'toast',
    'toaster_oven': 'toaster oven',
    'tobacco_pipe': 'pipe',
    'toilet_tissue': 'toilet paper',
    'tote_bag': 'tote bag',
    'tow_truck': 'truck',
    'towel_rack': 'towel',
    'tractor_farm_equipment': 'tractor',
    'traffic_light': 'traffic light',
    'trailer_truck': 'semi truck',
    'train_railroad_vehicle': 'train',
    'trash_can': 'trash can',
    'trench_coat': 'trench coat',
    'triangle_musical_instrument': 'triangle',
    'trophy_cup': 'trophy',
    'truffle_chocolate': 'truffle',
    'turkey_food': 'turkey',
    'turtleneck_clothing': 'turtleneck',
    'vacuum_cleaner': 'vacuum',
    'vending_machine': 'vending machine',
    'waffle_iron': 'waffle maker',
    'walking_cane': 'cane',
    'walking_stick': 'walking stick',
    'wall_clock': 'clock',
    'wall_socket': 'outlet',
    'water_bottle': 'water bottle',
    'water_cooler': 'water cooler',
    'water_faucet': 'faucet',
    'water_gun': 'water gun',
    'water_heater': 'water heater',
    'water_jug': 'water jug',
    'water_scooter': 'jet ski',
    'water_tower': 'water tower',
    'watering_can': 'watering can',
    'wedding_ring': 'ring',
    'whipped_cream': 'whipped cream',
    'wind_chime': 'wind chime',
    'window_box_for_plants': 'window planter',
    'wine_bottle': 'wine bottle',
    'wine_bucket': 'ice bucket',
    'wooden_leg': 'wooden leg',
    'wooden_spoon': 'wooden spoon',
}


def normalize_prompt(category: str, simplify: bool = True) -> str:
    """
    Normalize LVIS category names into cleaner prompts.

    Args:
        category: Raw LVIS category name (e.g., "dishwasher_detergent")
        simplify: Whether to use simplified mappings

    Returns:
        Normalized prompt string
    """
    # First check if we have a manual simplification
    if simplify and category in PROMPT_SIMPLIFICATIONS:
        return PROMPT_SIMPLIFICATIONS[category]

    # Replace underscores and hyphens with spaces
    prompt = category.replace('_', ' ').replace('-', ' ')

    # Clean up multiple spaces
    prompt = ' '.join(prompt.split())

    return prompt


class UCO3DMultiViewDataset(Dataset):
    """
    Multi-view dataset for uCO3D.

    Each sample returns multiple views of the same object with:
    - RGB images
    - Foreground segmentation masks (GT)
    - Depth maps (from DepthAnythingV2, scale-aligned)
    - Camera intrinsics/extrinsics
    - Category name as text prompt

    Sampling strategy (for evaluation as described in paper):
    - Sample 50 representative sequences across categories
    - Sample 50 frames uniformly from each sequence
    """

    def __init__(
        self,
        data_root: str = None,
        split: str = 'train',
        num_views: int = 8,
        image_size: Tuple[int, int] = (504, 504),
        mask_size: Tuple[int, int] = (128, 128),
        num_sequences: Optional[int] = None,  # None = all sequences
        frames_per_sequence: int = 50,  # Uniform sampling per sequence
        categories: Optional[List[str]] = None,  # Filter to specific categories
        use_depth: bool = True,
        subset_list: str = 'set_lists_all-categories.sqlite',
        seed: int = 42,
        samples_per_sequence: int = 1,  # For training: multiple samples per sequence
        normalize_prompts: bool = True,  # Simplify LVIS category names
        # K-fold cross validation
        num_folds: Optional[int] = None,  # Number of folds (e.g., 5 for 5-fold CV)
        fold: Optional[int] = None,  # Which fold to use as validation (0 to num_folds-1)
        # Cached DA3 depth
        use_cached_depth: bool = False,  # Load pre-computed DA3 depth from cache
        da3_cache_name: str = 'da3_metric_cache',  # Cache dir name under data_root
    ):
        """
        Args:
            data_root: Path to uCO3D dataset root. If None, uses UCO3D_DATASET_ROOT env var.
            split: 'train' or 'val'
            num_views: Number of views per sample
            image_size: (H, W) for image resizing
            mask_size: (H, W) for mask resizing
            num_sequences: Number of sequences to sample (None = all)
            frames_per_sequence: Number of frames to sample per sequence
            categories: List of categories to include (None = all)
            use_depth: Whether to load depth maps
            subset_list: Name of subset list file in set_lists/
            seed: Random seed for reproducible sampling
            samples_per_sequence: Number of training samples per sequence
            normalize_prompts: If True, simplify LVIS category names (e.g., "dishwasher_detergent" -> "detergent bottle")
            num_folds: For k-fold CV, number of folds. If set, ignores original train/val split.
            fold: Which fold to use as validation (0 to num_folds-1). Required if num_folds is set.
        """
        super().__init__()

        # Validate k-fold parameters
        if num_folds is not None:
            if fold is None:
                raise ValueError("fold must be specified when using num_folds")
            if fold < 0 or fold >= num_folds:
                raise ValueError(f"fold must be in [0, {num_folds-1}], got {fold}")

        self.split = split
        self.num_views = num_views
        self.image_size = image_size
        self.mask_size = mask_size
        self.num_sequences = num_sequences
        self.frames_per_sequence = frames_per_sequence
        self.categories = categories
        self.use_depth = use_depth
        self.seed = seed
        self.samples_per_sequence = samples_per_sequence
        self.normalize_prompts = normalize_prompts
        self.num_folds = num_folds
        self.fold = fold
        self.use_cached_depth = use_cached_depth
        self.da3_cache_name = da3_cache_name

        # Get dataset root
        if data_root is not None:
                self.data_root = Path(data_root)
        elif os.environ.get('UCO3D_DATASET_ROOT'):
            self.data_root = Path(os.environ['UCO3D_DATASET_ROOT'])
        else:
            # Try common locations (including symlinks)
            for path in ['data/uco3d', '/data/mv_sam3_data/uco3d', '/data/uco3d', '~/data/uco3d']:
                expanded = Path(path).expanduser()
                if expanded.exists():
                    self.data_root = expanded
                    break
            else:
                raise ValueError(
                    "uCO3D dataset root not found. Set UCO3D_DATASET_ROOT env var "
                    "or pass data_root argument."
                )

        print(f"[uCO3D] Loading from {self.data_root}")

        # Set up DA3 cache directory
        self.da3_cache_dir = self.data_root / da3_cache_name if use_cached_depth else None
        if use_cached_depth:
            if self.da3_cache_dir and self.da3_cache_dir.exists():
                print(f"[uCO3D] Using cached depth from {self.da3_cache_dir}")
            else:
                print(f"[uCO3D] Warning: DA3 cache not found: {self.da3_cache_dir}")
                print(f"  Run scripts/preprocess_da3_uco3d.py first.")

        # Initialize sequences
        self.sequences = []
        self._init_sequences(subset_list)

        # Sample sequences if requested
        if num_sequences is not None and len(self.sequences) > num_sequences:
            self._sample_representative_sequences(num_sequences)

        print(f"[uCO3D] Loaded {len(self.sequences)} sequences, "
              f"{self.frames_per_sequence} frames each, "
              f"{self.num_views} views per sample")

    def _init_sequences(self, subset_list: str):
        """Initialize sequence list from metadata."""

        # Try official sqlite set_lists first (has proper train/val splits)
        sqlite_path = self.data_root / "set_lists" / subset_list
        if sqlite_path.exists():
            self._init_from_sqlite(subset_list)
        elif HAS_UCO3D:
            self._init_with_uco3d_package(subset_list)
        else:
            self._init_standalone()

    def _init_with_uco3d_package(self, subset_list: str):
        """Initialize using official uCO3D package."""
        subset_lists_file = self.data_root / "set_lists" / subset_list

        if not subset_lists_file.exists():
            print(f"[uCO3D] Subset list not found: {subset_lists_file}")
            self._init_standalone()
            return

        # Build frame data builder for loading
        self.frame_builder = UCO3DFrameDataBuilder(
            apply_alignment=True,
            load_images=True,
            load_depths=self.use_depth,
            load_masks=True,
            load_depth_masks=False,
            load_gaussian_splats=False,
            load_point_clouds=False,
            load_segmented_point_clouds=False,
            load_sparse_point_clouds=False,
            box_crop=False,  # We handle cropping ourselves
            load_frames_from_videos=True,
            image_height=self.image_size[0],
            image_width=self.image_size[1],
            undistort_loaded_blobs=True,
        )

        # Create uCO3D dataset
        self.uco3d_dataset = UCO3DDatasetBase(
            subset_lists_file=str(subset_lists_file),
            subsets=[self.split],
            frame_data_builder=self.frame_builder,
            pick_categories=tuple(self.categories) if self.categories else (),
            n_frames_per_sequence=self.frames_per_sequence,
            seed=self.seed,
        )

        # Group frames by sequence
        sequence_frames = defaultdict(list)
        for i in range(len(self.uco3d_dataset)):
            try:
                # Get sequence name without loading full data
                meta = self.uco3d_dataset._index.iloc[i]
                seq_name = meta['sequence_name']
                sequence_frames[seq_name].append(i)
            except Exception:
                continue

        # Build sequence list
        for seq_name, frame_indices in sequence_frames.items():
            if len(frame_indices) >= self.num_views:
                category = seq_name.split('/')[0] if '/' in seq_name else 'unknown'
                # Skip categories with bad mask quality
                if category in SKIP_CATEGORIES:
                    continue
                # Skip specific mislabeled sequences
                seq_id = seq_name.split('/')[-1] if '/' in seq_name else seq_name
                if seq_id in SKIP_SEQUENCES:
                    continue
                self.sequences.append({
                    'sequence_name': seq_name,
                    'frame_indices': frame_indices,
                    'category': category,
                })

    def _init_from_sqlite(self, subset_list: str):
        """Initialize from official sqlite set_lists (proper train/val split or k-fold CV)."""
        import sqlite3

        sqlite_path = self.data_root / "set_lists" / subset_list
        if not sqlite_path.exists():
            print(f"[uCO3D] Set list not found: {sqlite_path}, falling back to directory scan")
            self._init_standalone()
            return

        print(f"[uCO3D] Loading from sqlite: {sqlite_path}")
        conn = sqlite3.connect(str(sqlite_path))
        cursor = conn.cursor()

        # K-fold cross validation: load ALL sequences and split into folds
        if self.num_folds is not None:
            cursor.execute(
                "SELECT sequence_name, category, super_category, subset FROM sequence_lengths"
            )
            all_rows = cursor.fetchall()
            conn.close()

            # Sort deterministically by sequence name for reproducible splits
            all_rows = sorted(all_rows, key=lambda x: x[0])

            # Shuffle with seed for randomized but reproducible folds
            rng = random.Random(self.seed)
            rng.shuffle(all_rows)

            # Split into folds
            fold_size = len(all_rows) // self.num_folds
            fold_starts = [i * fold_size for i in range(self.num_folds)]
            fold_starts.append(len(all_rows))  # End marker

            val_start = fold_starts[self.fold]
            val_end = fold_starts[self.fold + 1]

            if self.split == 'val':
                rows = all_rows[val_start:val_end]
            else:  # train
                rows = all_rows[:val_start] + all_rows[val_end:]

            # Strip the subset column (4th element) since we determined split via fold
            rows = [(r[0], r[1], r[2]) for r in rows]

            print(f"[uCO3D] K-fold CV: {self.num_folds} folds, fold {self.fold} as val")
            print(f"[uCO3D] Found {len(rows)} sequences for split '{self.split}' (fold {self.fold})")
        else:
            # Standard train/val split from dataset
            cursor.execute(
                "SELECT sequence_name, category, super_category FROM sequence_lengths WHERE subset = ?",
                (self.split,)
            )
            rows = cursor.fetchall()
            conn.close()

            print(f"[uCO3D] Found {len(rows)} sequences for split '{self.split}'")

        for seq_name, category, super_category in rows:
            # Skip categories with bad mask quality
            if category in SKIP_CATEGORIES:
                continue

            # Skip specific mislabeled sequences
            if seq_name in SKIP_SEQUENCES:
                continue

            # Build path: super_category/category/sequence_name
            seq_path = self.data_root / super_category / category / seq_name
            rgb_video = seq_path / "rgb_video.mp4"
            mask_video = seq_path / "mask_video.mkv"

            if rgb_video.exists() and mask_video.exists():
                self.sequences.append({
                    'sequence_name': f"{super_category}/{category}/{seq_name}",
                    'sequence_path': seq_path,
                    'category': category,
                    'super_category': super_category,
                    'rgb_video': rgb_video,
                    'mask_video': mask_video,
                    'depth_file': seq_path / "depth_maps.h5" if self.use_depth else None,
                })

        print(f"[uCO3D] Loaded {len(self.sequences)} sequences with valid videos")

    def _init_standalone(self):
        """Initialize without uCO3D package (directory scanning) - no train/val separation."""
        print("[uCO3D] Using standalone initialization (scanning directories)")
        print("[uCO3D] WARNING: No train/val separation - use sqlite set_lists for proper splits")

        # Scan for sequences in super_category/category/sequence structure
        for super_cat_dir in self.data_root.iterdir():
            if not super_cat_dir.is_dir() or super_cat_dir.name.startswith('.'):
                continue
            if super_cat_dir.name in ['set_lists', 'metadata.sqlite']:
                continue

            for cat_dir in super_cat_dir.iterdir():
                if not cat_dir.is_dir():
                    continue

                # Filter categories if specified
                if self.categories and cat_dir.name not in self.categories:
                    continue

                # Skip categories with bad mask quality
                if cat_dir.name in SKIP_CATEGORIES:
                    continue

                for seq_dir in cat_dir.iterdir():
                    if not seq_dir.is_dir():
                        continue

                    # Skip specific mislabeled sequences
                    if seq_dir.name in SKIP_SEQUENCES:
                        continue

                    # Check for required files
                    rgb_video = seq_dir / "rgb_video.mp4"
                    mask_video = seq_dir / "mask_video.mkv"

                    if rgb_video.exists() and mask_video.exists():
                        self.sequences.append({
                            'sequence_name': f"{super_cat_dir.name}/{cat_dir.name}/{seq_dir.name}",
                            'sequence_path': seq_dir,
                            'category': cat_dir.name,
                            'super_category': super_cat_dir.name,
                            'rgb_video': rgb_video,
                            'mask_video': mask_video,
                            'depth_file': seq_dir / "depth_maps.h5" if self.use_depth else None,
                        })

        print(f"[uCO3D] Found {len(self.sequences)} sequences via directory scan")

    def _sample_representative_sequences(self, num_sequences: int):
        """Sample representative sequences across categories."""
        random.seed(self.seed)

        # Group by category
        by_category = defaultdict(list)
        for seq in self.sequences:
            by_category[seq['category']].append(seq)

        # Sample proportionally from each category
        num_categories = len(by_category)
        seqs_per_category = max(1, num_sequences // num_categories)

        sampled = []
        for cat, seqs in by_category.items():
            n_sample = min(len(seqs), seqs_per_category)
            sampled.extend(random.sample(seqs, n_sample))

        # If we need more, sample randomly from remaining
        if len(sampled) < num_sequences:
            remaining = [s for s in self.sequences if s not in sampled]
            extra_needed = num_sequences - len(sampled)
            if remaining:
                sampled.extend(random.sample(remaining, min(len(remaining), extra_needed)))

        # If we have too many, trim
        if len(sampled) > num_sequences:
            sampled = random.sample(sampled, num_sequences)

        self.sequences = sampled
        print(f"[uCO3D] Sampled {len(self.sequences)} representative sequences "
              f"from {num_categories} categories")

    def __len__(self):
        return len(self.sequences) * self.samples_per_sequence

    def _make_fallback_sample(self, category: str) -> Dict:
        """Return a valid but empty sample to avoid DDP hangs on bad sequences."""
        prompt = normalize_prompt(category) if self.normalize_prompts else category
        return {
            'images': torch.zeros(self.num_views, 3, self.image_size[0], self.image_size[1]),
            'gt_masks': torch.zeros(self.num_views, self.mask_size[0], self.mask_size[1]),
            'intrinsics': torch.eye(3).unsqueeze(0).repeat(self.num_views, 1, 1),
            'extrinsics': torch.eye(4).unsqueeze(0).repeat(self.num_views, 1, 1),
            'orig_hw': self.image_size,
            'scene_id': 'fallback_bad_sequence',
            'prompt': prompt,
            'image_names': ['fallback'] * self.num_views,
            'has_metric_scale': True,
            'has_gt_mask': False,  # Signal to training loop: skip loss for this sample
            'category': category,
        }

    def _load_frames_from_video(
        self,
        video_path: Path,
        frame_indices: List[int],
        is_mask: bool = False
    ) -> List[np.ndarray]:
        """Load specific frames from video file."""
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        for idx in frame_indices:
            if idx >= total_frames:
                idx = total_frames - 1

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:
                if is_mask:
                    # Mask video is grayscale, threshold to binary
                    if len(frame.shape) == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = (frame > 127).astype(np.float32)
                else:
                    # RGB video
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # Fallback: return zeros
                if is_mask:
                    frames.append(np.zeros((self.mask_size[0], self.mask_size[1]), dtype=np.float32))
                else:
                    frames.append(np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8))

        cap.release()
        return frames

    def _load_depth_from_h5(
        self,
        h5_path: Path,
        frame_indices: List[int]
    ) -> List[np.ndarray]:
        """Load depth maps from HDF5 file."""
        if not HAS_H5PY:
            return [np.ones((self.image_size[0], self.image_size[1]), dtype=np.float32)
                    for _ in frame_indices]

        if not h5_path.exists():
            return [np.ones((self.image_size[0], self.image_size[1]), dtype=np.float32)
                    for _ in frame_indices]

        depths = []
        with h5py.File(h5_path, 'r') as f:
            # uCO3D stores 200 equidistant depth maps
            depth_keys = sorted(f.keys(), key=lambda x: int(x) if x.isdigit() else 0)
            total_depths = len(depth_keys)

            for idx in frame_indices:
                # Map frame index to depth index (200 depths for full video)
                depth_idx = min(idx, total_depths - 1)
                if depth_idx < len(depth_keys):
                    depth = np.array(f[depth_keys[depth_idx]])
                    depths.append(depth)
                else:
                    depths.append(np.ones((self.image_size[0], self.image_size[1]), dtype=np.float32))

        return depths

    def _sample_frame_indices(self, total_frames: int, num_samples: int) -> List[int]:
        """Uniformly sample frame indices."""
        if total_frames <= num_samples:
            return list(range(total_frames))

        # Uniform sampling
        step = total_frames / num_samples
        return [int(i * step) for i in range(num_samples)]

    def __getitem__(self, idx):
        # Map index to sequence
        seq_idx = idx % len(self.sequences)
        sample_idx = idx // len(self.sequences)

        seq = self.sequences[seq_idx]
        category = seq['category']

        try:
            # Use uCO3D package if available
            if HAS_UCO3D and hasattr(self, 'uco3d_dataset'):
                return self._getitem_uco3d(seq, sample_idx)
            else:
                return self._getitem_standalone(seq, sample_idx)
        except Exception as e:
            # Bad video/sequence — return a fallback sample to avoid DDP hang
            print(f"[uCO3D WARNING] Failed to load sequence {seq.get('sequence_name', seq_idx)}: {e}")
            return self._make_fallback_sample(category)

    def _getitem_uco3d(self, seq: Dict, sample_idx: int) -> Dict:
        """Load sample using uCO3D package."""
        frame_indices = seq['frame_indices']

        # Sample views from available frames
        if len(frame_indices) > self.num_views:
            # Random sampling for training, uniform for eval
            if self.split == 'train':
                random.seed(sample_idx)  # Reproducible per sample
                selected_indices = random.sample(frame_indices, self.num_views)
            else:
                step = len(frame_indices) // self.num_views
                selected_indices = [frame_indices[i * step] for i in range(self.num_views)]
        else:
            selected_indices = frame_indices[:self.num_views]

        images = []
        masks = []
        depths = []
        intrinsics_list = []
        extrinsics_list = []

        for fidx in selected_indices:
            frame_data = self.uco3d_dataset[fidx]

            # RGB image: [3, H, W]
            img = frame_data.image_rgb
            if img is not None:
                images.append(img)
            else:
                images.append(torch.zeros(3, self.image_size[0], self.image_size[1]))

            # Mask: [1, H, W] -> [H, W]
            mask = frame_data.fg_mask
            if mask is not None:
                if mask.dim() == 3:
                    mask = mask.squeeze(0)
                # Resize to mask_size
                mask_resized = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(),
                    size=self.mask_size,
                    mode='nearest'
                ).squeeze()
                masks.append(mask_resized)
            else:
                masks.append(torch.zeros(self.mask_size))

            # Depth
            if self.use_depth and frame_data.depth_map is not None:
                depth = frame_data.depth_map
                if depth.dim() == 3:
                    depth = depth.squeeze(0)
                depths.append(depth)

            # Camera parameters from uCO3D
            cam = frame_data.camera
            if cam is not None:
                # uCO3D uses PyTorch3D convention, convert to standard 4x4 extrinsics
                R = cam.R[0] if cam.R.dim() == 3 else cam.R  # [3, 3]
                T = cam.T[0] if cam.T.dim() == 2 else cam.T  # [3]

                extrinsics = torch.eye(4)
                extrinsics[:3, :3] = R
                extrinsics[:3, 3] = T
                extrinsics_list.append(extrinsics)

                # Intrinsics from focal length and principal point
                focal = cam.focal_length[0] if cam.focal_length.dim() == 2 else cam.focal_length
                pp = cam.principal_point[0] if cam.principal_point.dim() == 2 else cam.principal_point

                fx, fy = focal[0].item(), focal[1].item()
                cx, cy = pp[0].item(), pp[1].item()

                intrinsics = torch.tensor([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ], dtype=torch.float32)
                intrinsics_list.append(intrinsics)
            else:
                intrinsics_list.append(torch.eye(3))
                extrinsics_list.append(torch.eye(4))

        # Normalize prompt if enabled
        category = seq['category']
        prompt = normalize_prompt(category) if self.normalize_prompts else category

        result = {
            'images': torch.stack(images),  # [N, 3, H, W]
            'gt_masks': torch.stack(masks),  # [N, H, W]
            'intrinsics': torch.stack(intrinsics_list),  # [N, 3, 3]
            'extrinsics': torch.stack(extrinsics_list),  # [N, 4, 4]
            'orig_hw': self.image_size,
            'scene_id': seq['sequence_name'],
            'prompt': prompt,  # Normalized category name as text prompt
            'image_names': [str(i) for i in selected_indices],
            'has_metric_scale': True,  # uCO3D depth is aligned with VGGSfM
            'has_gt_mask': True,
            'category': category,  # Keep original category for metrics
        }

        if depths:
            result['depths'] = torch.stack(depths)  # [N, H, W]

        return result

    def _getitem_standalone(self, seq: Dict, sample_idx: int) -> Dict:
        """Load sample using direct video loading (no uCO3D package)."""
        import cv2

        rgb_video = seq['rgb_video']
        mask_video = seq['mask_video']
        depth_file = seq.get('depth_file')

        # Get total frames from RGB video
        cap = cv2.VideoCapture(str(rgb_video))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Sample frame indices uniformly from the video
        candidate_frames = self._sample_frame_indices(total_frames, self.frames_per_sequence)

        # Select views from candidate frames
        if len(candidate_frames) > self.num_views:
            if self.split == 'train':
                random.seed(sample_idx + self.seed)
                selected_frames = random.sample(candidate_frames, self.num_views)
            else:
                step = len(candidate_frames) // self.num_views
                selected_frames = [candidate_frames[i * step] for i in range(self.num_views)]
        else:
            selected_frames = candidate_frames[:self.num_views]
            # Pad with duplicates if needed
            while len(selected_frames) < self.num_views:
                selected_frames.append(selected_frames[-1])

        # Load RGB frames
        rgb_frames = self._load_frames_from_video(rgb_video, selected_frames, is_mask=False)

        # Load mask frames
        mask_frames = self._load_frames_from_video(mask_video, selected_frames, is_mask=True)

        # Load depth if available
        depth_frames = None
        if self.use_depth and depth_file and depth_file.exists():
            depth_frames = self._load_depth_from_h5(depth_file, selected_frames)

        # Process frames
        images = []
        masks = []
        depths = []

        for i, (rgb, mask) in enumerate(zip(rgb_frames, mask_frames)):
            # Resize and convert RGB
            rgb_pil = Image.fromarray(rgb)
            rgb_pil = rgb_pil.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
            rgb_tensor = torch.from_numpy(np.array(rgb_pil)).float() / 255.0
            rgb_tensor = rgb_tensor.permute(2, 0, 1)  # [3, H, W]
            images.append(rgb_tensor)

            # Resize mask
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((self.mask_size[1], self.mask_size[0]), Image.NEAREST)
            mask_tensor = torch.from_numpy(np.array(mask_pil)).float() / 255.0
            masks.append(mask_tensor)

            # Resize depth
            if depth_frames is not None and i < len(depth_frames):
                depth = depth_frames[i]
                depth_pil = Image.fromarray(depth.astype(np.float32), mode='F')
                depth_pil = depth_pil.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
                depth_tensor = torch.from_numpy(np.array(depth_pil)).float()
                depths.append(depth_tensor)

        # Default intrinsics/extrinsics (turntable assumption)
        # In standalone mode, we don't have camera params, use identity
        intrinsics = torch.eye(3).unsqueeze(0).repeat(self.num_views, 1, 1)
        extrinsics = torch.eye(4).unsqueeze(0).repeat(self.num_views, 1, 1)

        # Normalize prompt if enabled
        category = seq['category']
        prompt = normalize_prompt(category) if self.normalize_prompts else category

        result = {
            'images': torch.stack(images),  # [N, 3, H, W]
            'gt_masks': torch.stack(masks),  # [N, H, W]
            'intrinsics': intrinsics,  # [N, 3, 3]
            'extrinsics': extrinsics,  # [N, 4, 4]
            'orig_hw': self.image_size,
            'scene_id': seq['sequence_name'],
            'prompt': prompt,  # Normalized category name
            'image_names': [str(f) for f in selected_frames],
            'has_metric_scale': True,
            'has_gt_mask': True,
            'category': category,  # Keep original for metrics
        }

        if depths:
            result['depths'] = torch.stack(depths)  # [N, H, W]

        # Load cached DA3 depth if available (bypasses live DA3 inference)
        if self.use_cached_depth and self.da3_cache_dir is not None:
            seq_cache_dir = (self.da3_cache_dir / seq['super_category']
                             / seq['category'] / seq['sequence_name'])
            cached_depths = []
            cached_extrinsics = []
            cached_intrinsics = []
            all_cached = True
            has_poses = True

            for frame_idx in selected_frames:
                cache_path = seq_cache_dir / f"frame_{frame_idx:06d}.pt"
                if cache_path.exists():
                    try:
                        cache_data = torch.load(cache_path, map_location='cpu',
                                                weights_only=True, mmap=False)
                        depth = cache_data['depth']
                        if depth.dim() == 2:
                            depth = depth.unsqueeze(0)  # [H, W] -> [1, H, W]
                        cached_depths.append(depth.float())

                        if 'extrinsics' in cache_data:
                            cached_extrinsics.append(cache_data['extrinsics'].float())
                        else:
                            has_poses = False
                        if 'intrinsics' in cache_data:
                            cached_intrinsics.append(cache_data['intrinsics'].float())
                    except Exception:
                        all_cached = False
                        break
                else:
                    all_cached = False
                    break

            if all_cached and len(cached_depths) == len(selected_frames):
                cached_depth_tensor = torch.stack(cached_depths)  # (N, 1, H, W)

                # Resize cached depth to image_size if needed
                cache_h, cache_w = cached_depth_tensor.shape[-2:]
                target_h, target_w = self.image_size
                if cache_h != target_h or cache_w != target_w:
                    cached_depth_tensor = F.interpolate(
                        cached_depth_tensor,
                        size=(target_h, target_w),
                        mode='bilinear',
                        align_corners=False,
                    )

                result['cached_depth'] = cached_depth_tensor  # (N, 1, H, W)

                if has_poses and len(cached_extrinsics) == len(selected_frames):
                    result['cached_da3_extrinsics'] = torch.stack(cached_extrinsics)
                if len(cached_intrinsics) == len(selected_frames):
                    result['cached_da3_intrinsics'] = torch.stack(cached_intrinsics)

        return result


def get_uco3d_eval_config() -> Dict:
    """
    Get evaluation configuration matching the paper description:
    - 50 representative sequences
    - 50 frames per sequence (uniformly sampled)
    """
    return {
        'num_sequences': 50,
        'frames_per_sequence': 50,
        'num_views': 8,  # Views per evaluation sample
        'split': 'val',
    }


def get_uco3d_kfold_datasets(
    num_folds: int = 5,
    fold: int = 0,
    num_sequences: Optional[int] = None,
    **kwargs,
) -> Tuple['UCO3DMultiViewDataset', 'UCO3DMultiViewDataset']:
    """
    Create train and val datasets for k-fold cross validation.

    Args:
        num_folds: Number of folds (e.g., 5 for 5-fold CV)
        fold: Which fold to use as validation (0 to num_folds-1)
        num_sequences: Optional limit on sequences per split (after fold split)
        **kwargs: Additional arguments passed to UCO3DMultiViewDataset

    Returns:
        (train_dataset, val_dataset) tuple

    Example:
        # 5-fold cross validation, using fold 0 as validation
        train_ds, val_ds = get_uco3d_kfold_datasets(num_folds=5, fold=0)

        # Run all 5 folds
        for fold in range(5):
            train_ds, val_ds = get_uco3d_kfold_datasets(num_folds=5, fold=fold)
            # Train and evaluate...
    """
    train_dataset = UCO3DMultiViewDataset(
        split='train',
        num_folds=num_folds,
        fold=fold,
        num_sequences=num_sequences,
        **kwargs,
    )
    val_dataset = UCO3DMultiViewDataset(
        split='val',
        num_folds=num_folds,
        fold=fold,
        num_sequences=num_sequences,
        **kwargs,
    )
    return train_dataset, val_dataset


def get_uco3d_train_eval_split(
    data_root: str = None,
    num_train_sequences: int = 500,
    num_eval_sequences: int = 50,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Create train/eval split for uCO3D with non-overlapping sequences.

    Paper protocol (uco3d.md): 50 sequences for eval, 50 frames each.
    For training, use separate sequences.

    Args:
        data_root: Path to uCO3D dataset
        num_train_sequences: Number of sequences for training
        num_eval_sequences: Number of sequences for evaluation (default: 50 per paper)
        seed: Random seed for reproducible splits

    Returns:
        (train_sequences, eval_sequences) - lists of sequence paths
    """
    import random

    # Create temporary dataset to get all sequences
    temp_dataset = UCO3DMultiViewDataset(
        data_root=data_root,
        num_sequences=None,  # Get all
        frames_per_sequence=1,  # Minimal loading
        use_depth=False,
    )

    all_sequences = temp_dataset.sequences.copy()
    random.seed(seed)
    random.shuffle(all_sequences)

    # Split: first N for eval (held out), rest for train
    eval_sequences = all_sequences[:num_eval_sequences]
    train_sequences = all_sequences[num_eval_sequences:num_eval_sequences + num_train_sequences]

    print(f"[uCO3D Split] Train: {len(train_sequences)} sequences, Eval: {len(eval_sequences)} sequences")

    return train_sequences, eval_sequences


def create_uco3d_eval_dataset(
    data_root: str = None,
    num_views: int = 8,
    image_size: Tuple[int, int] = (504, 504),
    mask_size: Tuple[int, int] = (128, 128),
    normalize_prompts: bool = True,
    num_sequences: int = None,
    frames_per_sequence: int = None,
    seed: int = 42,
    # K-fold cross validation
    num_folds: Optional[int] = None,
    fold: Optional[int] = None,
) -> UCO3DMultiViewDataset:
    """
    Create uCO3D evaluation dataset with paper-specified sampling.

    From uco3d.md: "We sample 50 representative sequences...uniformly sample
    50 frames from each sequence"

    Args:
        normalize_prompts: If True, simplify LVIS category names for better text encoder understanding
        num_sequences: Number of sequences to sample (default: 50 from paper protocol)
        frames_per_sequence: Frames per sequence (default: 50 from paper protocol)
        seed: Random seed for reproducible sequence sampling
        num_folds: For k-fold CV, number of folds (e.g., 5). If set, ignores original train/val split.
        fold: Which fold to use as validation (0 to num_folds-1). Required if num_folds is set.
    """
    config = get_uco3d_eval_config()

    return UCO3DMultiViewDataset(
        data_root=data_root,
        split=config['split'],
        num_views=num_views,
        image_size=image_size,
        mask_size=mask_size,
        num_sequences=num_sequences if num_sequences is not None else config['num_sequences'],
        frames_per_sequence=frames_per_sequence if frames_per_sequence is not None else config['frames_per_sequence'],
        samples_per_sequence=1,  # One sample per sequence for eval
        normalize_prompts=normalize_prompts,
        seed=seed,
        num_folds=num_folds,
        fold=fold,
    )


if __name__ == '__main__':
    # Test the dataset
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default=None)
    parser.add_argument('--num-sequences', type=int, default=5)
    parser.add_argument('--num-views', type=int, default=4)
    args = parser.parse_args()

    print("Testing UCO3DMultiViewDataset...")

    dataset = UCO3DMultiViewDataset(
        data_root=args.data_root,
        split='train',
        num_views=args.num_views,
        num_sequences=args.num_sequences,
        frames_per_sequence=20,
    )

    print(f"Dataset length: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample keys: {sample.keys()}")
        print(f"Images shape: {sample['images'].shape}")
        print(f"GT masks shape: {sample['gt_masks'].shape}")
        print(f"Prompt: {sample['prompt']}")
        print(f"Scene ID: {sample['scene_id']}")
        print(f"Category: {sample['category']}")
