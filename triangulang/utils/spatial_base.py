"""Dependency-free spatial vocabulary tables shared by spatial_reasoning and spatial_context.

Holds the qualifier/relation lookup tables so both modules can import them at the
top without a circular import (spatial_reasoning <-> spatial_context).
"""

SPATIAL_QUALIFIERS = {
    # Depth-based (extremes)
    'nearest': 'depth_min', 'closest': 'depth_min', 'close': 'depth_min',
    'farthest': 'depth_max', 'far': 'depth_max', 'distant': 'depth_max',
    # Depth-based (ordinal)
    'second nearest': 'depth_2nd_min', 'second closest': 'depth_2nd_min',
    'second farthest': 'depth_2nd_max',
    # Horizontal position (extremes)
    'leftmost': 'x_min', 'left': 'x_min',
    'rightmost': 'x_max', 'right': 'x_max',
    # Horizontal position (ordinal)
    'second leftmost': 'x_2nd_min', 'second from left': 'x_2nd_min',
    'second rightmost': 'x_2nd_max', 'second from right': 'x_2nd_max',
    # Vertical position (extremes)
    'topmost': 'y_min', 'top': 'y_min', 'upper': 'y_min',
    'bottommost': 'y_max', 'bottom': 'y_max', 'lower': 'y_max',
    # Vertical position (ordinal)
    'second topmost': 'y_2nd_min', 'second from top': 'y_2nd_min',
    'second bottommost': 'y_2nd_max', 'second from bottom': 'y_2nd_max',
    # Middle/center position (for objects not at extremes)
    'middle': 'middle', 'central': 'middle', 'center': 'middle',
    'mid-depth': 'depth_mid', 'middle depth': 'depth_mid',
    # Size-based (using mask coverage)
    'largest': 'size_max', 'biggest': 'size_max', 'big': 'size_max',
    'smallest': 'size_min', 'small': 'size_min', 'tiny': 'size_min',
}

SPATIAL_QUALIFIER_TO_IDX = {
    None: 0,
    'depth_min': 1,      # nearest
    'depth_max': 2,      # farthest
    'x_min': 3,          # leftmost
    'x_max': 4,          # rightmost
    'y_min': 5,          # top
    'y_max': 6,          # bottom
    'middle': 7,         # middle/center
    # Extended qualifiers (share indices with related extremes for embedding)
    'depth_2nd_min': 1,  # second nearest -> nearest embedding
    'depth_2nd_max': 2,  # second farthest -> farthest embedding
    'x_2nd_min': 3,      # second leftmost -> leftmost embedding
    'x_2nd_max': 4,      # second rightmost -> rightmost embedding
    'y_2nd_min': 5,      # second topmost -> top embedding
    'y_2nd_max': 6,      # second bottommost -> bottom embedding
    'depth_mid': 7,      # middle depth -> middle embedding
    'size_max': 1,       # largest -> reuse nearest (both are "more")
    'size_min': 2,       # smallest -> reuse farthest (both are "less")
}

# Relational patterns for parsing "X <relation> Y" queries
RELATION_PATTERNS = [
    # "X to the right of Y", "X on the right of Y"
    (r'(.+?)\s+(?:to\s+the\s+|on\s+the\s+)?right\s+of\s+(?:the\s+)?(.+)', 'right_of'),
    (r'(.+?)\s+(?:to\s+the\s+|on\s+the\s+)?left\s+of\s+(?:the\s+)?(.+)', 'left_of'),
    # "X above Y", "X over Y"
    (r'(.+?)\s+(?:above|over)\s+(?:the\s+)?(.+)', 'above'),
    (r'(.+?)\s+(?:below|under|beneath)\s+(?:the\s+)?(.+)', 'below'),
    # "X near Y", "X next to Y", "X beside Y"
    (r'(.+?)\s+(?:near|next\s+to|beside|by)\s+(?:the\s+)?(.+)', 'near'),
    # "X on Y", "X on top of Y"
    (r'(.+?)\s+on\s+(?:top\s+of\s+)?(?:the\s+)?(.+)', 'on_top_of'),
    # "X in front of Y"
    (r'(.+?)\s+in\s+front\s+of\s+(?:the\s+)?(.+)', 'in_front_of'),
    # "X behind Y"
    (r'(.+?)\s+behind\s+(?:the\s+)?(.+)', 'behind'),
]
