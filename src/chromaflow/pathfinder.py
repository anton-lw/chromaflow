# src/chromaflow/pathfinder.py
from __future__ import annotations

from collections import deque
from collections.abc import Callable

import numpy as np

from . import functional as F
from .exceptions import ConversionPathError

ConversionFunc = Callable[[np.ndarray], np.ndarray]

_CONVERSION_GRAPH: dict[tuple[str, str], ConversionFunc] = {
    ("srgb", "srgb-linear"): F.srgb_to_srgb_linear,
    ("srgb-linear", "srgb"): F.srgb_linear_to_srgb,
    ("srgb-linear", "xyz-d65"): F.srgb_linear_to_xyz_d65,
    ("xyz-d65", "srgb-linear"): F.xyz_d65_to_srgb_linear,
    ("xyz-d65", "lab-d65"): F.xyz_d65_to_lab_d65,
    ("lab-d65", "xyz-d65"): F.lab_d65_to_xyz_d65,
    ("xyz-d65", "oklab"): F.xyz_d65_to_oklab,
    ("oklab", "xyz-d65"): F.oklab_to_xyz_d65,
    ("oklab", "oklch"): F.oklab_to_oklch,
    ("oklch", "oklab"): F.oklch_to_oklab,
    ("xyz-d65", "jzazbz"): F.xyz_d65_to_jzazbz,
    ("jzazbz", "xyz-d65"): F.jzazbz_to_xyz_d65,
    ("p3-d65", "p3-d65-linear"): F.p3_d65_to_p3_d65_linear,
    ("p3-d65-linear", "p3-d65"): F.p3_d65_linear_to_p3_d65,
    ("p3-d65-linear", "xyz-d65"): F.p3_d65_linear_to_xyz_d65,
    ("xyz-d65", "p3-d65-linear"): F.xyz_d65_to_p3_d65_linear,
    ("adobe-rgb", "adobe-rgb-linear"): F.adobe_rgb_to_adobe_rgb_linear,
    ("adobe-rgb-linear", "adobe-rgb"): F.adobe_rgb_linear_to_adobe_rgb,
    ("adobe-rgb-linear", "xyz-d65"): F.adobe_rgb_linear_to_xyz_d65,
    ("xyz-d65", "adobe-rgb-linear"): F.xyz_d65_to_adobe_rgb_linear,
}

_ADJACENCY_LIST: dict[str, set[str]] = {}
for from_space, to_space in _CONVERSION_GRAPH.keys():
    if from_space not in _ADJACENCY_LIST:
        _ADJACENCY_LIST[from_space] = set()
    _ADJACENCY_LIST[from_space].add(to_space)

# Cache for memoizing found paths to avoid re-computing.
_PATH_CACHE: dict[tuple[str, str], list[str] | None] = {}


def get_conversion_function(from_space: str, to_space: str) -> ConversionFunc:
    """Retrieves the specific function for a direct conversion step."""
    try:
        return _CONVERSION_GRAPH[(from_space, to_space)]
    except KeyError:
        raise ConversionPathError(
            f"No direct conversion function registered from '{from_space}' "
            f"to '{to_space}'."
        )


def find_conversion_path(start_node: str, end_node: str) -> list[str] | None:
    """
    Finds the shortest conversion path between two color spaces using
    Breadth-First Search (BFS).

    Args:
        start_node: The name of the starting color space.
        end_node: The name of the target color space.

    Returns:
        A list of color space names representing the full path
        (including start and end), or None if no path exists.
    """
    start_node = start_node.lower()
    end_node = end_node.lower()

    if start_node == end_node:
        return [start_node]

    # Check cache first
    if (start_node, end_node) in _PATH_CACHE:
        return _PATH_CACHE[(start_node, end_node)]

    # BFS implementation
    queue = deque([[start_node]])  # A queue of paths
    visited = {start_node}

    while queue:
        path = queue.popleft()
        node = path[-1]

        if node == end_node:
            # Path found, store in cache and return
            _PATH_CACHE[(start_node, end_node)] = path
            return path

        # Explore neighbors
        for neighbor in _ADJACENCY_LIST.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)

    # If the loop finishes, no path was found
    _PATH_CACHE[(start_node, end_node)] = None
    return None
