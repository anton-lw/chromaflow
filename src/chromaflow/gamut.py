# src/chromaflow/gamut.py
from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from .color_object import Color

def in_gamut(color: Color, target_gamut_space: str = "srgb", tolerance: float = 1e-5) -> bool:
    """
    Checks if a color is within the specified RGB gamut.

    This works by converting the color to the target RGB space and checking if
    all channel values are within the [0, 1] range.

    Args:
        color: The ChromaFlow Color object to check.
        target_gamut_space: The name of the target RGB space (e.g., 'srgb').
        tolerance: The tolerance for the check.

    Returns:
        True if the color is in gamut, False otherwise.
    """
    try:
        rgb = color.to(target_gamut_space)
        vals = np.array(rgb.values)
        return bool(np.all(vals >= -tolerance) and np.all(vals <= 1 + tolerance))
    except Exception:
        # If conversion fails for any reason, it's effectively out of gamut.
        return False

def clip(color: Color, target_gamut_space: str = "srgb") -> Color:
    """
    Clips a color to the target RGB gamut.

    Any channel value less than 0 is set to 0, and any value greater than 1
    is set to 1. This can change the perceived hue of the color.

    Args:
        color: The ChromaFlow Color object to clip.
        target_gamut_space: The name of the target RGB space.

    Returns:
        A new Color object clipped to the target gamut.
    """
    original_space = color.space
    rgb = color.to(target_gamut_space)
    clipped_values = tuple(np.clip(rgb.values, 0.0, 1.0))
    clipped_color = color.__class__(space=target_gamut_space, values=clipped_values)
    return clipped_color.to(original_space)

def oklch_chroma(
    color: Color,
    target_gamut_space: str = "srgb",
    tolerance: float = 1e-7,
    max_iterations: int = 20
) -> Color:
    """
    Maps a color to a target gamut using chroma reduction in Oklch space.

    This is a perceptually superior method that preserves hue and lightness
    by reducing chroma until the color fits inside the target gamut. It uses
    a binary search algorithm for efficiency.

    Args:
        color: The ChromaFlow Color object to map.
        target_gamut_space: The name of the target RGB space.
        tolerance: The precision for finding the gamut boundary.
        max_iterations: The maximum number of search steps.

    Returns:
        A new Color object mapped to the target gamut.
    """
    if in_gamut(color, target_gamut_space):
        return color

    original_space = color.space
    oklch = color.to("oklch")
    L, C, h = oklch.values

    # Binary search for the gamut boundary
    low_chroma = 0.0
    high_chroma = C
    mid_chroma = C
    
    # Start with the out-of-gamut color in the target space
    # This gives us a starting point for checking which channels are out
    rgb_high = oklch.to(target_gamut_space)

    for _ in range(max_iterations):
        mid_chroma = (high_chroma + low_chroma) / 2.0
        
        # Create a test color with the mid-chroma value
        test_color = color.__class__(space="oklch", values=(L, mid_chroma, h))
        
        if in_gamut(test_color, target_gamut_space, tolerance=0.0):
            # The color is in gamut, so this is our new best guess.
            # We try to find a color with even more chroma.
            low_chroma = mid_chroma
        else:
            # The color is out of gamut, so we must reduce chroma.
            high_chroma = mid_chroma

        if (high_chroma - low_chroma) < tolerance:
            break
            
    # The final color uses the last known "in-gamut" chroma value (low_chroma)
    final_oklch = color.__class__(space="oklch", values=(L, low_chroma, h))
    return final_oklch.to(original_space)