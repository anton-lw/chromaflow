# src/chromaflow/cvd.py
from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .color_object import Color

# Correction matrices for Machado et al. (2009) simulation model
# These matrices transform LMS colors for a CVD observer.
LMS_CVD_MATRICES = {
    # Severity 0 = no simulation
    "protanopia": np.array([
        [0.0, 2.02344, -2.52581],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]),
    "deuteranopia": np.array([
        [1.0, 0.0, 0.0],
        [0.494207, 0.0, 1.24827],
        [0.0, 0.0, 1.0]
    ]),
    "tritanopia": np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-0.395913, 0.801109, 0.0]
    ]),
}

# Linear transformation from sRGB-linear to LMS cone space
SRGB_LINEAR_TO_LMS = np.array([
    [17.8824, 43.5161, 4.11935],
    [3.45565, 27.1554, 3.86714],
    [0.0299566, 0.184309, 1.46709]
])

# Inverse transformation
LMS_TO_SRGB_LINEAR = np.linalg.inv(SRGB_LINEAR_TO_LMS)

CVD_Model = Literal["protanopia", "deuteranopia", "tritanopia"]

def simulate_machado(color: Color, deficiency: CVD_Model, severity: float = 1.0) -> Color:
    """
    Simulates color vision deficiency using the Machado (2009) model.

    This model operates in linear sRGB space. The color is converted, transformed
    into an LMS cone space, adjusted for the deficiency, and then transformed
    back.

    Args:
        color: The original color to simulate.
        deficiency: The type of deficiency ('protanopia', 'deuteranopia', 'tritanopia').
        severity: The severity of the deficiency, from 0.0 (none) to 1.0 (full).

    Returns:
        A new Color object representing the simulated perception.
    """
    if severity <= 0.0:
        return color
    if severity > 1.0:
        severity = 1.0
        
    original_space = color.space
    
    # The model operates on linear sRGB values
    srgb_linear = color.to("srgb-linear")
    rgb = np.array(srgb_linear.values)

    # Transform linear RGB to LMS cone space
    lms = np.dot(SRGB_LINEAR_TO_LMS, rgb)

    # Get the appropriate 100% severity matrix
    cvd_matrix = LMS_CVD_MATRICES[deficiency]
    
    # Apply the simulation (linear interpolation between identity and full-deficiency)
    # severity * (lms @ cvd_matrix) + (1 - severity) * (lms @ identity_matrix)
    lms_simulated = (1.0 - severity) * lms + severity * np.dot(cvd_matrix, lms)

    # Transform back to linear sRGB
    rgb_simulated = np.dot(LMS_TO_SRGB_LINEAR, lms_simulated)
    
    # Create the new color and convert it back to the original space
    simulated_color = color.__class__(
        space="srgb-linear",
        values=tuple(rgb_simulated)
    )
    
    # Clip the result to the sRGB gamut before returning
    # This prevents artifacts from the simulation process.
    from . import gamut
    return gamut.clip(simulated_color, "srgb").to(original_space)