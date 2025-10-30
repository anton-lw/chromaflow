# src/chromaflow/core/constants.py
import numpy as np
from numpy.typing import NDArray

ILLUMINANT_D65_XY = np.array([0.3127, 0.3290])
ILLUMINANT_D65_XYZ = np.array([0.95047, 1.0, 1.08883])
CIE_E = 216 / 24389
CIE_K = 24389 / 27

def _derive_xyz_matrix(primaries_xy: NDArray, whitepoint_xy: NDArray) -> NDArray:
    """
    Derives the RGB to XYZ conversion matrix from xy primaries and a whitepoint.
    """
    # Whitepoint XYZ, normalized to Y=1
    xw, yw = whitepoint_xy
    xyz_w = np.array([xw / yw, 1.0, (1 - xw - yw) / yw])
    
    # Primaries in a 3x3 matrix
    r, g, b = primaries_xy
    xr, yr = r
    xg, yg = g
    xb, yb = b
    
    M = np.array([
        [xr, xg, xb],
        [yr, yg, yb],
        [1 - xr - yr, 1 - xg - yg, 1 - xb - yb]
    ])
    
    # Calculate cone response domain scalars
    S = np.linalg.inv(M) @ xyz_w
    Sr, Sg, Sb = S
    
    # Scale the matrix M by the scalars
    xyz_matrix = M * np.array([Sr, Sg, Sb])
    
    return xyz_matrix

# sRGB (Rec. 709) constants
SRGB_PRIMARIES_XY = np.array([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]])
# This is the IEC 61966-2-1 standard matrix.
SRGB_TO_XYZ_MATRIX = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
XYZ_TO_SRGB_MATRIX = np.linalg.inv(SRGB_TO_XYZ_MATRIX)

# Display P3 constants 
P3_D65_PRIMARIES_XY = np.array([[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]])
# For other spaces, we derive the matrix programmatically.
P3_TO_XYZ_MATRIX = _derive_xyz_matrix(P3_D65_PRIMARIES_XY, ILLUMINANT_D65_XY)
XYZ_TO_P3_MATRIX = np.linalg.inv(P3_TO_XYZ_MATRIX)

# Rec. 2020 constants
REC2020_PRIMARIES_XY = np.array([[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]])
REC2020_TO_XYZ_MATRIX = _derive_xyz_matrix(REC2020_PRIMARIES_XY, ILLUMINANT_D65_XY)
XYZ_TO_REC2020_MATRIX = np.linalg.inv(REC2020_TO_XYZ_MATRIX)

# Adobe RGB (1998) constants
ADOBE_RGB_PRIMARIES_XY = np.array([[0.64, 0.33], [0.21, 0.71], [0.15, 0.06]])
ADOBE_RGB_TO_XYZ_MATRIX = _derive_xyz_matrix(ADOBE_RGB_PRIMARIES_XY, ILLUMINANT_D65_XY)
XYZ_TO_ADOBE_RGB_MATRIX = np.linalg.inv(ADOBE_RGB_TO_XYZ_MATRIX)

# Oklab constants
OKLAB_M1 = np.array([
    [+0.8189330101, +0.3618667424, -0.1288597137],
    [+0.0329845436, +0.9293118715, +0.0361456387],
    [+0.0482003018, +0.2643662691, +0.6338517070],
])
OKLAB_M1_INV = np.linalg.inv(OKLAB_M1)

OKLAB_M2 = np.array([
    [+0.2104542553, +0.7936177850, -0.0040720468],
    [+1.9779984951, -2.4285922050, +0.4505937099],
    [+0.0259040371, +0.7827717662, -0.8086757660],
])
OKLAB_M2_INV = np.linalg.inv(OKLAB_M2)

# CAM16 constants
CAT16_M = np.array([
    [0.7328, 0.4296, -0.1624],
    [-0.7036, 1.6975, 0.0061],
    [0.0030, 0.0136, 0.9834],
])
CAT16_M_INV = np.linalg.inv(CAT16_M)
CAM16_VC_AVG = {'Y_b': 20.0, 'L_A': 64.0 / np.pi / 5.0}

# Jzazbz Constants (Safdar et al. 2017)
JZAZBZ_B = 1.15
JZAZBZ_G = 0.66
JZAZBZ_C1 = 3424 / 4096
JZAZBZ_C2 = 2413 / 128
JZAZBZ_C3 = 2392 / 128
JZAZBZ_P1 = (1.6295499532821566e-11) * (10_000 ** (705 / 4096))
JZAZBZ_P2 = (3.050395210113353e-22) * (10_000 ** (-29.5 / 4096))
JZAZBZ_P3 = 705 / 4096
JZAZBZ_P4 = (1.723043234908023e-8) * (10_000 ** (705 / 4096))

M_HPE_TO_XYZ = np.array([
    [0.41478972, 0.17941231, 0.01439368],
    [0.22413217, 0.72338115, 0.07633722],
    [0.03858022, 0, 1.11184922]
])
M_XYZ_TO_HPE = np.linalg.inv(M_HPE_TO_XYZ)

JZAZBZ_I_CONST = (
    1.0 + JZAZBZ_P1 * (10_000 ** JZAZBZ_P3) /
    (1.0 + JZAZBZ_P4 * (10_000 ** JZAZBZ_P3))
)