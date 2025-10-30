# src/chromaflow/difference.py
from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .color_object import Color
    
def delta_e_1976(color1: Color, color2: Color) -> float:
    """Calculates the CIE Delta E 1976 color difference."""
    lab1 = np.array(color1.to("lab-d65").values)
    lab2 = np.array(color2.to("lab-d65").values)
    return float(np.linalg.norm(lab1 - lab2))

def delta_e_cmc(color1: Color, color2: Color, l: float = 2.0, c: float = 1.0) -> float:
    """Calculates the CMC l:c color difference."""
    L1, a1, b1 = color1.to("lab-d65").values
    L2, a2, b2 = color2.to("lab-d65").values
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    delta_L = L1 - L2
    delta_C = C1 - C2
    delta_a = a1 - a2
    delta_b = b1 - b2
    delta_H_sq = delta_a**2 + delta_b**2 - delta_C**2
    delta_H = np.sqrt(delta_H_sq) if delta_H_sq > 0 else 0.0
    h1 = np.degrees(np.arctan2(b1, a1)) % 360
    S_L = 0.511 if L1 < 16 else (0.040975 * L1) / (1.0 + 0.01765 * L1)
    S_C = (0.0638 * C1) / (1.0 + 0.0131 * C1) + 0.638
    T = 0.56 + abs(0.2 * np.cos(np.radians(h1 + 168))) if 164 <= h1 <= 345 else 0.36 + abs(0.4 * np.cos(np.radians(h1 + 35)))
    F = np.sqrt(C1**4 / (C1**4 + 1900))
    S_H = S_C * (F * T + 1 - F)
    term1, term2, term3 = delta_L / (l * S_L), delta_C / (c * S_C), delta_H / S_H
    return float(np.sqrt(term1**2 + term2**2 + term3**2))

def delta_e_2000(color1: Color, color2: Color, Kl: float = 1, Kc: float = 1, Kh: float = 1) -> float:
    """Calculates the CIE Delta E 2000 color difference."""
    # ... (full implementation as provided before) ...
    L1, a1, b1 = color1.to("lab-d65").values
    L2, a2, b2 = color2.to("lab-d65").values
    C1, C2 = np.sqrt(a1**2 + b1**2), np.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2.0
    G = 0.5 * (1 - np.sqrt(C_bar**7 / (C_bar**7 + 25**7)))
    a1_prime, a2_prime = (1 + G) * a1, (1 + G) * a2
    C1_prime, C2_prime = np.sqrt(a1_prime**2 + b1**2), np.sqrt(a2_prime**2 + b2**2)
    h1_prime, h2_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360, np.degrees(np.arctan2(b2, a2_prime)) % 360
    delta_L_prime, delta_C_prime = L2 - L1, C2_prime - C1_prime
    h_diff = h2_prime - h1_prime
    if C1_prime * C2_prime == 0: delta_h_prime = 0.0
    elif abs(h_diff) <= 180: delta_h_prime = h_diff
    elif h_diff > 180: delta_h_prime = h_diff - 360
    else: delta_h_prime = h_diff + 360
    delta_H_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(delta_h_prime / 2.0))
    L_bar_prime, C_bar_prime = (L1 + L2) / 2.0, (C1_prime + C2_prime) / 2.0
    if C1_prime * C2_prime == 0: h_bar_prime = h1_prime + h2_prime
    elif abs(h1_prime - h2_prime) <= 180: h_bar_prime = (h1_prime + h2_prime) / 2.0
    else: h_bar_prime = (h1_prime + h2_prime + 360) / 2.0 if (h1_prime + h2_prime) < 360 else (h1_prime + h2_prime - 360) / 2.0
    T = 1 - 0.17 * np.cos(np.radians(h_bar_prime - 30)) + 0.24 * np.cos(np.radians(2 * h_bar_prime)) + 0.32 * np.cos(np.radians(3 * h_bar_prime + 6)) - 0.20 * np.cos(np.radians(4 * h_bar_prime - 63))
    delta_theta = 30 * np.exp(-(((h_bar_prime - 275) / 25)**2))
    R_C = 2 * np.sqrt(C_bar_prime**7 / (C_bar_prime**7 + 25**7))
    S_L, S_C, S_H = 1 + (0.015 * (L_bar_prime - 50)**2) / np.sqrt(20 + (L_bar_prime - 50)**2), 1 + 0.045 * C_bar_prime, 1 + 0.015 * C_bar_prime * T
    R_T = -R_C * np.sin(np.radians(2 * delta_theta))
    term1, term2, term3 = delta_L_prime / (Kl * S_L), delta_C_prime / (Kc * S_C), delta_H_prime / (Kh * S_H)
    return float(np.sqrt(term1**2 + term2**2 + term3**2 + R_T * term2 * term3))

def delta_e_jz(color1: Color, color2: Color) -> float:
    """Calculates the Delta E Jz color difference."""
    jzazbz1 = np.array(color1.to("jzazbz").values)
    jzazbz2 = np.array(color2.to("jzazbz").values)
    return float(np.linalg.norm(jzazbz1 - jzazbz2))