# tests/test_cvd.py
import pytest
from chromaflow import Color
import numpy as np

def test_cvd_simulation_achromatic():
    """Achromatic colors (grays) should not change under CVD simulation."""
    gray = Color.from_hex("#808080")
    
    simulated = gray.simulate_cvd("protanopia")
    
    assert np.allclose(gray.values, simulated.values, atol=1e-5)

def test_cvd_simulation_severity():
    """Tests that severity=0 gives the original color and severity=1 is different."""
    red = Color.from_hex("#FF0000")
    
    sev_0 = red.simulate_cvd("deuteranopia", severity=0.0)
    sev_1 = red.simulate_cvd("deuteranopia", severity=1.0)
    
    assert np.allclose(red.values, sev_0.values)
    assert not np.allclose(red.values, sev_1.values)