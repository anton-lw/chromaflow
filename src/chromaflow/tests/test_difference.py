# tests/test_difference.py
import pytest
from chromaflow import Color

C1 = Color("lab-d65", (50, 20, -30))
C2 = Color("lab-d65", (52, 25, -28))

def test_delta_e_1976():
    de = C1.delta_e(C2, method="1976")
    # Expected: sqrt((52-50)^2 + (25-20)^2 + (-28 - -30)^2) = sqrt(4 + 25 + 4) = sqrt(33)
    assert pytest.approx(de) == 5.74456

def test_delta_e_2000():
    # Test against a known value from a color science calculator
    c1 = Color("lab-d65", (50, 2.6772, -79.7751))
    c2 = Color("lab-d65", (50, 0, -82.7485))
    de = c1.delta_e(c2, method="2000")
    assert pytest.approx(de, abs=1e-4) == 2.0425
    
    # A color against itself should have zero difference
    assert C1.delta_e(C1, method="2000") == 0.0