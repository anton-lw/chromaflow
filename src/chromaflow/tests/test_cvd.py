# tests/test_cvd.py
import numpy as np
import pytest

from chromaflow import Color, cvd


@pytest.mark.parametrize(
    ("deficiency", "expected"),
    [
        # Published by Machado et al. / UFRGS for full-severity simulation.
        (
            "protanopia",
            np.array(
                [
                    [0.152286, 1.052583, -0.204868],
                    [0.114503, 0.786281, 0.099216],
                    [-0.003882, -0.048116, 1.051998],
                ]
            ),
        ),
        (
            "deuteranopia",
            np.array(
                [
                    [0.367322, 0.860646, -0.227968],
                    [0.280085, 0.672501, 0.047413],
                    [-0.01182, 0.04294, 0.968881],
                ]
            ),
        ),
        (
            "tritanopia",
            np.array(
                [
                    [1.255528, -0.076749, -0.178779],
                    [-0.078411, 0.930809, 0.147602],
                    [0.004733, 0.691367, 0.3039],
                ]
            ),
        ),
    ],
)
def test_machado_reference_matrix_matches_published_values(
    deficiency: cvd.CVD_Model, expected: np.ndarray
) -> None:
    matrix = cvd._matrix_cvd_machado(deficiency, 1.0)
    assert np.allclose(matrix, expected, atol=1e-6)


def test_machado_interpolates_between_severity_steps() -> None:
    matrix = cvd._matrix_cvd_machado("protanopia", 0.15)
    expected = np.array(
        [
            [0.7954665, 0.258455, -0.053921],
            [0.040591, 0.9371565, 0.0222535],
            [-0.003904, -0.002886, 1.00679],
        ]
    )
    assert np.allclose(matrix, expected, atol=1e-6)


def test_cvd_simulation_achromatic() -> None:
    """Achromatic colors (grays) should not change under CVD simulation."""
    gray = Color.from_hex("#808080")

    simulated = gray.simulate_cvd("protanopia")

    assert np.allclose(gray.values, simulated.values, atol=1e-5)


def test_cvd_simulation_severity() -> None:
    """Tests that severity=0 gives the original color and severity=1 is different."""
    red = Color.from_hex("#FF0000")

    sev_0 = red.simulate_cvd("deuteranopia", severity=0.0)
    sev_1 = red.simulate_cvd("deuteranopia", severity=1.0)

    assert np.allclose(red.values, sev_0.values)
    assert not np.allclose(red.values, sev_1.values)
