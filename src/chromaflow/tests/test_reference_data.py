from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
import pytest
import tomllib

import chromaflow
from chromaflow import Color, config, functional
from chromaflow.exceptions import BackendConfigurationError
from chromaflow.plotting import _load_spectral_locus


def test_package_version_matches_pyproject() -> None:
    project_root = Path(__file__).resolve().parents[3]
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text())
    assert chromaflow.__version__ == pyproject["project"]["version"]


def test_display_p3_linear_space_round_trip() -> None:
    color = Color("p3-d65", (0.25, 0.5, 0.75))
    linear = color.to("p3-d65-linear")
    round_trip = linear.to("p3-d65")
    assert np.allclose(round_trip.values, color.values, atol=1e-12)


def test_adobe_rgb_red_matches_reference_xyz() -> None:
    adobe_red = Color("adobe-rgb", (1.0, 0.0, 0.0))
    xyz = adobe_red.to("xyz-d65")
    expected_xyz = np.array([0.57667, 0.29734, 0.02703])
    assert np.allclose(xyz.values, expected_xyz, atol=5e-5)


def test_jzazbz_matches_reference_values() -> None:
    xyz = Color("xyz-d65", (0.20654008, 0.12197225, 0.05136952))
    jzazbz = xyz.to("jzazbz")
    expected = np.array([0.00535048, 0.00924302, 0.00526007])
    assert np.allclose(jzazbz.values, expected, atol=1e-6)


def test_jzazbz_reference_inverse_matches_xyz() -> None:
    jzazbz = Color("jzazbz", (0.00535048, 0.00924302, 0.00526007))
    xyz = jzazbz.to("xyz-d65")
    expected = np.array([0.20654008, 0.12197225, 0.05136952])
    assert np.allclose(xyz.values, expected, atol=1e-6)


def test_srgb_oetf_handles_negative_values_without_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        converted = Color("srgb-linear", (-0.1, 0.5, 0.5)).to("srgb")

    assert np.isfinite(converted.values).all()
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_optional_backends_are_imported_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import_module = getattr(functional, "import_module")
    functional._MODULE_CACHE.clear()

    def fake_import_module(name: str) -> object:
        if name == "chromaflow.backends.jax_backend":
            raise ImportError("jax unavailable")
        return original_import_module(name)

    monkeypatch.setattr(functional, "import_module", fake_import_module)

    with config.backend("jax"):
        with pytest.raises(BackendConfigurationError):
            functional.srgb_to_srgb_linear(np.array([[0.2, 0.4, 0.6]]))


def test_spectral_locus_uses_official_cie_samples() -> None:
    data = _load_spectral_locus()
    assert data.shape == (471, 2)
    assert np.allclose(data[0], np.array([0.17556, 0.00529]), atol=1e-5)
    assert np.allclose(data[195], np.array([0.33736, 0.65885]), atol=1e-5)
    assert np.allclose(data[-1], np.array([0.73469, 0.26531]), atol=1e-5)


def test_spectral_locus_file_has_expected_wavelengths() -> None:
    path = Path(__file__).resolve().parents[1] / "data" / "cie_1931_2deg_locus.csv"
    with path.open(newline="") as handle:
        rows = [
            (int(row[0]), float(row[1]), float(row[2]), float(row[3]))
            for row in csv.reader(line for line in handle if not line.startswith("#"))
        ]

    assert rows[0] == (360, 0.17556, 0.00529, 0.81915)
    assert rows[195] == (555, 0.33736, 0.65885, 0.00379)
    assert rows[-1] == (830, 0.73469, 0.26531, 0.0)
