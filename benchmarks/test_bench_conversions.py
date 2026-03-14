# benchmarks/test_bench_conversions.py
import numpy as np
import pytest

import chromaflow
from chromaflow import config

IMAGE_DATA_SRGB = np.random.rand(1000 * 1000, 3)


@pytest.mark.parametrize("backend", ["numpy", "jax", "numba"])
def test_benchmark_srgb_to_lab_functional(benchmark, backend):
    if backend != "numpy":
        try:
            __import__(backend)
        except ImportError:
            pytest.skip(f"Backend '{backend}' not installed.")

    def convert_image():
        with config.backend(backend):
            linear = chromaflow.functional.srgb_to_srgb_linear(IMAGE_DATA_SRGB)
            xyz = chromaflow.functional.srgb_linear_to_xyz_d65(linear)
            lab = chromaflow.functional.xyz_d65_to_lab_d65(xyz)
            return lab

    result = benchmark(convert_image)

    assert result.shape == IMAGE_DATA_SRGB.shape
    assert not np.isnan(result).any()


@pytest.mark.parametrize("backend", ["numpy", "jax", "numba"])
def test_benchmark_srgb_to_oklab_functional(benchmark, backend):
    if backend != "numpy":
        try:
            __import__(backend)
        except ImportError:
            pytest.skip(f"Backend '{backend}' not installed.")

    def convert_image():
        with config.backend(backend):
            linear = chromaflow.functional.srgb_to_srgb_linear(IMAGE_DATA_SRGB)
            xyz = chromaflow.functional.srgb_linear_to_xyz_d65(linear)
            oklab = chromaflow.functional.xyz_d65_to_oklab(xyz)
            return oklab

    result = benchmark(convert_image)

    assert result.shape == IMAGE_DATA_SRGB.shape
    assert not np.isnan(result).any()
