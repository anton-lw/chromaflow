# tests/test_utils.py
import pytest

from chromaflow import utils


def test_parse_hex_valid() -> None:
    """Tests parsing of valid 6-digit and 3-digit hex strings."""
    r = 204 / 255
    assert utils.is_close(utils.parse_hex("#CC331A")[0], r)
    assert utils.is_close(utils.parse_hex("CC331A")[0], r)
    assert utils.is_close(
        utils.parse_hex("#C31")[0], 12 * 17 / 255
    )  # C -> CC, 3 -> 33, etc.


def test_parse_hex_invalid() -> None:
    """Tests that invalid hex strings raise ValueErrors."""
    with pytest.raises(ValueError):
        utils.parse_hex("#12345")
    with pytest.raises(ValueError):
        utils.parse_hex("GGHHII")
    with pytest.raises(ValueError):
        utils.parse_hex("#1234567")


def test_is_close() -> None:
    """Tests the floating point comparison utility."""
    assert utils.is_close(1.0, 1.0000000001)
    assert not utils.is_close(1.0, 1.0001)
