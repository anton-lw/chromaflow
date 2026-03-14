# src/chromaflow/utils.py
import re


def parse_hex(hex_string: str) -> tuple[float, float, float]:
    """
    Parses a 3 or 6-digit hex color string (e.g., "#RGB" or "#RRGGBB")
    into a tuple of normalized (0-1) float values.

    Args:
        hex_string: The hex color string, with or without the leading '#'.

    Returns:
        A tuple of (r, g, b) values between 0.0 and 1.0.

    Raises:
        ValueError: If the hex string is invalid.
    """
    hex_string = hex_string.lstrip("#")

    if len(hex_string) == 3:
        # Expand 3-digit hex to 6-digit, e.g., "F0C" -> "FF00CC"
        hex_string = "".join(c * 2 for c in hex_string)

    if len(hex_string) != 6 or not re.match(r"^[0-9a-fA-F]{6}$", hex_string):
        raise ValueError(f"Invalid hex color string format: '{hex_string}'")

    r = int(hex_string[0:2], 16)
    g = int(hex_string[2:4], 16)
    b = int(hex_string[4:6], 16)

    return r / 255.0, g / 255.0, b / 255.0


def is_close(
    a: float,
    b: float,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-9,
) -> bool:
    """
    Checks if two floats are close to each other, useful for testing.
    """
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
