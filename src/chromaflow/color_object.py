# src/chromaflow/color_object.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast, overload

import numpy as np

from . import cvd, difference, gamut, utils
from .core.spaces import get_space
from .exceptions import ConversionPathError
from .pathfinder import find_conversion_path, get_conversion_function


@dataclass(frozen=True)
class Color:
    """
    An immutable representation of a color in a specific color space.
    """

    space: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        # Validate that the space is registered upon creation.
        get_space(self.space)

    def __repr__(self) -> str:
        # Format values to a reasonable precision for display
        vals_str = ", ".join(f"{v:.4f}" for v in self.values)
        return f"Color(space='{self.space}', values=({vals_str}))"

    @classmethod
    def from_hex(cls, hex_string: str) -> Color:
        """Creates an sRGB color from a hex string."""
        return cls(space="srgb", values=utils.parse_hex(hex_string))

    def to(self, target_space: str) -> Color:
        """
        Converts the color to a different color space.

        This method uses a graph search algorithm to find the shortest
        possible conversion path between the color's current space and the
        target space.

        Args:
            target_space: The name of the destination color space.

        Returns:
            A new Color object in the target space.

        Raises:
            ConversionPathError: If no conversion path exists between the spaces.
        """
        target_space = target_space.lower()
        if self.space == target_space:
            return self

        full_path = find_conversion_path(self.space, target_space)

        if full_path is None:
            raise ConversionPathError(
                f"No conversion path found from '{self.space}' to '{target_space}'."
            )

        current_values = np.array(self.values).reshape(1, -1)

        for i in range(len(full_path) - 1):
            step_from = full_path[i]
            step_to = full_path[i + 1]
            conversion_func = get_conversion_function(step_from, step_to)
            current_values = conversion_func(current_values)

        # After all conversions, the result is still a 2D array (e.g., [[L, a, b]]).
        # We flatten it back to 1D before converting to a tuple.
        final_values_flat = current_values.flatten()
        final_values = tuple(cast(tuple[float, ...], final_values_flat.tolist()))
        return Color(space=target_space, values=final_values)

    # --- Gamut Methods ---
    def in_gamut(
        self,
        target_gamut_space: str = "srgb",
        tolerance: float = 1e-5,
    ) -> bool:
        """Checks if the color is within the specified RGB gamut."""
        return gamut.in_gamut(self, target_gamut_space, tolerance)

    @overload
    def to_gamut(
        self,
        target_gamut_space: str = "srgb",
        method: Literal["clip"] = "clip",
    ) -> Color: ...

    @overload
    def to_gamut(
        self,
        target_gamut_space: str = "srgb",
        method: Literal["oklch-chroma"] = "oklch-chroma",
    ) -> Color: ...

    def to_gamut(
        self,
        target_gamut_space: str = "srgb",
        method: str = "oklch-chroma",
    ) -> Color:
        """
        Maps the color to be within the target RGB gamut.

        Args:
            target_gamut_space: The target RGB space (e.g., 'srgb').
            method: The mapping algorithm to use.
                - 'clip': Simple clipping of RGB values. Fast but can change hue.
                - 'oklch-chroma': Perceptually superior chroma reduction in Oklch.
        """
        if method == "clip":
            return gamut.clip(self, target_gamut_space)
        if method == "oklch-chroma":
            return gamut.oklch_chroma(self, target_gamut_space)
        raise ValueError(f"Unknown gamut mapping method: '{method}'")

    # --- Color Difference Methods ---
    def delta_e(
        self,
        other: Color,
        method: Literal["1976", "2000", "cmc", "jz"] = "2000",
    ) -> float:
        """
        Calculates the perceptual color difference to another color.

        Args:
            other: The color to compare against.
            method: The Delta E formula to use ('1976', '2000', 'cmc', 'jz').
        """
        if method == "1976":
            return difference.delta_e_1976(self, other)
        if method == "2000":
            return difference.delta_e_2000(self, other)
        if method == "cmc":
            return difference.delta_e_cmc(self, other)
        if method == "jz":
            return difference.delta_e_jz(self, other)
        raise ValueError(f"Unknown Delta E method: '{method}'")

    def simulate_cvd(self, deficiency: cvd.CVD_Model, severity: float = 1.0) -> Color:
        """
        Simulates how this color would be perceived by a person with color
        vision deficiency.
        """
        return cvd.simulate_machado(self, deficiency, severity)

    def lighten(self, amount: float) -> Color:
        oklch = self.to("oklch")
        L, C, h = oklch.values
        new_L = np.clip(L + amount, 0.0, 1.0)  # Oklab L is 0-1
        new_oklch = Color("oklch", (new_L, C, h))
        return new_oklch.to(self.space)

    def darken(self, amount: float) -> Color:
        return self.lighten(-amount)

    def saturate(self, amount: float) -> Color:
        oklch = self.to("oklch")
        L, C, h = oklch.values
        new_C = max(0.0, C + amount)
        new_oklch = Color("oklch", (L, new_C, h))
        return new_oklch.to(self.space)

    def desaturate(self, amount: float) -> Color:
        return self.saturate(-amount)

    def rotate_hue(self, degrees: float) -> Color:
        oklch = self.to("oklch")
        L, C, h = oklch.values
        new_h = (h + degrees) % 360
        new_oklch = Color("oklch", (L, C, new_h))
        return new_oklch.to(self.space)
