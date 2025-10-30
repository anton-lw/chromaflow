# ChromaFlow

<div align="center">

**A modern, high-performance Python library for seamless perceptual color conversions.**

</div>

---

ChromaFlow is a Python color science library built for both ease of use and extreme performance. It bridges the gap between developer-friendly APIs and scientifically rigorous, high-throughput computation. Whether you're a web developer managing CSS colors, a data scientist creating vibrant visualizations, or a color scientist researching perceptual models, `ChromaFlow` provides a unified, powerful, and intuitive toolkit.

## Features

-   An intuitive, chainable API that makes color manipulation a breeze.
-   Achieve up to 9x performance gains over standard NumPy with optional JAX and Numba backends for processing large datasets and images.
-   Seamlessly convert between sRGB, Display P3, CIELAB, Oklab, Jzazbz, and more, with a dynamic pathfinder that always finds the shortest conversion route.
-   Go beyond simple clipping. Use our perceptual gamut mapping (`oklch-chroma`) to bring out-of-gamut colors into range while preserving their original hue and lightness.
-   Manipulation methods (lighten, saturate, rotate hue) operate in the perceptually uniform Oklch space by default, producing visually intuitive results.
-   Essential color science tools, including built-in functions for:
    -   Color difference (Delta E): all major standards (1976, CMC, 2000, Jz).
    -   Accessibility: simulate color vision deficiencies (Protanopia, Deuteranopia, Tritanopia).
-   Visualize color swatches and chromaticity diagrams with Matplotlib to gain instant insight into your colors and gamuts.

## Installation guide

As this library is not available on PyPI, you can install it directly from this GitHub repository using `pip`.

### Prerequisites

-   Python 3.9+
-   `pip`
-   `git` (must be installed and available in your system's PATH)

---

### Standard installation

This method installs the library just like any other package. This is the best option for using ChromaFlow as a dependency in your projects.

You can install the latest version from the `main` branch directly:

```bash
pip install git+https://github.com/anton-lw/chromaflow.git
```

#### Installing optional dependencies

To include the high-performance backends and plotting utilities, you must specify the "extras" in the URL.

```bash
# For JAX and Numba backends
pip install "git+https://github.com/anton-lw/chromaflow.git#egg=chromaflow[performance]"

# For plotting capabilities
pip install "git+https://github.com/anton-lw/chromaflow.git#egg=chromaflow[plotting]"

# To get everything at once
pip install "git+https://github.com/anton-lw/chromaflow.git#egg=chromaflow[performance,plotting]"
```
**Note:** The `#egg=chromaflow[...]` part is crucial for `pip` to correctly parse the extras.

---

### Editable installation (For development)

If you plan to contribute to `ChromaFlow` or want to make local changes, you should clone the repository and install it in "editable" mode. This links the installed package directly to your source code.

```bash
# 1. Clone the repository to your local machine
git clone https://github.com/anton-lw/chromaflow.git
cd chromaflow

# 2. Install in editable mode with all optional dependencies
pip install -e .[performance,plotting,dev]
```
Now, any changes you make in the `src/chromaflow` directory will be immediately available when you `import chromaflow` in Python.

---

## Quickstart

ChromaFlow is designed to be immediately useful.

```python
import chromaflow as cf

# 1. Create a color from a hex string
brand_color = cf.Color.from_hex("#CC331A")
print(f"Original Color: {brand_color}")

# 2. Seamlessly convert to any supported space
lab_color = brand_color.to("lab-d65")
print(f"CIELAB version: {lab_color}")

# 3. Manipulate perceptually and chain methods
modified_color = (
    brand_color.lighten(0.1)
               .saturate(0.05)
               .rotate_hue(-15)
)
print(f"Modified Color: {modified_color.to_string(hex=True)}")


# 4. Check if a color is in the sRGB gamut
p3_green = cf.Color("p3-d65", (0, 1, 0))
print(f"\nIs vibrant P3 green in sRGB gamut? {p3_green.in_gamut('srgb')}")

# 5. Bring it into gamut while preserving its appearance
srgb_safe_green = p3_green.to_gamut("srgb", method="oklch-chroma")
print(f"Closest sRGB equivalent: {srgb_safe_green.to_string(hex=True)}")

# 6. Check for accessibility issues
simulated_color = brand_color.simulate_cvd("deuteranopia")
print(f"How a user with Deuteranopia might see our brand color: {simulated_color.to_string(hex=True)}")

# 7. Compare two colors scientifically
color1 = cf.Color.from_hex("#808080")
color2 = cf.Color.from_hex("#828282")
delta_e = color1.delta_e(color2, method="2000")
print(f"\nPerceptual difference between grays: {delta_e:.2f} (imperceptible)")
```

## High-performance backends

By default, `ChromaFlow` uses a pure NumPy backend. For large-scale data processing, you can get a massive speedup by switching to JAX or Numba.

```python
import chromaflow as cf
import numpy as np

# Create a large array of colors (e.g., a 1MP image)
image_data = np.random.rand(1000 * 1000, 3)
image_colors = cf.Color("srgb", image_data)

# Use a context manager to temporarily switch to the JAX backend
with cf.config.backend("jax"):
    # This entire conversion pipeline is JIT-compiled and highly optimized
    lab_image = image_colors.to("lab-d65")
    # This will be ~8x faster than the default NumPy backend!
```
*Note: You must have JAX installed (see installation instructions) to use this feature.*

## Examples

## Creating and converting colors
Define a corporate brand color and find its representation in CIELAB (for print) and Oklab (for UI design).

```python
# Our brand color is a deep teal, defined in sRGB hex
brand_teal = cf.Color.from_hex("#008080")

# Convert to CIELAB for print specifications
lab_teal = brand_teal.to("lab-d65")

# Convert to Oklab for modern UI design (better perceptual uniformity)
oklab_teal = brand_teal.to("oklab")

print(f"sRGB Hex: {brand_teal.to_string(hex=True)}")
print(f"CIELAB:   {lab_teal}")
print(f"Oklab:    {oklab_teal}")
```

## Creating a perceptually uniform palette
A common design task is to create a palette where colors feel equally spaced visually. This is impossible in sRGB/HSL but easy in Oklch.
In this example, our goal is to create a 5-color analogous palette starting from our brand teal by rotating the hue.

```python
brand_teal = cf.Color.from_hex("#008080")

# Convert to Oklch (Lightness, Chroma, Hue) to manipulate perceptual properties
oklch_teal = brand_teal.to("oklch")

# Create a palette by rotating the hue by 20 degrees for each new color
# We keep Lightness and Chroma constant for a harmonious feel.
palette = [
    oklch_teal.rotate_hue(-40),
    oklch_teal.rotate_hue(-20),
    oklch_teal,
    oklch_teal.rotate_hue(20),
    oklch_teal.rotate_hue(40),
]

# Convert back to sRGB for display and get hex codes
hex_palette = [p.to_string(hex=True) for p in palette]
print("Generated Palette Hex Codes:", hex_palette)

plot_color_swatch(palette, labels=hex_palette)
plt.show()
```
## Checking and fixing out-of-gamut colors
When converting from a large color space (like Display P3) to a smaller one (sRGB), colors can be "out of gamut." `ChromaFlow` can both detect and fix this gracefully.
Our goal in this example is to take a vibrant P3 green that cannot be displayed on sRGB monitors and map it to the closest possible sRGB color.

```python
# A vibrant green, defined in the Display P3 color space
# This color is impossible to show accurately on a standard sRGB screen.
p3_green = cf.Color("p3-d65", (0, 1, 0))

# 1. Check if it's in the sRGB gamut
is_in_srgb = p3_green.in_gamut("srgb")
print(f"Is P3 Green in the sRGB gamut? {is_in_srgb}")

# 2. Naively clip the color to the sRGB gamut (fast, but can change hue)
clipped_green = p3_green.to_gamut("srgb", method="clip")

# 3. Perceptually map the color by reducing chroma (slower, but preserves hue)
mapped_green = p3_green.to_gamut("srgb", method="oklch-chroma")

print(f"\nOriginal P3 Hex (in sRGB): {p3_green.to_string(hex=True)}")
print(f"Clipped sRGB Hex:        {clipped_green.to_string(hex=True)}")
print(f"Mapped sRGB Hex:         {mapped_green.to_string(hex=True)}")

plot_color_swatch(
    [p3_green, clipped_green, mapped_green],
    labels=["Original P3\n(inaccurate)", "Clipped to sRGB", "Mapped to sRGB"]
)
plt.show()
```
## Simulating color vision deficiency
Ensure your designs are accessible to everyone by simulating how they appear to people with different types of color vision deficiency (CVD).
The goal in this example is to check if a red error message on a green background is readable for someone with Deuteranopia (red-green color blindness).

```python
# A typical "error" and "success" color pair
error_red = cf.Color.from_hex("#D55E00") # A reddish orange
success_green = cf.Color.from_hex("#009E73") # A bluish green

# Simulate how this pair looks to someone with Deuteranopia
simulated_red = error_red.simulate_cvd("deuteranopia")
simulated_green = success_green.simulate_cvd("deuteranopia")

# Display the original and simulated swatches
plot_color_swatch([error_red, success_green], labels=["Original Red", "Original Green"])
plt.title("Original colors")
plt.show()

plot_color_swatch([simulated_red, simulated_green], labels=["Simulated Red", "Simulated Green"])
plt.title("Deuteranopia simulation")
plt.show()
```
## Finding perceptual color difference (Delta E)
Don't rely on your eyes alone! Quantify the difference between two colors using Delta E. A value < 2.0 is often considered imperceptible.
The goal in this example is to determine if two shades of gray are "perceptibly different".

```python
gray1 = cf.Color.from_hex("#808080")
gray2 = cf.Color.from_hex("#828282")
subtly_different_gray = cf.Color.from_hex("#8F8F8F")

# Calculate Delta E 2000, the modern standard for perceptual difference
delta_e_small = gray1.delta_e(gray2, method="2000")
delta_e_large = gray1.delta_e(subtly_different_gray, method="2000")

print(f"Difference between #808080 and #828282: {delta_e_small:.2f} (likely imperceptible)")
print(f"Difference between #808080 and #8F8F8F: {delta_e_large:.2f} (clearly perceptible)")
```
## Visualizing color gamuts
Understand the limitations of your target display by plotting the footprint of different color spaces on a chromaticity diagram.
The goal in this example is to visualize how much larger the Display P3 gamut is compared to standard sRGB.

```python
# Use the built-in plotting utility
plot_chromaticity_diagram(
    gamut_footprints=["srgb", "p3-d65"],
    show_spectral_locus=True,
    show_whitepoints=True
)
plt.title("sRGB vs. Display P3 Gamut Comparison")
plt.show()
```

## Contributing

Contributions are welcome! Whether it's adding a new color space, improving documentation, or fixing a bug, we'd love your help. Please see our contributing guide for more details.

## License

`ChromaFlow` is licensed under the MIT License. See the `LICENSE` file for details.
