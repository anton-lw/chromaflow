# src/chromaflow/__init__.py

__version__ = "0.3.0"

from . import config
from .color_object import Color
from .exceptions import (
    BackendConfigurationError,
    ChromaFlowError,
    ColorSpaceError,
    ConversionPathError,
    GamutError,
)

__all__ = [
    "Color",
    "config",
    "ChromaFlowError",
    "ColorSpaceError",
    "ConversionPathError",
    "BackendConfigurationError",
    "GamutError",
]
