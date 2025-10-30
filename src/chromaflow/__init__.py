# src/chromaflow/__init__.py

__version__ = "0.2.0"

from .color_object import Color
from . import config
from .exceptions import (
    ChromaFlowError,
    ColorSpaceError,
    ConversionPathError,
    BackendConfigurationError,
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