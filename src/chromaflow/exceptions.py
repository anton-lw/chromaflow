# src/chromaflow/exceptions.py

class ChromaFlowError(Exception):
    """Base exception for all errors raised by ChromaFlow."""
    pass

class ColorSpaceError(ChromaFlowError):
    """Raised for errors related to color space definitions or lookups."""
    pass

class ConversionPathError(ChromaFlowError):
    """Raised when no conversion path can be found between two color spaces."""
    pass

class BackendConfigurationError(ChromaFlowError):
    """Raised when there is an issue with backend configuration, such as a
    missing optional dependency or an unknown backend name."""
    pass

class GamutError(ChromaFlowError):
    """Raised for operations involving out-of-gamut colors."""
    pass