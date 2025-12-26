"""WaveStack Hybrid research package."""

from importlib import metadata

try:
    __version__ = metadata.version("wavestack-hybrid")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
