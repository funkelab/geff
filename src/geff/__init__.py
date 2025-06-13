from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("geff")
except PackageNotFoundError:
    __version__ = "uninstalled"

from .utils import validate
from .metadata_schema import GeffMetadata
from .networkx.io import read_nx, write_nx

__all__ = ["validate", "GeffMetadata", "read_nx", "write_nx"]