"""jsonflat -- flatten nested JSON into DataFrames with controlled depth."""

from importlib.metadata import PackageNotFoundError, version

from jsonflat.core import aio, flatten, normalize_json, to_dataframe, unflatten

try:
    __version__ = version("jsonflat")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["__version__", "aio", "flatten", "normalize_json", "to_dataframe", "unflatten"]
