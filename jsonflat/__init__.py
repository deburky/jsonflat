"""jsonflat -- flatten nested JSON into DataFrames with controlled depth."""

from jsonflat.core import aio, flatten, normalize_json, to_dataframe

__all__ = ["aio", "flatten", "normalize_json", "to_dataframe"]
