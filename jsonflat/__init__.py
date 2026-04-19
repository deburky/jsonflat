"""jsonflat -- flatten nested JSON into DataFrames with controlled depth."""

from jsonflat.core import flatten, normalize_json, to_dataframe

__all__ = ["flatten", "normalize_json", "to_dataframe"]
