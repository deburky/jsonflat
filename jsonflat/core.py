"""
jsonflat -- flatten nested JSON into DataFrames with controlled depth.

Usage:
    from jsonflat import flatten, normalize_json, to_dataframe

    flat = flatten(data, max_nesting=3)           # single flat dict
    tables = normalize_json(data, max_nesting=3)  # parent + child tables
    df = to_dataframe(data, max_nesting=3)        # straight to DataFrame
"""

from __future__ import annotations

from typing import Any


def flatten(
    row: dict[str, Any],
    max_nesting: int | None = 3,
    _depth: int = 0,
    _path: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Flatten a nested dict into a single-level dict with __ separators.

    Args:
        row: nested dict to flatten
        max_nesting: max depth before storing as JSON blob (None = unlimited)
        _depth: current recursion depth (internal)
        _path: current key path (internal)

    Returns:
        Flat dict with __ separated keys. Nested dicts/lists beyond
        max_nesting are preserved as-is (JSON blobs).
    """
    flat: dict[str, Any] = {}
    stop = max_nesting is not None and _depth >= max_nesting

    for k, v in row.items():
        key = "__".join((*_path, k)) if _path else k

        if isinstance(v, dict) and not stop:
            flat |= flatten(v, max_nesting, _depth + 1, (*_path, k))
        else:
            flat[key] = v

    return flat


def normalize_json(
    data: dict[str, Any] | list[dict[str, Any]],
    max_nesting: int | None = 3,
    root_name: str = "main",
    separator: str = ".",
) -> dict[str, list[dict[str, Any]]]:
    """Normalize JSON into parent + child tables.

    Args:
        data: single record or list of records
        max_nesting: max depth before storing as JSON blob (None = unlimited)
        root_name: name for the root table
        separator: separator for child table names (e.g. "." -> "policy_output.offers")

    Returns:
        Dict mapping table_name -> list of row dicts.
    """
    if isinstance(data, dict):
        data = [data]

    tables: dict[str, list[dict[str, Any]]] = {root_name: []}

    for record in data:
        flat: dict[str, Any] = {}
        lists: dict[str, list[dict[str, Any]]] = {}

        for k, v in flatten(record, max_nesting).items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                lists[k] = v
            else:
                flat[k] = v

        tables[root_name].append(flat)

        for list_key, list_items in lists.items():
            child_name = list_key.replace("__", separator)
            child_nesting = None if max_nesting is None else max(max_nesting - list_key.count("__") - 1, 0)
            for item in list_items:
                child_flat = flatten(item, child_nesting)
                tables.setdefault(child_name, []).append(child_flat)

    return tables


def to_dataframe(
    data: dict[str, Any] | list[dict[str, Any]],
    max_nesting: int | None = 3,
    table: str = "main",
):
    """Flatten JSON directly to a pandas DataFrame.

    Args:
        data: single record or list of records
        max_nesting: max depth before storing as JSON blob (None = unlimited)
        table: which table to return ("main" for parent, or child table name)

    Returns:
        pandas DataFrame with flattened columns
    """
    import pandas as pd

    tables = normalize_json(data, max_nesting)
    if table not in tables:
        available = list(tables.keys())
        raise KeyError(f"Table '{table}' not found. Available: {available}")
    return pd.DataFrame(tables[table])
