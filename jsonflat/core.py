"""
jsonflat -- flatten nested JSON into DataFrames with controlled depth.

Usage:
    from jsonflat import flatten, normalize_json, to_dataframe

    flat = flatten(data, max_nesting=3)           # single flat dict
    tables = normalize_json(data, max_nesting=3)  # parent + child tables
    df = to_dataframe(data, max_nesting=3)        # straight to DataFrame
"""

from __future__ import annotations

import asyncio
import functools
import warnings
from typing import Any, Literal, cast


def aio(
    workers: int = 32, pool=None, service: str | None = None, profile: str | None = None, region: str | None = None
):
    """Decorator that runs an async function over a list concurrently.

    :param workers: max concurrent coroutines / connection pool size
    :param pool: async context manager factory, opened once and injected as second arg
    :param service: AWS service shortcut (e.g. ``"s3"``); builds an aioboto3 pool automatically
    :param profile: AWS profile (used with ``service``)
    :param region: AWS region (used with ``service``)
    :returns: decorator that replaces ``async fn(item)`` with ``fn(items) -> list``

    Usage::

        @aio(workers=32, service="s3", profile="my-profile")
        async def fetch(key, s3):
            resp = await s3.get_object(Bucket=bucket, Key=key)
            return json.loads(await resp['Body'].read())

        records = fetch(keys)
    """
    if service is not None:
        try:
            import aioboto3
            from aiobotocore.config import AioConfig as Config
        except ImportError as e:
            raise ImportError("pip install aioboto3") from e
        _session = aioboto3.Session(profile_name=profile, region_name=region)
        _svc = cast(Literal["s3"], service)
        pool = lambda: _session.client(_svc, config=Config(max_pool_connections=workers))  # noqa: E731

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(items, *args, **kwargs):
            async def run():
                if pool is not None:
                    async with pool() as client:
                        return list(await asyncio.gather(*[fn(item, client, *args, **kwargs) for item in items]))
                sem = asyncio.Semaphore(workers)

                async def call(item):
                    async with sem:
                        return await fn(item, *args, **kwargs)

                return list(await asyncio.gather(*[call(item) for item in items]))

            return asyncio.run(run())

        return wrapper

    return decorator


def flatten(
    row: dict[str, Any],
    max_nesting: int | None = 3,
    _depth: int = 0,
    _path: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Flatten a nested dict into a single-level dict with __ separators.

    :param row: nested dict to flatten
    :param max_nesting: max depth before storing as JSON blob (None = unlimited)
    :param _depth: current recursion depth (internal)
    :param _path: current key path (internal)
    :returns: flat dict with __ separated keys; nested dicts/lists beyond max_nesting are preserved as-is
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
    key: str | None = None,
    hoist: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Normalize JSON into parent + child tables.

    :param data: single record or list of records
    :param max_nesting: max depth before storing as JSON blob (None = unlimited)
    :param root_name: name for the root table
    :param separator: separator for child table names (e.g. ``"."`` → ``"policy_output.offers"``)
    :param key: field to copy from each parent into child rows as a foreign key
    :param hoist: prefixes where dict keys are IDs — hoisted as row values instead of table name segments.
                  Each entry is a string (uses ``{prefix}_id``) or a ``(prefix, id_col)`` tuple.
                  e.g. ``hoist=["loans"]`` → ``loans_id`` column; ``hoist=[("loans", "loan_id")]`` → ``loan_id``
    :returns: dict mapping table_name → list of row dicts
    """
    if isinstance(data, dict):
        data = [data]

    hoist_map: dict[str, str] = {}
    for entry in hoist or []:
        if isinstance(entry, tuple):
            prefix, id_col_name = entry
        else:
            prefix, id_col_name = entry, f"{entry}_id"
        hoist_map[prefix] = id_col_name

    tables: dict[str, list[dict[str, Any]]] = {root_name: []}

    for record in data:
        flat: dict[str, Any] = {}
        lists: dict[str, list[dict[str, Any]]] = {}

        for k, v in flatten(record, max_nesting).items():
            parts = k.split("__")
            if hoist_map and parts[0] in hoist_map and len(parts) >= 2:
                hoist_name, hoist_id, *rest = parts
                id_col = hoist_map[hoist_name]
                child_key = "__".join([hoist_name, *rest]) if rest else hoist_name
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    for item in v:
                        row = flatten(item, max_nesting)
                        row[id_col] = hoist_id
                        tables.setdefault(child_key.replace("__", separator), []).append(row)
                else:
                    tables.setdefault(hoist_name, []).append({id_col: hoist_id, "value": v})
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                lists[k] = v
            else:
                flat[k] = v

        tables[root_name].append(flat)

        if key and key not in flat:
            raise KeyError(f"Key '{key}' not found in record. Available keys: {list(flat.keys())}")
        parent_key_value = flat.get(key) if key else None

        for list_key, list_items in lists.items():
            child_name = list_key.replace("__", separator)
            child_nesting = None if max_nesting is None else max(max_nesting - list_key.count("__") - 1, 0)
            for item in list_items:
                child_flat = flatten(item, child_nesting)
                if key and parent_key_value is not None:
                    if key in child_flat:
                        if child_flat[key] != parent_key_value:
                            raise ValueError(
                                f"Key '{key}' already exists in child table '{child_name}' with a different value "
                                f"({child_flat[key]!r} vs parent {parent_key_value!r})."
                            )
                        warnings.warn(
                            f"Key '{key}' already exists in child table '{child_name}' "
                            "with the same value (skipping overwrite).",
                            stacklevel=2,
                        )
                    else:
                        child_flat[key] = parent_key_value
                tables.setdefault(child_name, []).append(child_flat)

    return tables


def to_dataframe(
    data: dict[str, Any] | list[dict[str, Any]],
    max_nesting: int | None = 3,
    table: str = "main",
):
    """Flatten JSON directly to a pandas DataFrame.

    :param data: single record or list of records
    :param max_nesting: max depth before storing as JSON blob (None = unlimited)
    :param table: which table to return (``"main"`` for parent, or child table name)
    :returns: pandas DataFrame with flattened columns
    """
    import pandas as pd

    tables = normalize_json(data, max_nesting)
    if table not in tables:
        available = list(tables.keys())
        raise KeyError(f"Table '{table}' not found. Available: {available}")
    return pd.DataFrame(tables[table])
