"""
DynamoDB format decoder for jsonflat.

Converts DynamoDB JSON (typed value dicts) to plain Python, ready for
``flatten()`` or ``normalize_json()``.

Usage::

    from jsonflat.aws.dynamodb import decode
    from jsonflat import flatten

    item = {"id": {"S": "A1"}, "qty": {"N": "3"}, "active": {"BOOL": True}}
    flat = flatten(decode(item))
"""

from __future__ import annotations

from typing import Any


def decode(item: dict[str, Any]) -> dict[str, Any]:
    """Convert a DynamoDB item (typed value dicts) to a plain Python dict.

    :param item: DynamoDB item as returned by ``GetItem``, ``Scan``, or stream images
    :returns: plain Python dict with all DynamoDB type wrappers removed
    """
    return {key: decode_value(value) for key, value in item.items()}


def decode_value(value: dict[str, Any]) -> Any:
    """Convert a single DynamoDB typed value to its Python equivalent.

    :param value: DynamoDB typed value dict, e.g. ``{"S": "hello"}`` or ``{"N": "42"}``
    :returns: plain Python value with the type wrapper removed
    """
    if "S" in value:
        return value["S"]
    if "N" in value:
        n = value["N"]
        return int(n) if "." not in n else float(n)
    if "BOOL" in value:
        return value["BOOL"]
    if "NULL" in value:
        return None
    if "L" in value:
        return [decode_value(v) for v in value["L"]]
    if "M" in value:
        return decode(value["M"])
    if "SS" in value:
        return set(value["SS"])
    if "NS" in value:
        return {int(n) if "." not in n else float(n) for n in value["NS"]}
    if "B" in value:
        return value["B"]
    return set(value["BS"]) if "BS" in value else value
