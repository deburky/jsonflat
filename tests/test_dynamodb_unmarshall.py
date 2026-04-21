"""Unit tests for the private ``_unmarshall_value`` helper.

DynamoDB's ``GetRecords`` / ``Scan`` responses encode values with type-marker
dicts (``{"S": "..."}``, ``{"N": "42"}``, etc.). ``_unmarshall_value`` dispatches
on the marker. These tests exercise every branch without needing real AWS or
moto.
"""

from __future__ import annotations

from jsonflat.aws.dynamodb import _unmarshall, _unmarshall_value


def test_string():
    """``S`` returns the raw string."""
    assert _unmarshall_value({"S": "hello"}) == "hello"


def test_integer_number():
    """``N`` without a decimal point returns an int."""
    assert _unmarshall_value({"N": "42"}) == 42
    assert isinstance(_unmarshall_value({"N": "42"}), int)


def test_float_number():
    """``N`` with a decimal point returns a float."""
    assert _unmarshall_value({"N": "3.14"}) == 3.14
    assert isinstance(_unmarshall_value({"N": "3.14"}), float)


def test_bool():
    """``BOOL`` returns the Python bool."""
    assert _unmarshall_value({"BOOL": True}) is True
    assert _unmarshall_value({"BOOL": False}) is False


def test_null():
    """``NULL`` returns None."""
    assert _unmarshall_value({"NULL": True}) is None


def test_list():
    """``L`` returns a list with each element recursively unmarshalled."""
    got = _unmarshall_value({"L": [{"S": "a"}, {"N": "1"}, {"BOOL": False}]})
    assert got == ["a", 1, False]


def test_map():
    """``M`` returns a dict with each value recursively unmarshalled."""
    got = _unmarshall_value({"M": {"x": {"S": "hi"}, "y": {"N": "2"}}})
    assert got == {"x": "hi", "y": 2}


def test_string_set():
    """``SS`` returns a Python set of strings."""
    assert _unmarshall_value({"SS": ["a", "b", "c"]}) == {"a", "b", "c"}


def test_number_set_integers():
    """``NS`` with integer strings returns a set of ints."""
    assert _unmarshall_value({"NS": ["1", "2", "3"]}) == {1, 2, 3}


def test_number_set_mixed():
    """``NS`` converts each element individually (int or float per-element)."""
    got = _unmarshall_value({"NS": ["1", "2.5"]})
    assert got == {1, 2.5}


def test_binary():
    """``B`` returns the raw bytes."""
    assert _unmarshall_value({"B": b"\x01\x02"}) == b"\x01\x02"


def test_binary_set():
    """``BS`` returns a set of bytes."""
    assert _unmarshall_value({"BS": [b"a", b"b"]}) == {b"a", b"b"}


def test_unknown_marker_passthrough():
    """An unknown marker returns the value dict as-is (defensive fallback)."""
    assert _unmarshall_value({"WHAT": "???"}) == {"WHAT": "???"}


def test_unmarshall_full_item():
    """``_unmarshall`` unwraps every field of a full DynamoDB item dict."""
    item = {
        "id": {"S": "A1"},
        "qty": {"N": "3"},
        "active": {"BOOL": True},
        "tags": {"SS": ["x", "y"]},
        "meta": {"M": {"created": {"S": "2026-04-21"}}},
    }
    assert _unmarshall(item) == {
        "id": "A1",
        "qty": 3,
        "active": True,
        "tags": {"x", "y"},
        "meta": {"created": "2026-04-21"},
    }
