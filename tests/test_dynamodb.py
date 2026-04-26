"""Tests for the DynamoDB decoder.

All tests go through the public API: ``decode`` for full items and
``decode_value`` for individual typed values.
"""

from __future__ import annotations

from jsonflat.aws.dynamodb import decode, decode_value


def test_string() -> None:
    """``S`` returns the raw string."""
    assert decode_value({"S": "hello"}) == "hello"


def test_integer_number() -> None:
    """``N`` without a decimal point returns an int."""
    assert decode_value({"N": "42"}) == 42
    assert isinstance(decode_value({"N": "42"}), int)


def test_float_number() -> None:
    """``N`` with a decimal point returns a float."""
    assert decode_value({"N": "3.14"}) == 3.14
    assert isinstance(decode_value({"N": "3.14"}), float)


def test_bool() -> None:
    """``BOOL`` returns the Python bool."""
    assert decode_value({"BOOL": True}) is True
    assert decode_value({"BOOL": False}) is False


def test_null() -> None:
    """``NULL`` returns None."""
    assert decode_value({"NULL": True}) is None


def test_list() -> None:
    """``L`` returns a list with each element recursively decoded."""
    assert decode_value({"L": [{"S": "a"}, {"N": "1"}, {"BOOL": False}]}) == ["a", 1, False]


def test_map() -> None:
    """``M`` returns a dict with each value recursively decoded."""
    assert decode_value({"M": {"x": {"S": "hi"}, "y": {"N": "2"}}}) == {"x": "hi", "y": 2}


def test_string_set() -> None:
    """``SS`` returns a Python set of strings."""
    assert decode_value({"SS": ["a", "b", "c"]}) == {"a", "b", "c"}


def test_number_set_integers() -> None:
    """``NS`` with integer strings returns a set of ints."""
    assert decode_value({"NS": ["1", "2", "3"]}) == {1, 2, 3}


def test_number_set_mixed() -> None:
    """``NS`` converts each element individually (int or float per-element)."""
    assert decode_value({"NS": ["1", "2.5"]}) == {1, 2.5}


def test_binary() -> None:
    """``B`` returns the raw bytes."""
    assert decode_value({"B": b"\x01\x02"}) == b"\x01\x02"


def test_binary_set() -> None:
    """``BS`` returns a set of bytes."""
    assert decode_value({"BS": [b"a", b"b"]}) == {b"a", b"b"}


def test_unknown_marker_passthrough() -> None:
    """An unknown marker returns the value dict as-is (defensive fallback)."""
    assert decode_value({"WHAT": "???"}) == {"WHAT": "???"}


def test_decode_empty_item() -> None:
    """``decode`` on an empty item returns an empty dict."""
    assert decode({}) == {}


def test_decode_full_item() -> None:
    """``decode`` unwraps every field of a full DynamoDB item dict."""
    item = {
        "id": {"S": "A1"},
        "qty": {"N": "3"},
        "active": {"BOOL": True},
        "tags": {"SS": ["x", "y"]},
        "meta": {"M": {"created": {"S": "2026-04-21"}}},
    }
    assert decode(item) == {
        "id": "A1",
        "qty": 3,
        "active": True,
        "tags": {"x", "y"},
        "meta": {"created": "2026-04-21"},
    }


def test_decode_deeply_nested() -> None:
    """``decode`` recurses correctly through nested M and L combinations."""
    item = {
        "order": {
            "M": {
                "id": {"S": "O1"},
                "lines": {
                    "L": [
                        {"M": {"sku": {"S": "W1"}, "qty": {"N": "2"}}},
                        {"M": {"sku": {"S": "G1"}, "qty": {"N": "1"}}},
                    ]
                },
            }
        }
    }
    assert decode(item) == {
        "order": {
            "id": "O1",
            "lines": [
                {"sku": "W1", "qty": 2},
                {"sku": "G1", "qty": 1},
            ],
        }
    }
