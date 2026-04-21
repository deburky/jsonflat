"""Tests for unflatten — the dual of flatten."""

from __future__ import annotations

import pytest

from jsonflat import flatten, unflatten


def test_basic():
    """Flat keys with a single separator rebuild into a nested dict."""
    assert unflatten({"a__b": 1, "a__c": 2}) == {"a": {"b": 1, "c": 2}}


def test_deep_nesting():
    """Multi-segment keys rebuild into fully nested dicts."""
    assert unflatten({"a__b__c__d": 1}) == {"a": {"b": {"c": {"d": 1}}}}


def test_top_level_scalars_passthrough():
    """Keys without the separator stay at the top level unchanged."""
    assert unflatten({"x": 1, "y": "hello"}) == {"x": 1, "y": "hello"}


def test_preserves_list_values():
    """Lists (even of dicts) are treated as leaves and never split."""
    data = {"items": [{"sku": "W1"}, {"sku": "G1"}], "user__name": "Alice"}
    assert unflatten(data) == {"items": [{"sku": "W1"}, {"sku": "G1"}], "user": {"name": "Alice"}}


def test_preserves_stored_dict_values():
    """Dict values stored as blobs by flatten at max_nesting pass through unchanged."""
    data = {"a__b": {"c": {"d": 1}}}
    assert unflatten(data) == {"a": {"b": {"c": {"d": 1}}}}


def test_custom_separator():
    """A non-default separator is honored when splitting keys."""
    assert unflatten({"a.b.c": 1}, separator=".") == {"a": {"b": {"c": 1}}}


def test_roundtrip_flatten_unflatten():
    """Round-trip through flatten then unflatten returns the original nested dict."""
    original = {
        "order_id": "A1",
        "user": {"name": "Alice", "address": {"city": "NYC", "zip": "10001"}},
        "amount": 42.5,
        "tags": ["a", "b"],
    }
    assert unflatten(flatten(original, max_nesting=None)) == original


def test_roundtrip_with_max_nesting_blob():
    """Round-trip works even when flatten stores deep subtrees as blobs."""
    original = {"a": {"b": {"c": {"d": 1}}}}
    flat = flatten(original, max_nesting=2)
    assert unflatten(flat) == original


def test_conflict_leaf_and_parent_raises():
    """A key used as both a leaf and a parent (leaf first) raises ValueError."""
    with pytest.raises(ValueError, match="both a leaf and a parent"):
        unflatten({"a": 1, "a__b": 2})


def test_conflict_parent_and_leaf_raises():
    """A key used as both a parent and a leaf (parent first) raises ValueError."""
    with pytest.raises(ValueError, match="both a leaf and a parent"):
        unflatten({"a__b": 2, "a": 1})


def test_empty_dict():
    """Empty input returns an empty dict."""
    assert unflatten({}) == {}


def test_version_is_exposed():
    """The package exposes __version__ as a non-empty string."""
    import jsonflat

    assert isinstance(jsonflat.__version__, str)
    assert jsonflat.__version__  # non-empty
