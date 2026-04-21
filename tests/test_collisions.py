"""Tests for key-collision detection in flatten.

Native keys containing the ``__`` separator conflict with nested paths.
Previously flatten silently overwrote; it now raises ``ValueError`` to
prevent silent data loss.
"""

from __future__ import annotations

import pytest

from jsonflat import flatten


def test_native_double_underscore_vs_nested_path():
    """A native 'user__id' and a nested {'user': {'id': ...}} collide."""
    data = {"user__id": 1, "user": {"id": 2}}
    with pytest.raises(ValueError, match="Key collision on 'user__id'"):
        flatten(data, max_nesting=None)


def test_reversed_order_still_raises():
    """Collision is detected regardless of insertion order."""
    data = {"user": {"id": 2}, "user__id": 1}
    with pytest.raises(ValueError, match="Key collision"):
        flatten(data, max_nesting=None)


def test_deep_collision_between_branches():
    """Two distinct branches that flatten to the same key raise."""
    data = {"a": {"b__c": 1}, "a__b": {"c": 2}}
    with pytest.raises(ValueError, match="Key collision"):
        flatten(data, max_nesting=None)


def test_no_collision_when_keys_differ():
    """Normal nested data without ambiguity flattens as expected."""
    data = {"user": {"id": 2, "name": "Alice"}, "score": 90}
    flat = flatten(data, max_nesting=None)
    assert flat == {"user__id": 2, "user__name": "Alice", "score": 90}


def test_native_double_underscore_without_collision():
    """A native '__' key is fine when no nested path produces the same key."""
    data = {"weird__key": "hi", "user": {"name": "Alice"}}
    flat = flatten(data, max_nesting=None)
    assert flat == {"weird__key": "hi", "user__name": "Alice"}
