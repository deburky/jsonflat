"""Tests for key, hoist, aio, and pandas Series input — patterns from real usage."""
from __future__ import annotations

import asyncio
import warnings

import pandas as pd
import pytest

from jsonflat import aio, normalize_json


# ---------------------------------------------------------------------------
# key — foreign key injection into child rows
# ---------------------------------------------------------------------------
class TestKey:
    """Tests for the key= parameter on normalize_json."""

    def test_key_injected_into_child_rows(self) -> None:
        """Parent key value is copied into each child row."""
        record = {"client_id": "c1", "docs": [{"type": "NIN"}]}
        tables = normalize_json(record, key="client_id")
        assert tables["docs"][0]["client_id"] == "c1"

    def test_key_missing_raises(self) -> None:
        """Raises KeyError when the specified key does not exist in the record."""
        record = {"name": "Alice", "docs": [{"type": "NIN"}]}
        with pytest.raises(KeyError, match="encodedKey"):
            normalize_json(record, key="encodedKey")

    def test_key_collision_same_value_warns(self) -> None:
        """Warning is issued when child already has the key with the same value."""
        record = {"client_id": "c1", "docs": [{"type": "NIN", "client_id": "c1"}]}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tables = normalize_json(record, key="client_id")
        assert any("same value" in str(warning.message) for warning in w)
        assert tables["docs"][0]["client_id"] == "c1"

    def test_key_collision_different_value_raises(self) -> None:
        """Raises ValueError when child already has the key with a different value."""
        record = {"client_id": "c1", "docs": [{"type": "NIN", "client_id": "c2"}]}
        with pytest.raises(ValueError, match="different value"):
            normalize_json(record, key="client_id")

    def test_key_multiple_records(self) -> None:
        """Each parent's key value is injected into its own child rows only."""
        records = [
            {"client_id": "c1", "docs": [{"type": "A"}]},
            {"client_id": "c2", "docs": [{"type": "B"}]},
        ]
        tables = normalize_json(records, key="client_id")
        assert tables["docs"][0]["client_id"] == "c1"
        assert tables["docs"][1]["client_id"] == "c2"

    def test_key_no_children_no_error(self) -> None:
        """Specifying a key on a flat record with no child lists is safe."""
        record = {"client_id": "c1", "name": "Alice"}
        tables = normalize_json(record, key="client_id")
        assert tables["main"][0]["client_id"] == "c1"


# ---------------------------------------------------------------------------
# hoist — dict keys are IDs, not table name segments
# ---------------------------------------------------------------------------
class TestHoist:
    """Tests for the hoist= parameter on normalize_json."""

    def test_hoist_string_form_uses_prefix_id_column(self) -> None:
        """String hoist entry produces a {prefix}_id column."""
        record = {"loans": {"L001": {"status": "active"}}}
        tables = normalize_json(record, hoist=["loans"])
        row = tables["loans"][0]
        assert row["loans_id"] == "L001"

    def test_hoist_tuple_form_uses_custom_column(self) -> None:
        """Tuple hoist entry names the ID column explicitly."""
        record = {"loans": {"L001": {"status": "active"}}}
        tables = normalize_json(record, hoist=[("loans", "loan_id")])
        row = tables["loans"][0]
        assert row["loan_id"] == "L001"
        assert "loans_id" not in row

    def test_hoist_list_of_dicts_becomes_child_table(self) -> None:
        """Nested list of dicts under a hoisted ID becomes a child table with the ID column."""
        record = {"loans": {"L001": {"fields": [{"name": "purpose", "value": "biz"}]}}}
        tables = normalize_json(record, hoist=[("loans", "loan_id")])
        assert "loans.fields" in tables
        row = tables["loans.fields"][0]
        assert row["loan_id"] == "L001"
        assert row["name"] == "purpose"

    def test_hoist_multiple_ids_all_linked(self) -> None:
        """Multiple hoisted IDs each link their child rows correctly."""
        record = {
            "loans": {
                "L001": {"items": [{"type": "A"}]},
                "L002": {"items": [{"type": "B"}]},
            }
        }
        tables = normalize_json(record, hoist=[("loans", "loan_id")])
        assert len(tables["loans.items"]) == 2
        ids = {r["loan_id"] for r in tables["loans.items"]}
        assert ids == {"L001", "L002"}

    def test_hoist_multiple_records(self) -> None:
        """Hoisted IDs from multiple parent records all accumulate in the child table."""
        records = [
            {"loans": {"L001": {"items": [{"type": "A"}]}}},
            {"loans": {"L002": {"items": [{"type": "B"}]}}},
        ]
        tables = normalize_json(records, hoist=[("loans", "loan_id")])
        assert len(tables["loans.items"]) == 2


# ---------------------------------------------------------------------------
# aio — async function over list
# ---------------------------------------------------------------------------
class TestAio:
    """Tests for the @aio decorator."""

    def test_no_pool_runs_concurrently(self) -> None:
        """Decorated async function runs over every item and returns results."""
        @aio(workers=4)
        async def double(x):
            return x * 2

        assert double([1, 2, 3]) == [2, 4, 6]

    def test_no_pool_preserves_order(self) -> None:
        """Results are returned in input order even when coroutines finish out of order."""
        @aio(workers=4)
        async def slow_first(x):
            if x == 0:
                await asyncio.sleep(0.05)
            return x

        assert slow_first([0, 1, 2]) == [0, 1, 2]

    def test_with_pool_injects_client(self) -> None:
        """Pool context manager is opened once and injected as the second argument."""
        class FakeClient:
            async def compute(self, x):
                return x + 10

        class FakePool:
            async def __aenter__(self):
                return FakeClient()

            async def __aexit__(self, *_):
                pass

        @aio(workers=4, pool=FakePool)
        async def fetch(x, client):
            return await client.compute(x)

        assert fetch([1, 2, 3]) == [11, 12, 13]

    def test_empty_list(self) -> None:
        """Empty input returns an empty list without error."""
        @aio(workers=4)
        async def noop(x):
            return x

        assert noop([]) == []


# ---------------------------------------------------------------------------
# pandas Series input
# ---------------------------------------------------------------------------
class TestPandasSeriesInput:
    """Tests for passing a pandas Series to normalize_json."""

    def test_series_of_dicts(self) -> None:
        """A Series of flat dicts normalizes into the main table."""
        series = pd.Series([{"id": 1, "val": "a"}, {"id": 2, "val": "b"}])
        tables = normalize_json(series)
        assert len(tables["main"]) == 2

    def test_series_with_child_tables(self) -> None:
        """A Series of nested dicts splits into main and child tables."""
        series = pd.Series([
            {"id": 1, "items": [{"x": 10}, {"x": 20}]},
            {"id": 2, "items": [{"x": 30}]},
        ])
        tables = normalize_json(series)
        assert len(tables["main"]) == 2
        assert len(tables["items"]) == 3
