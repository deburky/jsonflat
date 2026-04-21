"""Tests for flatten, normalize_json, and to_dataframe core functions."""
from __future__ import annotations

import pytest

from jsonflat import flatten, normalize_json, to_dataframe

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
SAMPLE = {
    "id": "order-123",
    "customer": {
        "name": "Alice",
        "age": 30,
        "score": 650,
        "items": [
            {"product": "Widget", "qty": 2, "price": 9.99},
            {"product": "Gadget", "qty": 1, "price": 24.50},
        ],
        "tags": {"vip": True, "region": "EU"},
    },
    "metadata": {
        "source": {
            "system": {"name": "web", "version": "2.1"},
            "tracking": {"session_id": "s-001"},
            "extra": {
                "nested": {
                    "deep_key": "deep_val",
                }
            },
        }
    },
}

DEEP = {"a": {"b": {"c": {"d": {"e": 1}}}}}


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------
class TestFlatten:
    """Tests for the flatten() function."""

    def test_flat_dict_unchanged(self) -> None:
        """Already-flat dict passes through unmodified."""
        row = {"x": 1, "y": "hello"}
        assert flatten(row) == {"x": 1, "y": "hello"}

    def test_one_level_nesting(self) -> None:
        """Single nested dict is joined with __ separator."""
        row = {"a": {"b": 1, "c": 2}}
        result = flatten(row)
        assert result == {"a__b": 1, "a__c": 2}

    def test_max_nesting_none_flattens_all(self) -> None:
        """max_nesting=None flattens arbitrarily deep dicts."""
        result = flatten(DEEP, max_nesting=None)
        assert result == {"a__b__c__d__e": 1}

    def test_max_nesting_0_no_flattening(self) -> None:
        """max_nesting=0 leaves the dict entirely unflattened."""
        result = flatten(DEEP, max_nesting=0)
        assert result == {"a": {"b": {"c": {"d": {"e": 1}}}}}

    def test_max_nesting_2_stops_at_depth_2(self) -> None:
        """max_nesting=2 flattens two levels and stores deeper nesting as a blob."""
        result = flatten(DEEP, max_nesting=2)
        assert "a__b__c" in result
        assert isinstance(result["a__b__c"], dict)
        assert result["a__b__c"] == {"d": {"e": 1}}

    def test_list_of_dicts_preserved(self) -> None:
        """A list of dicts within the max_nesting limit is kept as-is."""
        row = {"items": [{"x": 1}, {"x": 2}]}
        result = flatten(row, max_nesting=3)
        assert result["items"] == [{"x": 1}, {"x": 2}]

    def test_list_of_dicts_at_max_nesting_preserved(self) -> None:
        """A list of dicts at the nesting limit is stored as a blob."""
        row = {"a": {"items": [{"x": 1}]}}
        result = flatten(row, max_nesting=0)
        assert result == {"a": {"items": [{"x": 1}]}}

    def test_list_of_scalars_stays_as_value(self) -> None:
        """A list of scalars is kept as a plain value."""
        row = {"tags": [1, 2, 3]}
        result = flatten(row)
        assert result == {"tags": [1, 2, 3]}

    def test_empty_list_stays_as_value(self) -> None:
        """An empty list is kept as a plain value."""
        row: dict[str, list] = {"items": []}
        result = flatten(row)
        assert result == {"items": []}

    def test_empty_dict(self) -> None:
        """Empty dict returns empty dict."""
        assert flatten({}) == {}

    def test_sample_nesting_3(self) -> None:
        """Realistic nested record flattens correctly at depth 3."""
        result = flatten(SAMPLE, max_nesting=3)
        assert result["id"] == "order-123"
        assert result["customer__name"] == "Alice"
        assert result["customer__score"] == 650
        assert result["customer__tags__vip"] is True
        assert result["metadata__source__system__name"] == "web"
        assert isinstance(result["metadata__source__extra__nested"], dict)


# ---------------------------------------------------------------------------
# Normalize JSON
# ---------------------------------------------------------------------------
class TestNormalizeJson:
    """Tests for the normalize_json() function."""

    def test_single_record_wrapped(self) -> None:
        """A single dict is wrapped into a list under the root table."""
        tables = normalize_json({"x": 1}, max_nesting=3)
        assert "main" in tables
        assert len(tables["main"]) == 1

    def test_list_of_records(self) -> None:
        """A list of records each becomes a row in the root table."""
        tables = normalize_json([{"x": 1}, {"x": 2}], max_nesting=3)
        assert len(tables["main"]) == 2

    def test_child_tables_created(self) -> None:
        """Lists of dicts are extracted into named child tables."""
        tables = normalize_json(SAMPLE, max_nesting=3)
        assert "customer.items" in tables
        assert len(tables["customer.items"]) == 2
        assert tables["customer.items"][0]["product"] == "Widget"

    def test_child_table_separator(self) -> None:
        """Custom separator changes child table name format."""
        tables = normalize_json(SAMPLE, max_nesting=3, separator="/")
        assert "customer/items" in tables

    def test_custom_root_name(self) -> None:
        """root_name parameter renames the main table."""
        tables = normalize_json({"x": 1}, root_name="raw")
        assert "raw" in tables
        assert "main" not in tables

    def test_no_child_tables_when_nesting_0(self) -> None:
        """max_nesting=0 stores everything as blobs with no child tables."""
        tables = normalize_json(SAMPLE, max_nesting=0)
        assert list(tables.keys()) == ["main"]
        row = tables["main"][0]
        assert isinstance(row["customer"], dict)
        assert isinstance(row["metadata"], dict)

    def test_max_nesting_none_still_extracts_child_tables(self) -> None:
        """max_nesting=None still splits lists of dicts into child tables."""
        tables = normalize_json(SAMPLE, max_nesting=None)
        assert "customer.items" in tables

    def test_parent_row_excludes_list_of_dicts(self) -> None:
        """Parent row does not contain the key for a list that became a child table."""
        tables = normalize_json(SAMPLE, max_nesting=3)
        parent = tables["main"][0]
        assert "customer__items" not in parent

    def test_multiple_records_accumulate_children(self) -> None:
        """Child rows from multiple parent records are all collected in the child table."""
        records = [SAMPLE, SAMPLE]
        tables = normalize_json(records, max_nesting=3)
        assert len(tables["main"]) == 2
        assert len(tables["customer.items"]) == 4


# ---------------------------------------------------------------------------
# To Dataframe
# ---------------------------------------------------------------------------
class TestToDataframe:
    """Tests for the to_dataframe() function."""

    def test_returns_dataframe(self) -> None:
        """Returns a DataFrame with flattened columns for the main table."""
        df = to_dataframe(SAMPLE, max_nesting=3)
        assert len(df) == 1
        assert "customer__score" in df.columns

    def test_child_table(self) -> None:
        """Returns a DataFrame for a named child table."""
        df = to_dataframe(SAMPLE, max_nesting=3, table="customer.items")
        assert len(df) == 2
        assert list(df["product"]) == ["Widget", "Gadget"]

    def test_missing_table_raises(self) -> None:
        """Raises KeyError when the requested table does not exist."""
        with pytest.raises(KeyError, match="not_real"):
            to_dataframe(SAMPLE, max_nesting=3, table="not_real")

    def test_nesting_none(self) -> None:
        """max_nesting=None flattens fully into DataFrame columns."""
        df = to_dataframe({"a": {"b": {"c": 1}}}, max_nesting=None)
        assert "a__b__c" in df.columns
