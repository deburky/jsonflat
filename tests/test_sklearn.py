"""Tests for the JsonFlattener sklearn transformer."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from jsonflat.sklearn import JsonFlattener

RECORDS = [
    {"id": 1, "user": {"name": "Alice", "address": {"city": "NYC"}}, "score": 90},
    {"id": 2, "user": {"name": "Bob", "address": {"city": "SF"}}, "score": 60},
    {"id": 3, "user": {"name": "Cara", "address": {"city": "LA"}}, "score": 75},
]


def test_list_of_dicts_input() -> None:
    """A list of dicts flattens to a DataFrame with expected nested columns."""
    out = JsonFlattener().fit_transform(RECORDS)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 3
    assert {"id", "user__name", "user__address__city", "score"} <= set(out.columns)
    assert out["user__address__city"].tolist() == ["NYC", "SF", "LA"]


def test_dataframe_with_json_string_column() -> None:
    """A DataFrame with a JSON-string column is parsed and flattened via ``column=``."""
    df = pd.DataFrame({"payload": [json.dumps(r) for r in RECORDS]})
    out = JsonFlattener(column="payload").fit_transform(df)
    assert len(out) == 3
    assert out["user__name"].tolist() == ["Alice", "Bob", "Cara"]


def test_dataframe_with_dict_valued_column() -> None:
    """A DataFrame whose payload column already holds dicts is handled without parsing."""
    df = pd.DataFrame({"payload": RECORDS})  # column already holds dicts
    out = JsonFlattener(column="payload").fit_transform(df)
    assert len(out) == 3
    assert out["user__address__city"].tolist() == ["NYC", "SF", "LA"]


def test_dataframe_without_column() -> None:
    """Without ``column=``, each DataFrame row is treated as a dict."""
    # Pre-flattened frame — treats each row as a dict
    df = pd.DataFrame([{"id": 1, "city": "NYC"}, {"id": 2, "city": "SF"}, {"id": 3, "city": "LA"}])
    out = JsonFlattener().fit_transform(df)
    assert len(out) == 3
    assert out["city"].tolist() == ["NYC", "SF", "LA"]


def test_table_param_selects_child() -> None:
    """``table=`` selects a child table produced by normalize_json."""
    records = [
        {"order_id": "A1", "items": [{"sku": "W1", "qty": 2}, {"sku": "G1", "qty": 1}]},
        {"order_id": "A2", "items": [{"sku": "X1", "qty": 5}]},
    ]
    out = JsonFlattener(table="items").fit_transform(records)
    assert len(out) == 3  # 2 + 1 items
    assert set(out.columns) == {"sku", "qty"}
    assert out["sku"].tolist() == ["W1", "G1", "X1"]


def test_missing_table_raises_keyerror() -> None:
    """Requesting an unknown table raises KeyError with a helpful message."""
    with pytest.raises(KeyError, match="Table 'nope' not found"):
        JsonFlattener(table="nope").fit_transform(RECORDS)


def test_max_nesting_respected() -> None:
    """``max_nesting`` caps flatten depth inside the transformer."""
    data = [{"a": {"b": {"c": {"d": 1}}}}]
    out_unlimited = JsonFlattener(max_nesting=None).fit_transform(data)
    out_shallow = JsonFlattener(max_nesting=1).fit_transform(data)
    assert "a__b__c__d" in out_unlimited.columns
    assert "a__b" in out_shallow.columns
    assert isinstance(out_shallow["a__b"].iloc[0], dict)


def test_pipeline_fit_predict_numeric_records() -> None:
    """Chain the transformer directly into a classifier when all leaves are numeric."""
    # Matches the README example — all-numeric leaves, direct chain works.
    records = [
        {"info": {"age": 30, "score": 90}},
        {"info": {"age": 25, "score": 60}},
        {"info": {"age": 40, "score": 75}},
    ]
    y = [1, 0, 1]
    pipe = Pipeline(
        [
            ("flatten", JsonFlattener(max_nesting=3)),
            ("model", DecisionTreeClassifier(random_state=0)),
        ]
    )
    pipe.fit(records, y)
    preds = pipe.predict(records)
    assert list(preds) == y


def test_pipeline_with_mixed_types_needs_encoding() -> None:
    """Mixed-type output requires a ColumnTransformer before the model; the chain itself is fine."""
    # Direct chain with string leaves fails at the model step — not the transformer.
    # Users need a ColumnTransformer (or drop strings) between flatten and model.
    from sklearn.compose import ColumnTransformer

    y = [1, 0, 1]
    pipe = Pipeline(
        [
            ("flatten", JsonFlattener(max_nesting=3)),
            (
                "select_numeric",
                ColumnTransformer(
                    [("num", "passthrough", ["id", "score"])],
                    remainder="drop",
                ),
            ),
            ("model", DecisionTreeClassifier(random_state=0)),
        ]
    )
    pipe.fit(RECORDS, y)
    preds = pipe.predict(RECORDS)
    assert list(preds) == y


def test_fit_learns_schema_and_returns_self() -> None:
    """``fit`` stores the column schema as ``columns_`` and returns self."""
    t = JsonFlattener()
    result = t.fit(RECORDS)
    assert result is t
    assert hasattr(t, "columns_")
    assert set(t.columns_) == {"id", "user__name", "user__address__city", "score"}


def test_transform_before_fit_raises() -> None:
    """``transform`` raises NotFittedError when called before ``fit``."""
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        JsonFlattener().transform(RECORDS)


def test_transform_reindexes_to_fitted_schema() -> None:
    """Columns unseen at transform time are filled with NaN; extra columns are dropped."""
    train = [{"a": 1, "b": 2}]
    test = [{"a": 10, "c": 99}]  # "b" missing, "c" new
    t = JsonFlattener().fit(train)
    out = t.transform(test)
    assert list(out.columns) == ["a", "b"]
    assert out["a"].tolist() == [10]
    assert pd.isna(out["b"].iloc[0])


def test_fit_transform_equals_transform_after_fit() -> None:
    """``fit_transform`` and ``fit`` + ``transform`` produce identical output."""
    t = JsonFlattener()
    a = t.fit_transform(RECORDS)
    b = JsonFlattener().fit(RECORDS).transform(RECORDS)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


# ---------------------------------------------------------------------------
# join tests
# ---------------------------------------------------------------------------

JOIN_RECORDS = [
    {"client_id": "C1", "name": "Alice", "transactions": [{"amount": 10.0}, {"amount": 20.0}]},
    {"client_id": "C2", "name": "Bob", "transactions": [{"amount": 5.0}]},
]


def test_join_explodes_child_onto_spine() -> None:
    """join= merges child rows onto the spine, one result row per child row."""
    out = JsonFlattener(join=["transactions"], on="client_id").fit_transform(JOIN_RECORDS)
    assert len(out) == 3  # 2 txns for C1, 1 for C2
    assert "transactions__amount" in out.columns
    assert "name" in out.columns
    assert "client_id" in out.columns
    assert out["transactions__amount"].tolist() == [10.0, 20.0, 5.0]


def test_join_child_columns_are_prefixed() -> None:
    """Child columns are prefixed with <child_name>__ to avoid collision with spine columns."""
    out = JsonFlattener(join=["transactions"], on="client_id").fit_transform(JOIN_RECORDS)
    assert "transactions__amount" in out.columns
    assert "amount" not in out.columns


def test_join_key_not_duplicated() -> None:
    """The join key column appears once, not as both bare and prefixed."""
    out = JsonFlattener(join=["transactions"], on="client_id").fit_transform(JOIN_RECORDS)
    assert out.columns.tolist().count("client_id") == 1


def test_join_inner_how_drops_unmatched() -> None:
    """how='inner' drops spine rows with no matching child rows."""
    records: list[dict[str, Any]] = [
        {"client_id": "C1", "transactions": [{"amount": 10.0}]},
        {"client_id": "C2"},  # no transactions
    ]
    out = JsonFlattener(join=["transactions"], on="client_id", how="inner").fit_transform(records)
    assert set(out["client_id"].tolist()) == {"C1"}


def test_join_without_on_raises() -> None:
    """Specifying join= without on= raises ValueError."""
    with pytest.raises(ValueError, match="'on' must be specified"):
        JsonFlattener(join=["transactions"]).fit_transform(JOIN_RECORDS)


def test_join_missing_child_table_raises() -> None:
    """Requesting a child table that does not exist raises KeyError."""
    with pytest.raises(KeyError, match="Table 'nope' not found"):
        JsonFlattener(join=["nope"], on="client_id").fit_transform(JOIN_RECORDS)
