"""Tests for the JsonFlattener sklearn transformer."""

from __future__ import annotations

import json

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


def test_list_of_dicts_input():
    """A list of dicts flattens to a DataFrame with expected nested columns."""
    out = JsonFlattener().fit_transform(RECORDS)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 3
    assert {"id", "user__name", "user__address__city", "score"} <= set(out.columns)
    assert out["user__address__city"].tolist() == ["NYC", "SF", "LA"]


def test_dataframe_with_json_string_column():
    """A DataFrame with a JSON-string column is parsed and flattened via ``column=``."""
    df = pd.DataFrame({"payload": [json.dumps(r) for r in RECORDS]})
    out = JsonFlattener(column="payload").fit_transform(df)
    assert len(out) == 3
    assert out["user__name"].tolist() == ["Alice", "Bob", "Cara"]


def test_dataframe_with_dict_valued_column():
    """A DataFrame whose payload column already holds dicts is handled without parsing."""
    df = pd.DataFrame({"payload": RECORDS})  # column already holds dicts
    out = JsonFlattener(column="payload").fit_transform(df)
    assert len(out) == 3
    assert out["user__address__city"].tolist() == ["NYC", "SF", "LA"]


def test_dataframe_without_column():
    """Without ``column=``, each DataFrame row is treated as a dict."""
    # Pre-flattened frame — treats each row as a dict
    df = pd.DataFrame(
        [{"id": 1, "city": "NYC"}, {"id": 2, "city": "SF"}, {"id": 3, "city": "LA"}]
    )
    out = JsonFlattener().fit_transform(df)
    assert len(out) == 3
    assert out["city"].tolist() == ["NYC", "SF", "LA"]


def test_table_param_selects_child():
    """``table=`` selects a child table produced by normalize_json."""
    records = [
        {"order_id": "A1", "items": [{"sku": "W1", "qty": 2}, {"sku": "G1", "qty": 1}]},
        {"order_id": "A2", "items": [{"sku": "X1", "qty": 5}]},
    ]
    out = JsonFlattener(table="items").fit_transform(records)
    assert len(out) == 3  # 2 + 1 items
    assert set(out.columns) == {"sku", "qty"}
    assert out["sku"].tolist() == ["W1", "G1", "X1"]


def test_missing_table_raises_keyerror():
    """Requesting an unknown table raises KeyError with a helpful message."""
    with pytest.raises(KeyError, match="Table 'nope' not found"):
        JsonFlattener(table="nope").fit_transform(RECORDS)


def test_max_nesting_respected():
    """``max_nesting`` caps flatten depth inside the transformer."""
    data = [{"a": {"b": {"c": {"d": 1}}}}]
    out_unlimited = JsonFlattener(max_nesting=None).fit_transform(data)
    out_shallow = JsonFlattener(max_nesting=1).fit_transform(data)
    assert "a__b__c__d" in out_unlimited.columns
    assert "a__b" in out_shallow.columns
    assert isinstance(out_shallow["a__b"].iloc[0], dict)


def test_pipeline_fit_predict_numeric_records():
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


def test_pipeline_with_mixed_types_needs_encoding():
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


def test_fit_is_noop_and_returns_self():
    """``fit`` is a no-op and returns the transformer instance."""
    t = JsonFlattener()
    assert t.fit(RECORDS) is t


def test_fit_transform_equals_transform_after_fit():
    """``fit_transform`` and ``fit`` + ``transform`` produce identical output."""
    t = JsonFlattener()
    a = t.fit_transform(RECORDS)
    b = t.fit(RECORDS).transform(RECORDS)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
