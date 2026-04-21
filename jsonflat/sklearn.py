"""
Scikit-learn compatible transformer for jsonflat.

Usage:
    from jsonflat.sklearn import JsonFlattener
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ("flatten", JsonFlattener(max_nesting=3)),
        ("model", SomeClassifier()),
    ])

    # From list of dicts
    pipe.fit(json_records, y)

    # From DataFrame with a JSON column
    pipe.fit(df, y)
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from jsonflat.core import normalize_json


class JsonFlattener(BaseEstimator, TransformerMixin):
    """Flatten nested JSON records into a DataFrame for use in sklearn pipelines.

    :param max_nesting: max depth before storing as JSON blob (None = unlimited)
    :param column: when input is a DataFrame, the column containing nested dicts or JSON strings;
                   if None, treats each row as a dict
    :param table: which table to return from normalize_json (default: ``"main"``)
    """

    def __init__(
        self,
        max_nesting: int | None = 3,
        column: str | None = None,
        table: str = "main",
    ) -> None:
        """Initialize with flattening parameters."""
        self.max_nesting = max_nesting
        self.column = column
        self.table = table

    def fit(
        self,
        X: list[dict[str, Any]] | pd.DataFrame,
        y: Any = None,
    ) -> JsonFlattener:
        """No-op. JsonFlattener is stateless."""
        return self

    def transform(
        self,
        X: list[dict[str, Any]] | pd.DataFrame,
    ) -> pd.DataFrame:
        """Flatten nested JSON records into a DataFrame.

        :param X: list of dicts or a DataFrame; if a DataFrame and ``column`` is set,
                  that column is used as the source of nested records
        :returns: DataFrame with flattened columns
        """
        records = self._extract_records(X)
        tables = normalize_json(records, max_nesting=self.max_nesting)
        if self.table not in tables:
            available = list(tables.keys())
            raise KeyError(f"Table '{self.table}' not found. Available: {available}")
        return pd.DataFrame(tables[self.table])

    def _extract_records(
        self,
        X: list[dict[str, Any]] | pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Convert input to a list of dicts."""
        if isinstance(X, list):
            return X

        if self.column is not None:
            raw = X[self.column].tolist()
            return [json.loads(r) if isinstance(r, str) else r for r in raw]

        return [{str(k): v for k, v in row.items()} for row in X.to_dict(orient="records")]
