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
from typing import Any, Literal

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from jsonflat.core import normalize_json


class JsonFlattener(BaseEstimator, TransformerMixin):
    """Flatten nested JSON records into a DataFrame for use in sklearn pipelines.

    :param max_nesting: max depth before storing as JSON blob (None = unlimited)
    :param column: when input is a DataFrame, the column containing nested dicts or JSON strings;
                   if None, treats each row as a dict
    :param table: spine table returned from normalize_json (default: ``"main"``)
    :param join: child table names to merge onto the spine; requires ``on``
    :param on: foreign-key column used both to link child rows during normalization and
               to join them back onto the spine; child columns are prefixed with
               ``<child_name>__`` to avoid collisions (the key column itself is not prefixed)
    :param how: pandas merge strategy — ``"left"`` (default), ``"inner"``, ``"outer"``
    """

    def __init__(
        self,
        max_nesting: int | None = 3,
        column: str | None = None,
        table: str = "main",
        join: list[str] | None = None,
        on: str | None = None,
        how: Literal["left", "right", "outer", "inner", "cross"] = "left",
    ) -> None:
        self.max_nesting = max_nesting
        self.column = column
        self.table = table
        self.join = join
        self.on = on
        self.how = how

    def fit(
        self,
        X: list[dict[str, Any]] | pd.DataFrame,
        y: Any = None,
    ) -> JsonFlattener:
        """Learn the column schema from training data.

        Runs the full flattening and join pipeline on ``X`` and stores the
        resulting column order as ``columns_``.  Subsequent calls to
        :meth:`transform` reindex their output to this schema.
        """
        records = self._extract_records(X)
        self.columns_: list[str] = self._build(records).columns.tolist()
        return self

    def fit_transform(
        self,
        X: Any,
        y: Any = None,
        **fit_params: Any,
    ) -> Any:
        """Fit on ``X``, then transform ``X`` — shorthand for ``fit(X).transform(X)``."""
        return self.fit(X, y).transform(X)

    def transform(
        self,
        X: list[dict[str, Any]] | pd.DataFrame,
    ) -> pd.DataFrame:
        """Flatten records and reindex to the schema learned during :meth:`fit`.

        Missing columns are filled with ``NaN``; columns not seen during fit
        are dropped.  Raises :class:`sklearn.exceptions.NotFittedError` if
        called before :meth:`fit`.

        :param X: list of dicts or a DataFrame; if a DataFrame and ``column`` is set,
                  that column is used as the source of nested records
        :returns: DataFrame with columns matching the fitted schema
        """
        check_is_fitted(self)
        columns: list[str] = getattr(self, "columns_")
        records = self._extract_records(X)
        return self._build(records).reindex(columns=columns)

    def _build(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        """Run normalize_json and optional joins; return the raw result before schema enforcement."""
        if self.join and self.on is None:
            raise ValueError("'on' must be specified when 'join' is used")

        tables = normalize_json(records, max_nesting=self.max_nesting, key=self.on)

        if self.table not in tables:
            available = list(tables.keys())
            raise KeyError(f"Table '{self.table}' not found. Available: {available}")

        result = pd.DataFrame(tables[self.table])

        for child_name in self.join or []:
            if child_name not in tables:
                available = list(tables.keys())
                raise KeyError(f"Table '{child_name}' not found. Available: {available}")
            child_df = pd.DataFrame(tables[child_name])
            child_df = child_df.rename(columns={c: f"{child_name}__{c}" for c in child_df.columns if c != self.on})
            result = result.merge(child_df, on=self.on, how=self.how)

        return result

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
