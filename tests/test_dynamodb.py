from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pandas as pd

from jsonflat.integrations.dynamodb import read_dynamodb

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ITEMS = [
    {"id": "order-001", "customer": {"name": "Alice", "age": Decimal("30")}, "status": "complete"},
    {"id": "order-002", "customer": {"name": "Bob", "age": Decimal("25")}, "status": "pending"},
    {"id": "order-003", "customer": {"name": "Carol", "age": Decimal("40")}, "status": "complete"},
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestReadDynamodb:
    @patch("jsonflat.integrations.dynamodb.boto3")
    def test_basic_scan(self, mock_boto3):
        table = MagicMock()
        mock_boto3.Session.return_value.resource.return_value.Table.return_value = table

        table.scan.return_value = {"Items": ITEMS}

        df = read_dynamodb(table_name="t", max_nesting=2)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "customer__name" in df.columns

    @patch("jsonflat.integrations.dynamodb.boto3")
    def test_pagination(self, mock_boto3):
        table = MagicMock()
        mock_boto3.Session.return_value.resource.return_value.Table.return_value = table

        table.scan.side_effect = [
            {"Items": ITEMS[:2], "LastEvaluatedKey": {"id": "order-002"}},
            {"Items": ITEMS[2:]},
        ]

        df = read_dynamodb(table_name="t", max_nesting=2)
        assert len(df) == 3
        assert table.scan.call_count == 2

    @patch("jsonflat.integrations.dynamodb.boto3")
    def test_max_items(self, mock_boto3):
        table = MagicMock()
        mock_boto3.Session.return_value.resource.return_value.Table.return_value = table

        table.scan.return_value = {"Items": ITEMS}

        df = read_dynamodb(table_name="t", max_items=2)
        assert len(df) == 2

    @patch("jsonflat.integrations.dynamodb.boto3")
    def test_filter_fn(self, mock_boto3):
        table = MagicMock()
        mock_boto3.Session.return_value.resource.return_value.Table.return_value = table

        table.scan.return_value = {"Items": ITEMS}

        df = read_dynamodb(
            table_name="t",
            filter_fn=lambda item: item.get("status") == "complete",
        )
        assert len(df) == 2

    @patch("jsonflat.integrations.dynamodb.boto3")
    def test_empty_table(self, mock_boto3):
        table = MagicMock()
        mock_boto3.Session.return_value.resource.return_value.Table.return_value = table

        table.scan.return_value = {"Items": []}

        df = read_dynamodb(table_name="t")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
