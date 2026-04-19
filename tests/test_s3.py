from __future__ import annotations

from unittest.mock import MagicMock, patch

import orjson
import pandas as pd

from jsonflat.integrations.s3 import read_s3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_s3_page(keys: list[str]) -> dict:
    return {"Contents": [{"Key": k} for k in keys]}


def _make_s3_body(data: dict) -> MagicMock:
    body = MagicMock()
    body.read.return_value = orjson.dumps(data)
    return body


RECORDS = [
    {"id": 1, "info": {"name": "Alice", "score": 90}},
    {"id": 2, "info": {"name": "Bob", "score": 80}},
    {"id": 3, "info": {"name": "Carol", "score": 70}},
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestReadS3:
    @patch("jsonflat.integrations.s3.boto3")
    def test_basic_read(self, mock_boto3):
        keys = ["data/1.json", "data/2.json", "data/3.json"]
        s3 = MagicMock()
        mock_boto3.Session.return_value.client.return_value = s3

        paginator = MagicMock()
        paginator.paginate.return_value = [_make_s3_page(keys)]
        s3.get_paginator.return_value = paginator

        s3.get_object.side_effect = [{"Body": _make_s3_body(r)} for r in RECORDS]

        df = read_s3(bucket="b", prefix="data/", max_nesting=2)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "info__name" in df.columns
        assert "_source_file" in df.columns

    @patch("jsonflat.integrations.s3.boto3")
    def test_max_files_limits_keys(self, mock_boto3):
        keys = [f"data/{i}.json" for i in range(10)]
        s3 = MagicMock()
        mock_boto3.Session.return_value.client.return_value = s3

        paginator = MagicMock()
        paginator.paginate.return_value = [_make_s3_page(keys)]
        s3.get_paginator.return_value = paginator

        s3.get_object.side_effect = [{"Body": _make_s3_body({"id": i})} for i in range(2)]

        df = read_s3(bucket="b", prefix="data/", max_files=2)
        assert len(df) == 2

    @patch("jsonflat.integrations.s3.boto3")
    def test_suffix_filter(self, mock_boto3):
        keys = ["a.json", "b.csv", "c.json"]
        s3 = MagicMock()
        mock_boto3.Session.return_value.client.return_value = s3

        paginator = MagicMock()
        paginator.paginate.return_value = [_make_s3_page(keys)]
        s3.get_paginator.return_value = paginator

        s3.get_object.side_effect = [{"Body": _make_s3_body({"id": i})} for i in range(2)]

        df = read_s3(bucket="b", prefix="", suffix=".json")
        # Only a.json and c.json match
        assert len(df) == 2
        assert s3.get_object.call_count == 2

    @patch("jsonflat.integrations.s3.boto3")
    def test_filter_fn(self, mock_boto3):
        keys = ["1.json", "2.json"]
        s3 = MagicMock()
        mock_boto3.Session.return_value.client.return_value = s3

        paginator = MagicMock()
        paginator.paginate.return_value = [_make_s3_page(keys)]
        s3.get_paginator.return_value = paginator

        s3.get_object.side_effect = [
            {"Body": _make_s3_body({"id": 1, "status": "complete"})},
            {"Body": _make_s3_body({"id": 2, "status": "pending"})},
        ]

        df = read_s3(
            bucket="b",
            prefix="",
            filter_fn=lambda d: d.get("status") == "complete",
        )
        assert len(df) == 1
        assert df.iloc[0]["id"] == 1

    @patch("jsonflat.integrations.s3.boto3")
    def test_fetch_error_skipped(self, mock_boto3):
        keys = ["ok.json", "bad.json"]
        s3 = MagicMock()
        mock_boto3.Session.return_value.client.return_value = s3

        paginator = MagicMock()
        paginator.paginate.return_value = [_make_s3_page(keys)]
        s3.get_paginator.return_value = paginator

        s3.get_object.side_effect = [
            {"Body": _make_s3_body({"id": 1})},
            Exception("network error"),
        ]

        df = read_s3(bucket="b", prefix="")
        assert len(df) == 1

    @patch("jsonflat.integrations.s3.boto3")
    def test_empty_bucket(self, mock_boto3):
        s3 = MagicMock()
        mock_boto3.Session.return_value.client.return_value = s3

        paginator = MagicMock()
        paginator.paginate.return_value = [{"Contents": []}]
        s3.get_paginator.return_value = paginator

        df = read_s3(bucket="b", prefix="empty/")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
