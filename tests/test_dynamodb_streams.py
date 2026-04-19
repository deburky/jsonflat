from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from jsonflat.integrations.aws.dynamodb import read_stream, stream_records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_stream_record(event_id: str, event_name: str, new_image: dict | None = None, old_image: dict | None = None):
    dynamodb = {}
    if new_image:
        dynamodb["NewImage"] = new_image
    if old_image:
        dynamodb["OldImage"] = old_image
    return {
        "eventID": event_id,
        "eventName": event_name,
        "dynamodb": dynamodb,
    }


# DynamoDB JSON format
STREAM_RECORDS = [
    _make_stream_record(
        "e1",
        "INSERT",
        new_image={
            "id": {"S": "order-001"},
            "customer": {"M": {"name": {"S": "Alice"}, "age": {"N": "30"}}},
            "status": {"S": "complete"},
        },
    ),
    _make_stream_record(
        "e2",
        "MODIFY",
        new_image={"id": {"S": "order-002"}, "status": {"S": "shipped"}},
        old_image={"id": {"S": "order-002"}, "status": {"S": "pending"}},
    ),
    _make_stream_record(
        "e3",
        "REMOVE",
        old_image={"id": {"S": "order-003"}, "status": {"S": "cancelled"}},
    ),
]


def _setup_mocks(mock_boto3, records):
    """Wire up mock DynamoDB and Streams clients."""
    session = MagicMock()
    mock_boto3.Session.return_value = session

    ddb = MagicMock()
    streams = MagicMock()

    def client_factory(service, **kwargs):
        if service == "dynamodb":
            return ddb
        return streams

    session.client.side_effect = client_factory

    ddb.describe_table.return_value = {
        "Table": {"LatestStreamArn": "arn:aws:dynamodb:us-east-1:123:table/t/stream/123"}
    }

    streams.describe_stream.return_value = {
        "StreamDescription": {
            "Shards": [{"ShardId": "shard-001"}],
        }
    }

    streams.get_shard_iterator.return_value = {"ShardIterator": "iter-001"}
    streams.get_records.side_effect = [
        {"Records": records, "NextShardIterator": None},
    ]

    return ddb, streams


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestReadStream:
    @patch("jsonflat.integrations.aws.dynamodb.boto3")
    def test_new_images(self, mock_boto3):
        _setup_mocks(mock_boto3, STREAM_RECORDS)

        df = read_stream(table_name="t", max_nesting=2, image="new")

        assert isinstance(df, pd.DataFrame)
        # INSERT has NewImage, MODIFY has NewImage, REMOVE has no NewImage
        assert len(df) == 2
        assert "_event_name" in df.columns
        assert list(df["_event_name"]) == ["INSERT", "MODIFY"]

    @patch("jsonflat.integrations.aws.dynamodb.boto3")
    def test_old_images(self, mock_boto3):
        _setup_mocks(mock_boto3, STREAM_RECORDS)

        df = read_stream(table_name="t", max_nesting=2, image="old")

        # MODIFY has OldImage, REMOVE has OldImage, INSERT has no OldImage
        assert len(df) == 2
        assert list(df["_event_name"]) == ["MODIFY", "REMOVE"]

    @patch("jsonflat.integrations.aws.dynamodb.boto3")
    def test_both_images(self, mock_boto3):
        _setup_mocks(mock_boto3, STREAM_RECORDS)

        df = read_stream(table_name="t", max_nesting=2, image="both")

        # INSERT: 1 new, MODIFY: 1 new + 1 old, REMOVE: 1 old = 4
        assert len(df) == 4

    @patch("jsonflat.integrations.aws.dynamodb.boto3")
    def test_unmarshalling(self, mock_boto3):
        _setup_mocks(mock_boto3, STREAM_RECORDS[:1])  # Just INSERT

        df = read_stream(table_name="t", max_nesting=2, image="new")

        assert df.iloc[0]["id"] == "order-001"
        assert df.iloc[0]["customer__name"] == "Alice"
        assert df.iloc[0]["customer__age"] == 30

    @patch("jsonflat.integrations.aws.dynamodb.boto3")
    def test_filter_fn(self, mock_boto3):
        _setup_mocks(mock_boto3, STREAM_RECORDS)

        df = read_stream(
            table_name="t",
            image="new",
            filter_fn=lambda img: img.get("status") == "complete",
        )
        assert len(df) == 1

    @patch("jsonflat.integrations.aws.dynamodb.boto3")
    def test_no_stream_raises(self, mock_boto3):
        session = MagicMock()
        mock_boto3.Session.return_value = session

        ddb = MagicMock()
        session.client.return_value = ddb
        ddb.describe_table.return_value = {"Table": {}}

        with pytest.raises(ValueError, match="No stream enabled"):
            read_stream(table_name="t")

    @patch("jsonflat.integrations.aws.dynamodb.boto3")
    def test_empty_stream(self, mock_boto3):
        _setup_mocks(mock_boto3, [])

        df = read_stream(table_name="t", image="new")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestStreamRecords:
    @patch("jsonflat.integrations.aws.dynamodb.boto3")
    def test_yields_batches(self, mock_boto3):
        _setup_mocks(mock_boto3, STREAM_RECORDS)

        batches = list(stream_records(table_name="t", image="new"))
        assert len(batches) >= 1
        total = sum(len(b) for b in batches)
        assert total == 2
