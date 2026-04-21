"""Tests for DynamoDB stream consumption via read_stream and stream_records."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from jsonflat.aws.dynamodb import read_stream, stream_records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_stream_record(
    event_id: str, event_name: str, new_image: dict | None = None, old_image: dict | None = None
) -> dict:
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


def _setup_mocks(mock_boto3: MagicMock, records: list) -> tuple[MagicMock, MagicMock]:
    """Wire up mock DynamoDB and Streams clients."""
    session = MagicMock()
    mock_boto3.Session.return_value = session

    ddb = MagicMock()
    streams = MagicMock()

    def client_factory(service, **kwargs):
        return ddb if service == "dynamodb" else streams

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
    """Tests for read_stream()."""

    @patch("jsonflat.aws.dynamodb.boto3")
    def test_new_images(self, mock_boto3) -> None:
        """Returns only new images; INSERT and MODIFY have NewImage, REMOVE does not."""
        _setup_mocks(mock_boto3, STREAM_RECORDS)

        df = read_stream(table_name="t", max_nesting=2, image="new")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "_event_name" in df.columns
        assert list(df["_event_name"]) == ["INSERT", "MODIFY"]

    @patch("jsonflat.aws.dynamodb.boto3")
    def test_old_images(self, mock_boto3) -> None:
        """Returns only old images; MODIFY and REMOVE have OldImage, INSERT does not."""
        _setup_mocks(mock_boto3, STREAM_RECORDS)

        df = read_stream(table_name="t", max_nesting=2, image="old")

        assert len(df) == 2
        assert list(df["_event_name"]) == ["MODIFY", "REMOVE"]

    @patch("jsonflat.aws.dynamodb.boto3")
    def test_both_images(self, mock_boto3) -> None:
        """Returns both images; INSERT=1, MODIFY=2, REMOVE=1 → 4 total rows."""
        _setup_mocks(mock_boto3, STREAM_RECORDS)

        df = read_stream(table_name="t", max_nesting=2, image="both")

        assert len(df) == 4

    @patch("jsonflat.aws.dynamodb.boto3")
    def test_unmarshalling(self, mock_boto3) -> None:
        """Typed values are unmarshalled from DynamoDB format to Python before flattening."""
        _setup_mocks(mock_boto3, STREAM_RECORDS[:1])

        df = read_stream(table_name="t", max_nesting=2, image="new")

        assert df.iloc[0]["id"] == "order-001"
        assert df.iloc[0]["customer__name"] == "Alice"
        assert df.iloc[0]["customer__age"] == 30

    @patch("jsonflat.aws.dynamodb.boto3")
    def test_filter_fn(self, mock_boto3) -> None:
        """filter_fn is applied to each image dict before flattening."""
        _setup_mocks(mock_boto3, STREAM_RECORDS)

        df = read_stream(
            table_name="t",
            image="new",
            filter_fn=lambda img: img.get("status") == "complete",
        )
        assert len(df) == 1

    @patch("jsonflat.aws.dynamodb.boto3")
    def test_no_stream_raises(self, mock_boto3) -> None:
        """Raises ValueError when the table has no stream enabled."""
        session = MagicMock()
        mock_boto3.Session.return_value = session

        ddb = MagicMock()
        session.client.return_value = ddb
        ddb.describe_table.return_value = {"Table": {}}

        with pytest.raises(ValueError, match="No stream enabled"):
            read_stream(table_name="t")

    @patch("jsonflat.aws.dynamodb.boto3")
    def test_empty_stream(self, mock_boto3) -> None:
        """Empty stream returns an empty DataFrame."""
        _setup_mocks(mock_boto3, [])

        df = read_stream(table_name="t", image="new")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestStreamRecords:
    """Tests for stream_records()."""

    @patch("jsonflat.aws.dynamodb.boto3")
    def test_yields_batches(self, mock_boto3) -> None:
        """Yields at least one DataFrame batch covering all matching records."""
        _setup_mocks(mock_boto3, STREAM_RECORDS)

        batches = list(stream_records(table_name="t", image="new"))
        assert batches
        total = sum(len(b) for b in batches)
        assert total == 2
