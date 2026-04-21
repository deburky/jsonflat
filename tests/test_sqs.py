"""Tests for SQS message consumption via read_sqs and stream_sqs."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd

from jsonflat.aws.sqs import read_sqs, stream_sqs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_message(msg_id: str, body: dict) -> dict:
    """Constructs a mock SQS message with JSON body."""
    return {
        "MessageId": msg_id,
        "ReceiptHandle": f"rh-{msg_id}",
        "Body": json.dumps(body),
    }


RECORDS = [
    {"id": 1, "info": {"name": "Alice", "score": 90}},
    {"id": 2, "info": {"name": "Bob", "score": 80}},
    {"id": 3, "info": {"name": "Carol", "score": 70}},
]


# ---------------------------------------------------------------------------
# Tests — read_sqs
# ---------------------------------------------------------------------------
class TestReadSqs:
    """Tests for read_sqs()."""

    @patch("jsonflat.aws.sqs.boto3")
    def test_basic_read(self, mock_boto3) -> None:
        """Polls queue and returns a flattened DataFrame with _message_id column."""
        sqs = MagicMock()
        mock_boto3.Session.return_value.client.return_value = sqs

        messages = [_make_message(f"m{i}", r) for i, r in enumerate(RECORDS)]
        sqs.receive_message.side_effect = [
            {"Messages": messages},
            {"Messages": []},
            {"Messages": []},
        ]

        df = read_sqs(queue_url="https://q", max_nesting=2)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "info__name" in df.columns
        assert "_message_id" in df.columns

    @patch("jsonflat.aws.sqs.boto3")
    def test_max_messages_limits(self, mock_boto3) -> None:
        """Stops consuming after max_messages regardless of queue depth."""
        sqs = MagicMock()
        mock_boto3.Session.return_value.client.return_value = sqs

        messages = [_make_message(f"m{i}", {"id": i}) for i in range(10)]
        sqs.receive_message.side_effect = [
            {"Messages": messages[:5]},
            {"Messages": messages[5:]},
        ]

        df = read_sqs(queue_url="https://q", max_messages=5)
        assert len(df) == 5

    @patch("jsonflat.aws.sqs.boto3")
    def test_filter_fn(self, mock_boto3) -> None:
        """filter_fn excludes messages before flattening."""
        sqs = MagicMock()
        mock_boto3.Session.return_value.client.return_value = sqs

        messages = [
            _make_message("m0", {"id": 1, "status": "complete"}),
            _make_message("m1", {"id": 2, "status": "pending"}),
        ]
        sqs.receive_message.side_effect = [
            {"Messages": messages},
            {"Messages": []},
            {"Messages": []},
        ]

        df = read_sqs(
            queue_url="https://q",
            filter_fn=lambda d: d.get("status") == "complete",
        )
        assert len(df) == 1
        assert df.iloc[0]["id"] == 1

    @patch("jsonflat.aws.sqs.boto3")
    def test_empty_queue(self, mock_boto3) -> None:
        """Empty queue returns an empty DataFrame."""
        sqs = MagicMock()
        mock_boto3.Session.return_value.client.return_value = sqs

        sqs.receive_message.return_value = {"Messages": []}

        df = read_sqs(queue_url="https://q")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    @patch("jsonflat.aws.sqs.boto3")
    def test_delete_called(self, mock_boto3) -> None:
        """delete=True calls delete_message_batch after receiving messages."""
        sqs = MagicMock()
        mock_boto3.Session.return_value.client.return_value = sqs

        messages = [_make_message("m0", {"id": 1})]
        sqs.receive_message.side_effect = [
            {"Messages": messages},
            {"Messages": []},
            {"Messages": []},
        ]

        read_sqs(queue_url="https://q", delete=True)
        sqs.delete_message_batch.assert_called_once()

    @patch("jsonflat.aws.sqs.boto3")
    def test_no_delete_when_disabled(self, mock_boto3) -> None:
        """delete=False leaves messages on the queue."""
        sqs = MagicMock()
        mock_boto3.Session.return_value.client.return_value = sqs

        messages = [_make_message("m0", {"id": 1})]
        sqs.receive_message.side_effect = [
            {"Messages": messages},
            {"Messages": []},
            {"Messages": []},
        ]

        read_sqs(queue_url="https://q", delete=False)
        sqs.delete_message_batch.assert_not_called()


# ---------------------------------------------------------------------------
# Tests — stream_sqs
# ---------------------------------------------------------------------------
class TestStreamSqs:
    """Tests for stream_sqs()."""

    @patch("jsonflat.aws.sqs.boto3")
    def test_batching(self, mock_boto3) -> None:
        """Yields one DataFrame per receive_message call, respecting batch_size."""
        sqs = MagicMock()
        mock_boto3.Session.return_value.client.return_value = sqs

        messages = [_make_message(f"m{i}", {"id": i}) for i in range(7)]
        sqs.receive_message.side_effect = [
            {"Messages": messages[:5]},
            {"Messages": messages[5:]},
            {"Messages": []},
            {"Messages": []},
        ]

        batches = list(stream_sqs(queue_url="https://q", batch_size=5))
        assert len(batches) == 2
        assert len(batches[0]) == 5
        assert len(batches[1]) == 2

    @patch("jsonflat.aws.sqs.boto3")
    def test_malformed_json_skipped(self, mock_boto3) -> None:
        """Messages with non-JSON bodies are silently skipped."""
        sqs = MagicMock()
        mock_boto3.Session.return_value.client.return_value = sqs

        messages = [
            _make_message("m0", {"id": 1}),
            {"MessageId": "m1", "ReceiptHandle": "rh", "Body": "not-json"},
        ]
        sqs.receive_message.side_effect = [
            {"Messages": messages},
            {"Messages": []},
            {"Messages": []},
        ]

        batches = list(stream_sqs(queue_url="https://q"))
        assert len(batches) == 1
        assert len(batches[0]) == 1
