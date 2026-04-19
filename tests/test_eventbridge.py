from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd

from jsonflat.integrations.eventbridge import flatten_event, read_events

EB_EVENT = {
    "version": "0",
    "id": "abc-123",
    "detail-type": "OrderCreated",
    "source": "myapp.orders",
    "account": "123456789",
    "time": "2026-01-01T00:00:00Z",
    "region": "us-east-1",
    "detail": {
        "order_id": "ord-001",
        "customer": {"name": "Alice", "tier": "gold"},
        "total": 150.0,
    },
}

SQS_MESSAGES = [
    {
        "MessageId": f"msg-{i}",
        "ReceiptHandle": f"rh-{i}",
        "Body": json.dumps(event),
    }
    for i, event in enumerate(
        [
            EB_EVENT,
            {**EB_EVENT, "id": "abc-456", "detail": {"order_id": "ord-002", "total": 75.0}},
        ]
    )
]


class TestFlattenEvent:
    def test_basic(self):
        flat = flatten_event(EB_EVENT)
        assert flat["source"] == "myapp.orders"
        assert flat["detail__order_id"] == "ord-001"
        assert flat["detail__customer__name"] == "Alice"

    def test_max_nesting(self):
        flat = flatten_event(EB_EVENT, max_nesting=1)
        # depth 1 flattens one level: detail__order_id exists, but detail__customer stays as dict
        assert flat["detail__order_id"] == "ord-001"
        assert isinstance(flat["detail__customer"], dict)


def _setup_mocks(mock_boto3, messages, empty_after=True):
    session = MagicMock()
    mock_boto3.Session.return_value = session

    sqs = MagicMock()
    session.client.return_value = sqs

    responses = [{"Messages": messages}]
    if empty_after:
        responses.append({"Messages": []})
    sqs.receive_message.side_effect = responses

    return sqs


class TestReadEvents:
    @patch("jsonflat.integrations.eventbridge.boto3")
    def test_basic_read(self, mock_boto3):
        _setup_mocks(mock_boto3, SQS_MESSAGES)

        df = read_events(queue_url="https://sqs.../q", max_nesting=3)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "detail__order_id" in df.columns
        assert "_message_id" in df.columns

    @patch("jsonflat.integrations.eventbridge.boto3")
    def test_delete_called(self, mock_boto3):
        sqs = _setup_mocks(mock_boto3, SQS_MESSAGES)

        read_events(queue_url="https://sqs.../q", delete=True)

        sqs.delete_message_batch.assert_called_once()

    @patch("jsonflat.integrations.eventbridge.boto3")
    def test_no_delete(self, mock_boto3):
        sqs = _setup_mocks(mock_boto3, SQS_MESSAGES)

        read_events(queue_url="https://sqs.../q", delete=False)

        sqs.delete_message_batch.assert_not_called()

    @patch("jsonflat.integrations.eventbridge.boto3")
    def test_filter_fn(self, mock_boto3):
        _setup_mocks(mock_boto3, SQS_MESSAGES)

        df = read_events(
            queue_url="https://sqs.../q",
            filter_fn=lambda e: e.get("detail", {}).get("total", 0) > 100,
        )

        assert len(df) == 1

    @patch("jsonflat.integrations.eventbridge.boto3")
    def test_empty_queue(self, mock_boto3):
        _setup_mocks(mock_boto3, [], empty_after=False)

        df = read_events(queue_url="https://sqs.../q")

        assert len(df) == 0
