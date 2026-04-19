from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd

from jsonflat.integrations.aws.cloudwatch import read_logs


def _make_event(message: str, stream: str = "stream-1", ts: int = 1000):
    return {
        "logStreamName": stream,
        "timestamp": ts,
        "ingestionTime": ts + 100,
        "message": message,
    }


LOG_EVENTS = [
    _make_event(json.dumps({"level": "INFO", "msg": "started", "req": {"id": "r1"}}), ts=1000),
    _make_event(json.dumps({"level": "ERROR", "msg": "failed", "req": {"id": "r2"}}), ts=2000),
    _make_event("plain text log line", ts=3000),
    _make_event(json.dumps({"level": "INFO", "msg": "done", "req": {"id": "r3"}}), ts=4000),
]


def _setup_mocks(mock_boto3, events):
    session = MagicMock()
    mock_boto3.Session.return_value = session

    client = MagicMock()
    session.client.return_value = client

    paginator = MagicMock()
    client.get_paginator.return_value = paginator
    paginator.paginate.return_value = [{"events": events}]

    return client


class TestReadLogs:
    @patch("jsonflat.integrations.aws.cloudwatch.boto3")
    def test_basic_read(self, mock_boto3):
        _setup_mocks(mock_boto3, LOG_EVENTS)

        df = read_logs(log_group="/aws/lambda/test", max_nesting=2)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert "_log_stream" in df.columns
        assert "_timestamp" in df.columns

    @patch("jsonflat.integrations.aws.cloudwatch.boto3")
    def test_json_parsed(self, mock_boto3):
        _setup_mocks(mock_boto3, LOG_EVENTS[:1])

        df = read_logs(log_group="/aws/lambda/test", max_nesting=2)

        assert df.iloc[0]["level"] == "INFO"
        assert df.iloc[0]["msg"] == "started"
        assert df.iloc[0]["req__id"] == "r1"

    @patch("jsonflat.integrations.aws.cloudwatch.boto3")
    def test_plain_text_fallback(self, mock_boto3):
        _setup_mocks(mock_boto3, [LOG_EVENTS[2]])

        df = read_logs(log_group="/aws/lambda/test")

        assert df.iloc[0]["_raw_message"] == "plain text log line"

    @patch("jsonflat.integrations.aws.cloudwatch.boto3")
    def test_max_events(self, mock_boto3):
        _setup_mocks(mock_boto3, LOG_EVENTS)

        df = read_logs(log_group="/aws/lambda/test", max_events=2)

        assert len(df) == 2

    @patch("jsonflat.integrations.aws.cloudwatch.boto3")
    def test_filter_fn(self, mock_boto3):
        _setup_mocks(mock_boto3, LOG_EVENTS)

        df = read_logs(
            log_group="/aws/lambda/test",
            filter_fn=lambda d: d.get("level") == "ERROR",
        )

        assert len(df) == 1
        assert df.iloc[0]["msg"] == "failed"

    @patch("jsonflat.integrations.aws.cloudwatch.boto3")
    def test_empty_logs(self, mock_boto3):
        _setup_mocks(mock_boto3, [])

        df = read_logs(log_group="/aws/lambda/test")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
