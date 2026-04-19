from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd

from jsonflat.integrations.stepfunctions import read_execution_history, read_executions

EXECUTIONS = [
    {
        "executionArn": "arn:aws:states:us-east-1:123:execution:sm:exec-1",
        "name": "exec-1",
        "status": "SUCCEEDED",
        "startDate": "2026-01-01T00:00:00Z",
        "stopDate": "2026-01-01T00:00:01Z",
    },
    {
        "executionArn": "arn:aws:states:us-east-1:123:execution:sm:exec-2",
        "name": "exec-2",
        "status": "FAILED",
        "startDate": "2026-01-01T00:01:00Z",
        "stopDate": "2026-01-01T00:01:05Z",
    },
]

DESCRIBE_RESPONSES = {
    "arn:aws:states:us-east-1:123:execution:sm:exec-1": {
        "input": json.dumps({"order_id": "o1", "customer": {"name": "Alice"}}),
        "output": json.dumps({"result": {"status": "ok", "score": 0.95}}),
    },
    "arn:aws:states:us-east-1:123:execution:sm:exec-2": {
        "input": json.dumps({"order_id": "o2", "customer": {"name": "Bob"}}),
        "error": "TaskFailed",
        "cause": "timeout",
    },
}

HISTORY_EVENTS = [
    {
        "timestamp": "2026-01-01T00:00:00Z",
        "type": "ExecutionStarted",
        "id": 1,
        "previousEventId": 0,
        "executionStartedEventDetails": {
            "input": json.dumps({"order_id": "o1"}),
        },
    },
    {
        "timestamp": "2026-01-01T00:00:01Z",
        "type": "ExecutionSucceeded",
        "id": 2,
        "previousEventId": 1,
        "executionSucceededEventDetails": {
            "output": json.dumps({"result": "ok"}),
        },
    },
]


def _setup_mocks(mock_boto3):
    session = MagicMock()
    mock_boto3.Session.return_value = session

    client = MagicMock()
    session.client.return_value = client

    paginator = MagicMock()
    client.get_paginator.return_value = paginator
    paginator.paginate.return_value = [{"executions": EXECUTIONS}]

    def describe_execution(executionArn):
        return DESCRIBE_RESPONSES[executionArn]

    client.describe_execution.side_effect = describe_execution

    return client


class TestReadExecutions:
    @patch("jsonflat.integrations.stepfunctions.boto3")
    def test_basic_read(self, mock_boto3):
        _setup_mocks(mock_boto3)

        df = read_executions(
            state_machine_arn="arn:aws:states:us-east-1:123:stateMachine:sm",
            max_nesting=3,
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "input__order_id" in df.columns
        assert "input__customer__name" in df.columns

    @patch("jsonflat.integrations.stepfunctions.boto3")
    def test_output_flattened(self, mock_boto3):
        _setup_mocks(mock_boto3)

        df = read_executions(
            state_machine_arn="arn:aws:states:us-east-1:123:stateMachine:sm",
        )

        succeeded = df[df["status"] == "SUCCEEDED"].iloc[0]
        assert succeeded["output__result__status"] == "ok"
        assert succeeded["output__result__score"] == 0.95

    @patch("jsonflat.integrations.stepfunctions.boto3")
    def test_error_captured(self, mock_boto3):
        _setup_mocks(mock_boto3)

        df = read_executions(
            state_machine_arn="arn:aws:states:us-east-1:123:stateMachine:sm",
        )

        failed = df[df["status"] == "FAILED"].iloc[0]
        assert failed["error"] == "TaskFailed"
        assert failed["cause"] == "timeout"

    @patch("jsonflat.integrations.stepfunctions.boto3")
    def test_filter_fn(self, mock_boto3):
        _setup_mocks(mock_boto3)

        df = read_executions(
            state_machine_arn="arn:aws:states:us-east-1:123:stateMachine:sm",
            filter_fn=lambda r: r.get("status") == "SUCCEEDED",
        )

        assert len(df) == 1

    @patch("jsonflat.integrations.stepfunctions.boto3")
    def test_max_executions(self, mock_boto3):
        _setup_mocks(mock_boto3)

        df = read_executions(
            state_machine_arn="arn:aws:states:us-east-1:123:stateMachine:sm",
            max_executions=1,
        )

        assert len(df) == 1


class TestReadExecutionHistory:
    @patch("jsonflat.integrations.stepfunctions.boto3")
    def test_basic(self, mock_boto3):
        session = MagicMock()
        mock_boto3.Session.return_value = session

        client = MagicMock()
        session.client.return_value = client

        paginator = MagicMock()
        client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [{"events": HISTORY_EVENTS}]

        df = read_execution_history(
            execution_arn="arn:aws:states:us-east-1:123:execution:sm:exec-1",
        )

        assert len(df) == 2
        assert "type" in df.columns
        assert "id" in df.columns
