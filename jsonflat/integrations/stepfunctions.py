"""Step Functions integration for jsonflat.

Usage:
    from jsonflat.integrations.stepfunctions import read_executions

    df = read_executions(
        state_machine_arn="arn:aws:states:...:stateMachine:my-machine",
        max_nesting=3,
    )
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import boto3
import pandas as pd

from jsonflat.core import flatten


def read_executions(
    state_machine_arn: str,
    max_nesting: int | None = 3,
    max_executions: int | None = 100,
    status_filter: str | None = None,
    filter_fn: Callable[[dict], bool] | None = None,
    profile_name: str | None = None,
    region_name: str | None = None,
) -> pd.DataFrame:
    """Read Step Functions executions, flatten input/output, return DataFrame.

    Args:
        state_machine_arn: ARN of the state machine.
        max_nesting: flatten depth (None = unlimited).
        max_executions: stop after this many executions (None = read all).
        status_filter: filter by status (RUNNING, SUCCEEDED, FAILED, TIMED_OUT, ABORTED).
        filter_fn: optional filter on the execution record before flattening.
        profile_name: AWS profile name (None = default).
        region_name: AWS region (None = default).

    Returns:
        pandas DataFrame with flattened columns.
    """
    session = boto3.Session(profile_name=profile_name, region_name=region_name)
    client = session.client("stepfunctions")

    # List executions
    list_kwargs: dict[str, Any] = {"stateMachineArn": state_machine_arn}
    if status_filter:
        list_kwargs["statusFilter"] = status_filter

    executions: list[dict[str, Any]] = []
    paginator = client.get_paginator("list_executions")

    for page in paginator.paginate(**list_kwargs):
        for exc in page.get("executions", []):
            executions.append(exc)
            if max_executions is not None and len(executions) >= max_executions:
                break
        if max_executions is not None and len(executions) >= max_executions:
            break

    # Describe each execution to get input/output
    records: list[dict[str, Any]] = []
    for exc in executions:
        desc = client.describe_execution(executionArn=exc["executionArn"])

        record: dict[str, Any] = {
            "execution_arn": exc["executionArn"],
            "name": exc.get("name", ""),
            "status": exc.get("status", ""),
            "start_date": exc.get("startDate", ""),
            "stop_date": exc.get("stopDate", ""),
        }

        # Parse and flatten input
        raw_input = desc.get("input", "{}")
        try:
            input_data = json.loads(raw_input)
            for k, v in flatten(input_data, max_nesting).items():
                record[f"input__{k}"] = v
        except json.JSONDecodeError:
            record["input__raw"] = raw_input

        # Parse and flatten output
        raw_output = desc.get("output")
        if raw_output:
            try:
                output_data = json.loads(raw_output)
                for k, v in flatten(output_data, max_nesting).items():
                    record[f"output__{k}"] = v
            except json.JSONDecodeError:
                record["output__raw"] = raw_output

        # Flatten error if present
        error = desc.get("error")
        if error:
            record["error"] = error
            record["cause"] = desc.get("cause", "")

        if filter_fn and not filter_fn(record):
            continue

        records.append(record)

    return pd.DataFrame(records)


def read_execution_history(
    execution_arn: str,
    max_nesting: int | None = 3,
    profile_name: str | None = None,
    region_name: str | None = None,
) -> pd.DataFrame:
    """Read the event history of a single execution, flatten, return DataFrame.

    Args:
        execution_arn: ARN of the execution.
        max_nesting: flatten depth (None = unlimited).
        profile_name: AWS profile name (None = default).
        region_name: AWS region (None = default).

    Returns:
        pandas DataFrame with one row per history event.
    """
    session = boto3.Session(profile_name=profile_name, region_name=region_name)
    client = session.client("stepfunctions")

    records: list[dict[str, Any]] = []
    paginator = client.get_paginator("get_execution_history")

    for page in paginator.paginate(executionArn=execution_arn):
        for event in page.get("events", []):
            flat = flatten(event, max_nesting)
            records.append(flat)

    return pd.DataFrame(records)
