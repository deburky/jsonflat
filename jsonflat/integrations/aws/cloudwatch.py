"""CloudWatch Logs integration for jsonflat.

Usage:
    from jsonflat.aws.cloudwatch import read_logs

    df = read_logs(
        log_group="/aws/lambda/my-function",
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


def read_logs(
    log_group: str,
    max_nesting: int | None = 3,
    max_events: int | None = 1000,
    start_time: int | None = None,
    end_time: int | None = None,
    filter_pattern: str | None = None,
    filter_fn: Callable[[dict], bool] | None = None,
    profile_name: str | None = None,
    region_name: str | None = None,
) -> pd.DataFrame:
    """Read CloudWatch log events, parse JSON, flatten, return DataFrame.

    Args:
        log_group: CloudWatch log group name.
        max_nesting: flatten depth (None = unlimited).
        max_events: stop after this many events (None = read all).
        start_time: start timestamp in ms (None = no lower bound).
        end_time: end timestamp in ms (None = no upper bound).
        filter_pattern: CloudWatch filter pattern (e.g. '{ $.level = "ERROR" }').
        filter_fn: optional filter on parsed JSON before flattening.
        profile_name: AWS profile name (None = default).
        region_name: AWS region (None = default).

    Returns:
        pandas DataFrame with flattened columns.
    """
    session = boto3.Session(profile_name=profile_name, region_name=region_name)
    client = session.client("logs")

    kwargs: dict[str, Any] = {"logGroupName": log_group, "interleaved": True}
    if start_time is not None:
        kwargs["startTime"] = start_time
    if end_time is not None:
        kwargs["endTime"] = end_time
    if filter_pattern:
        kwargs["filterPattern"] = filter_pattern

    records: list[dict[str, Any]] = []
    paginator = client.get_paginator("filter_log_events")

    for page in paginator.paginate(**kwargs):
        for event in page.get("events", []):
            message = event.get("message", "").strip()

            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                # Not JSON — store raw message
                data = {"_raw_message": message}

            if filter_fn and not filter_fn(data):
                continue

            flat = flatten(data, max_nesting)
            flat["_log_stream"] = event.get("logStreamName", "")
            flat["_timestamp"] = event.get("timestamp", 0)
            flat["_ingestion_time"] = event.get("ingestionTime", 0)
            records.append(flat)

            if max_events is not None and len(records) >= max_events:
                return pd.DataFrame(records)

    return pd.DataFrame(records)
