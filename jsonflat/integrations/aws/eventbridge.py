"""EventBridge integration for jsonflat.

Usage:
    from jsonflat.aws.eventbridge import read_events, flatten_event

    # Flatten a single EventBridge event
    flat = flatten_event(event)

    # Read events from an SQS target queue (common pattern)
    df = read_events(queue_url="https://sqs.../my-eb-queue", max_nesting=3)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import boto3
import pandas as pd

from jsonflat.core import flatten


def flatten_event(
    event: dict[str, Any],
    max_nesting: int | None = None,
) -> dict[str, Any]:
    """Flatten a single EventBridge event into a flat dict.

    Args:
        event: EventBridge event dict (with source, detail-type, detail, etc.).
        max_nesting: flatten depth (None = unlimited).

    Returns:
        Flat dict with __ separated keys.
    """
    return flatten(event, max_nesting)


def read_events(
    queue_url: str,
    max_nesting: int | None = 3,
    max_events: int | None = 100,
    delete: bool = True,
    filter_fn: Callable[[dict], bool] | None = None,
    profile_name: str | None = None,
    region_name: str | None = None,
) -> pd.DataFrame:
    """Read EventBridge events from an SQS target queue, flatten, return DataFrame.

    A common pattern is routing EventBridge events to an SQS queue for
    consumption. This function reads from that queue, unwraps the
    EventBridge envelope, and flattens the event detail.

    Args:
        queue_url: SQS queue URL receiving EventBridge events.
        max_nesting: flatten depth (None = unlimited).
        max_events: stop after this many events (None = read all available).
        delete: delete messages from queue after reading.
        filter_fn: optional filter on the parsed event before flattening.
        profile_name: AWS profile name (None = default).
        region_name: AWS region (None = default).

    Returns:
        pandas DataFrame with flattened columns.
    """
    session = boto3.Session(profile_name=profile_name, region_name=region_name)
    sqs = session.client("sqs")

    records: list[dict[str, Any]] = []

    while max_events is None or len(records) < max_events:
        batch_size = min(10, (max_events or 10) - len(records))
        resp = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=batch_size,
            WaitTimeSeconds=1,
        )
        messages = resp.get("Messages", [])
        if not messages:
            break

        entries_to_delete = []
        for msg in messages:
            try:
                event = json.loads(msg["Body"])
            except (json.JSONDecodeError, KeyError):
                continue

            if filter_fn and not filter_fn(event):
                continue

            flat = flatten(event, max_nesting)
            flat["_message_id"] = msg.get("MessageId", "")
            records.append(flat)

            entries_to_delete.append({"Id": msg["MessageId"], "ReceiptHandle": msg["ReceiptHandle"]})

        if delete and entries_to_delete:
            sqs.delete_message_batch(QueueUrl=queue_url, Entries=entries_to_delete)

    return pd.DataFrame(records)
