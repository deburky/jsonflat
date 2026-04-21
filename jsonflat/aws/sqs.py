"""
SQS integration for jsonflat -- consume JSON messages and flatten into DataFrames.

Usage:
    from jsonflat.aws.sqs import read_sqs

    # Poll and flatten up to 100 messages
    df = read_sqs(
        queue_url="https://sqs.us-east-1.amazonaws.com/123/my-queue",
        max_messages=100,
        max_nesting=3,
    )

    # Stream as batches
    from jsonflat.aws.sqs import stream_sqs

    for df_batch in stream_sqs(
        queue_url="https://sqs.us-east-1.amazonaws.com/123/my-queue",
        batch_size=10,
        max_nesting=3,
    ):
        process(df_batch)
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import boto3
import pandas as pd

from jsonflat.core import flatten


def read_sqs(
    queue_url: str,
    max_messages: int = 100,
    max_nesting: int | None = 3,
    wait_time: int = 5,
    filter_fn: Callable[[dict[str, Any]], bool] | None = None,
    profile_name: str | None = None,
    delete: bool = True,
) -> pd.DataFrame:
    """Poll SQS, flatten JSON messages, return a single DataFrame.

    :param queue_url: SQS queue URL
    :param max_messages: max number of messages to consume
    :param max_nesting: flatten depth (None = unlimited)
    :param wait_time: long-poll wait time in seconds (0-20)
    :param filter_fn: optional filter applied to parsed message before flattening
    :param profile_name: AWS profile name (None = default)
    :param delete: whether to delete messages after processing
    :returns: pandas DataFrame with flattened columns
    """
    records: list[pd.DataFrame] = []
    records.extend(
        iter(
            stream_sqs(
                queue_url=queue_url,
                batch_size=min(max_messages, 10),
                max_messages=max_messages,
                max_nesting=max_nesting,
                wait_time=wait_time,
                filter_fn=filter_fn,
                profile_name=profile_name,
                delete=delete,
            )
        )
    )
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def stream_sqs(
    queue_url: str,
    batch_size: int = 10,
    max_messages: int | None = None,
    max_nesting: int | None = 3,
    wait_time: int = 5,
    filter_fn: Callable[[dict[str, Any]], bool] | None = None,
    profile_name: str | None = None,
    delete: bool = True,
) -> Iterator[pd.DataFrame]:
    """Stream SQS messages as DataFrame batches.

    :param queue_url: SQS queue URL
    :param batch_size: messages per yielded DataFrame (max 10, SQS limit)
    :param max_messages: stop after this many messages (None = drain queue)
    :param max_nesting: flatten depth (None = unlimited)
    :param wait_time: long-poll wait time in seconds (0-20)
    :param filter_fn: optional filter applied to parsed message before flattening
    :param profile_name: AWS profile name (None = default)
    :param delete: whether to delete messages after processing
    :yields: pandas DataFrame with flattened columns, one per batch
    """
    session = boto3.Session(profile_name=profile_name)
    sqs = session.client("sqs")
    batch_size = min(batch_size, 10)  # SQS max per receive

    total = 0
    empty_polls = 0

    while max_messages is None or total < max_messages:
        receive_count = batch_size
        if max_messages is not None:
            receive_count = min(batch_size, max_messages - total)

        resp = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=receive_count,
            WaitTimeSeconds=wait_time,
        )

        messages = resp.get("Messages", [])
        if not messages:
            empty_polls += 1
            if empty_polls >= 2:
                break
            continue

        empty_polls = 0
        records: list[dict[str, Any]] = []
        receipt_handles: list[str] = []

        for msg in messages:
            try:
                data = json.loads(msg["Body"])
            except (json.JSONDecodeError, KeyError):
                continue

            if filter_fn and not filter_fn(data):
                receipt_handles.append(msg["ReceiptHandle"])
                continue

            data["_message_id"] = msg["MessageId"]
            records.append(flatten(data, max_nesting))
            receipt_handles.append(msg["ReceiptHandle"])

        if delete and receipt_handles:
            entries = [{"Id": str(i), "ReceiptHandle": rh} for i, rh in enumerate(receipt_handles)]
            sqs.delete_message_batch(QueueUrl=queue_url, Entries=entries)

        total += len(messages)

        if records:
            yield pd.DataFrame(records)
