"""
DynamoDB integration for jsonflat -- scan tables and consume streams.

Usage:
    from jsonflat.aws.dynamodb import read_dynamodb

    # Scan entire table
    df = read_dynamodb(table_name="my-table", max_nesting=3)

    # Read stream changes
    from jsonflat.aws.dynamodb import read_stream, stream_records

    df = read_stream(table_name="my-table", image="new")

    for df_batch in stream_records(table_name="my-table", image="both"):
        process(df_batch)
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import boto3
import pandas as pd

from jsonflat.core import flatten


# ---------------------------------------------------------------------------
# Table scan
# ---------------------------------------------------------------------------
def read_dynamodb(
    table_name: str,
    max_nesting: int | None = 3,
    max_items: int | None = None,
    filter_fn: Callable[[dict], bool] | None = None,
    profile_name: str | None = None,
    region_name: str | None = None,
    **scan_kwargs: Any,
) -> pd.DataFrame:
    """Scan a DynamoDB table, flatten items, return a DataFrame.

    :param table_name: DynamoDB table name
    :param max_nesting: flatten depth (None = unlimited)
    :param max_items: stop after this many items (None = scan all)
    :param filter_fn: optional filter applied to each item before flattening
    :param profile_name: AWS profile name (None = default)
    :param region_name: AWS region (None = default)
    :param scan_kwargs: additional kwargs passed to ``table.scan()`` (e.g. FilterExpression)
    :returns: pandas DataFrame with flattened columns
    """
    session = boto3.Session(profile_name=profile_name, region_name=region_name)
    table = session.resource("dynamodb").Table(table_name)

    records: list[dict[str, Any]] = []
    last_key = None

    while True:
        kwargs: dict[str, Any] = {**scan_kwargs}
        if last_key:
            kwargs["ExclusiveStartKey"] = last_key

        resp = table.scan(**kwargs)

        for item in resp.get("Items", []):
            if filter_fn and not filter_fn(item):
                continue
            records.append(flatten(dict(item), max_nesting))
            if max_items is not None and len(records) >= max_items:
                return pd.DataFrame(records)

        last_key = resp.get("LastEvaluatedKey")
        if not last_key:
            break

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------
def read_stream(
    table_name: str,
    max_nesting: int | None = 3,
    max_records: int | None = None,
    image: str = "new",
    iterator_type: str = "TRIM_HORIZON",
    filter_fn: Callable[[dict], bool] | None = None,
    profile_name: str | None = None,
    region_name: str | None = None,
) -> pd.DataFrame:
    """Read DynamoDB stream records, flatten, return a single DataFrame.

    :param table_name: DynamoDB table name (stream ARN is resolved automatically)
    :param max_nesting: flatten depth (None = unlimited)
    :param max_records: stop after this many records (None = read all available)
    :param image: which image to flatten — ``"new"``, ``"old"``, or ``"both"``
    :param iterator_type: ``TRIM_HORIZON`` (all) or ``LATEST`` (new only)
    :param filter_fn: optional filter applied to the image dict before flattening
    :param profile_name: AWS profile name (None = default)
    :param region_name: AWS region (None = default)
    :returns: pandas DataFrame with flattened columns
    """
    batches = []
    for df_batch in stream_records(
        table_name=table_name,
        max_nesting=max_nesting,
        max_records=max_records,
        image=image,
        iterator_type=iterator_type,
        filter_fn=filter_fn,
        profile_name=profile_name,
        region_name=region_name,
    ):
        batches.append(df_batch)

    if batches:
        return pd.concat(batches, ignore_index=True)
    return pd.DataFrame()


def stream_records(
    table_name: str,
    max_nesting: int | None = 3,
    max_records: int | None = None,
    image: str = "new",
    iterator_type: str = "TRIM_HORIZON",
    filter_fn: Callable[[dict], bool] | None = None,
    profile_name: str | None = None,
    region_name: str | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield DataFrames from DynamoDB stream shards.

    :param table_name: DynamoDB table name
    :param max_nesting: flatten depth (None = unlimited)
    :param max_records: stop after this many records (None = read all available)
    :param image: which image to flatten — ``"new"``, ``"old"``, or ``"both"``
    :param iterator_type: ``TRIM_HORIZON`` or ``LATEST``
    :param filter_fn: optional filter on image dict before flattening
    :param profile_name: AWS profile name (None = default)
    :param region_name: AWS region (None = default)
    :yields: pandas DataFrame per shard with flattened columns
    """
    session = boto3.Session(profile_name=profile_name, region_name=region_name)
    ddb = session.client("dynamodb")
    streams = session.client("dynamodbstreams")

    # Resolve stream ARN from table
    desc = ddb.describe_table(TableName=table_name)
    stream_arn = desc["Table"].get("LatestStreamArn")
    if not stream_arn:
        raise ValueError(f"No stream enabled on table '{table_name}'")

    # Get shards
    stream_desc = streams.describe_stream(StreamArn=stream_arn)
    shards = stream_desc["StreamDescription"]["Shards"]

    total = 0
    for shard in shards:
        iterator_resp = streams.get_shard_iterator(
            StreamArn=stream_arn,
            ShardId=shard["ShardId"],
            ShardIteratorType=iterator_type,
        )
        shard_iterator = iterator_resp["ShardIterator"]

        while shard_iterator:
            resp = streams.get_records(ShardIterator=shard_iterator, Limit=100)
            raw_records = resp.get("Records", [])
            shard_iterator = resp.get("NextShardIterator")

            if not raw_records:
                break

            rows = _process_records(raw_records, image, max_nesting, filter_fn)
            total += len(raw_records)

            if rows:
                yield pd.DataFrame(rows)

            if max_records is not None and total >= max_records:
                return


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _process_records(
    records: list[dict[str, Any]],
    image: str,
    max_nesting: int | None,
    filter_fn: Callable[[dict], bool] | None,
) -> list[dict[str, Any]]:
    """Extract and flatten images from stream records."""
    rows: list[dict[str, Any]] = []

    for record in records:
        dynamodb = record.get("dynamodb", {})
        event_name = record.get("eventName", "")

        images = _extract_images(dynamodb, image)

        for img in images:
            if filter_fn and not filter_fn(img):
                continue
            flat = flatten(img, max_nesting)
            flat["_event_name"] = event_name
            flat["_event_id"] = record.get("eventID", "")
            rows.append(flat)

    return rows


def _extract_images(
    dynamodb: dict[str, Any],
    image: str,
) -> list[dict[str, Any]]:
    """Extract image dicts based on the image parameter."""
    images: list[dict[str, Any]] = []

    if image in ("new", "both"):
        new = dynamodb.get("NewImage")
        if new:
            images.append(_unmarshall(new))

    if image in ("old", "both"):
        old = dynamodb.get("OldImage")
        if old:
            images.append(_unmarshall(old))

    return images


def _unmarshall(item: dict[str, Any]) -> dict[str, Any]:
    """Convert DynamoDB JSON format to plain Python dicts."""
    result: dict[str, Any] = {}
    for key, value in item.items():
        result[key] = _unmarshall_value(value)
    return result


def _unmarshall_value(value: dict[str, Any]) -> Any:
    """Convert a single DynamoDB typed value to Python."""
    if "S" in value:
        return value["S"]
    if "N" in value:
        n = value["N"]
        return int(n) if "." not in n else float(n)
    if "BOOL" in value:
        return value["BOOL"]
    if "NULL" in value:
        return None
    if "L" in value:
        return [_unmarshall_value(v) for v in value["L"]]
    if "M" in value:
        return _unmarshall(value["M"])
    if "SS" in value:
        return set(value["SS"])
    if "NS" in value:
        return {int(n) if "." not in n else float(n) for n in value["NS"]}
    if "B" in value:
        return value["B"]
    if "BS" in value:
        return set(value["BS"])
    return value
