"""
AWS Lambda handler for jsonflat -- flattens JSON from S3 and writes parquet back.

Supports two trigger modes:

1. S3 event trigger:
   Fires on new .json uploads. Flattens the file and writes
   a .parquet to the output prefix (env var OUTPUT_PREFIX).

2. Direct invocation:
   Invoke with {"bucket": "...", "prefix": "...", "max_nesting": 3}
   to batch-process files under a prefix.

Environment variables:
    OUTPUT_PREFIX: S3 key prefix for parquet output (default: "jsonflat-output/")
    MAX_NESTING: default flatten depth (default: 3)
"""

from __future__ import annotations

import io
import os
from typing import Any

import boto3
import orjson
import pandas as pd

from jsonflat.core import flatten

s3 = boto3.client("s3")

OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "jsonflat-output/")
DEFAULT_MAX_NESTING = int(os.environ.get("MAX_NESTING", "3"))


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Lambda entry point. Detects S3 event vs direct invoke."""
    if "Records" in event:
        return _handle_s3_event(event)
    return _handle_invoke(event)


def _handle_s3_event(event: dict[str, Any]) -> dict[str, Any]:
    """Process a single S3 object from an event notification."""
    results = []
    for record in event["Records"]:
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]

        if not key.endswith(".json"):
            results.append({"key": key, "status": "skipped", "reason": "not .json"})
            continue

        resp = s3.get_object(Bucket=bucket, Key=key)
        data = orjson.loads(resp["Body"].read())
        flat = flatten(data, DEFAULT_MAX_NESTING)
        flat["_source_file"] = key

        df = pd.DataFrame([flat])
        output_key = OUTPUT_PREFIX + key.rsplit("/", 1)[-1].replace(".json", ".parquet")
        _write_parquet(df, bucket, output_key)

        results.append({"key": key, "output": output_key, "status": "ok"})

    return {"processed": len(results), "results": results}


def _handle_invoke(event: dict[str, Any]) -> dict[str, Any]:
    """Batch-process files under a prefix."""
    bucket = event["bucket"]
    prefix = event.get("prefix", "")
    max_nesting = event.get("max_nesting", DEFAULT_MAX_NESTING)
    max_files = event.get("max_files", 1000)
    suffix = event.get("suffix", ".json")
    output_key = event.get("output_key", f"{OUTPUT_PREFIX}batch_result.parquet")

    keys = _list_keys(bucket, prefix, suffix, max_files)
    records = []
    for key in keys:
        try:
            resp = s3.get_object(Bucket=bucket, Key=key)
            data = orjson.loads(resp["Body"].read())
            flat = flatten(data, max_nesting)
            flat["_source_file"] = key
            records.append(flat)
        except Exception as e:
            print(f"Skipping {key}: {e}")

    if records:
        df = pd.DataFrame(records)
        _write_parquet(df, bucket, output_key)
    else:
        df = pd.DataFrame()

    return {
        "files_found": len(keys),
        "records": len(records),
        "output": f"s3://{bucket}/{output_key}" if records else None,
        "columns": list(df.columns) if not df.empty else [],
    }


def _list_keys(bucket: str, prefix: str, suffix: str, max_files: int) -> list[str]:
    """List S3 keys matching prefix and suffix."""
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(suffix):
                keys.append(obj["Key"])
                if len(keys) >= max_files:
                    return keys
    return keys


def _write_parquet(df: pd.DataFrame, bucket: str, key: str) -> None:
    """Write a DataFrame as parquet to S3."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
