"""
S3 integration for jsonflat -- sync and async readers for JSON files.

Usage:
    from jsonflat.aws.s3 import read_s3

    df = read_s3(
        bucket="my-bucket",
        prefix="data/",
        max_files=100,
        max_nesting=3,
    )

    # Async streaming
    from jsonflat.aws.s3 import read_s3_async

    async for df_batch in read_s3_async(
        bucket="my-bucket",
        prefix="events/2025/",
        max_nesting=3,
        batch_size=100,
    ):
        process(df_batch)

CLI:
    python -m jsonflat.integrations.s3 --bucket my-bucket --prefix data/ \\
        --max-files 100 --nesting 3 --output result.parquet
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import boto3
import orjson
import pandas as pd
from botocore.config import Config

try:
    import aioboto3
except ImportError:
    aioboto3 = None  # type: ignore[assignment]

from jsonflat.core import flatten


def read_s3(
    bucket: str,
    prefix: str,
    max_files: int = 100,
    max_nesting: int | None = 3,
    max_workers: int = 50,
    suffix: str = ".json",
    filter_fn: Callable[[dict], bool] | None = None,
    profile_name: str | None = None,
) -> pd.DataFrame:
    """Read JSON files from S3 in parallel, flatten, return DataFrame.

    Args:
        bucket: S3 bucket name
        prefix: S3 key prefix
        max_files: max number of files to process
        max_nesting: flatten depth (None = unlimited)
        max_workers: concurrent S3 downloads
        suffix: file suffix filter
        filter_fn: optional filter applied to raw JSON before flattening
        profile_name: AWS profile name (None = default)

    Returns:
        pandas DataFrame with flattened columns
    """
    session = boto3.Session(profile_name=profile_name)
    s3_config = Config(
        max_pool_connections=max_workers,
        retries={"max_attempts": 3, "mode": "adaptive"},
    )
    s3 = session.client("s3", config=s3_config)

    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(suffix):
                keys.append(obj["Key"])
                if len(keys) >= max_files:
                    break
        if len(keys) >= max_files:
            break

    print(f"Found {len(keys)} files")

    def fetch_one(key: str) -> dict[str, Any] | None:
        try:
            resp = s3.get_object(Bucket=bucket, Key=key)
            data = orjson.loads(resp["Body"].read())
            if filter_fn and not filter_fn(data):
                return None
            data["_source_file"] = key
            return flatten(data, max_nesting)
        except Exception as e:
            print(f"Skipping {key}: {e}")
            return None

    records = []
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_one, k): k for k in keys}
        for future in as_completed(futures):
            result = future.result()
            if result:
                records.append(result)
            done += 1
            if done % 100 == 0:
                print(f"Processed {done}/{len(keys)} files, {len(records)} records")

    print(f"Done: {len(records)} records from {len(keys)} files")
    return pd.DataFrame(records)


async def read_s3_async(
    bucket: str,
    prefix: str,
    max_nesting: int | None = 3,
    batch_size: int = 100,
    max_concurrency: int = 50,
    suffix: str = ".json",
    filter_fn: Callable[[dict], bool] | None = None,
    profile_name: str | None = None,
) -> AsyncIterator[pd.DataFrame]:
    """Async generator that yields DataFrames in batches from S3 JSON files.

    Args:
        bucket: S3 bucket name
        prefix: S3 key prefix
        max_nesting: flatten depth (None = unlimited)
        batch_size: number of files per yielded DataFrame
        max_concurrency: max concurrent S3 downloads per batch
        suffix: file suffix filter
        filter_fn: optional filter applied to raw JSON before flattening
        profile_name: AWS profile name (None = default)

    Yields:
        pandas DataFrame with flattened columns, one per batch
    """
    if aioboto3 is None:
        raise ImportError("aioboto3 is required: pip install jsonflat[s3-async]")

    session = aioboto3.Session(profile_name=profile_name)
    semaphore = asyncio.Semaphore(max_concurrency)

    # List all matching keys using async paginator
    keys: list[str] = []
    async with session.client("s3") as s3:
        paginator = s3.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(suffix):
                    keys.append(obj["Key"])

    # Process keys in batches
    for i in range(0, len(keys), batch_size):
        batch_keys = keys[i : i + batch_size]

        async def fetch_one(key: str, s3_client: Any) -> dict[str, Any] | None:
            async with semaphore:
                try:
                    resp = await s3_client.get_object(Bucket=bucket, Key=key)
                    body = await resp["Body"].read()
                    data = orjson.loads(body)
                    if filter_fn and not filter_fn(data):
                        return None
                    data["_source_file"] = key
                    return flatten(data, max_nesting)
                except Exception as e:
                    print(f"Skipping {key}: {e}")
                    return None

        async with session.client("s3") as s3:
            tasks = [fetch_one(k, s3) for k in batch_keys]
            results = await asyncio.gather(*tasks)

        records = [r for r in results if r is not None]
        if records:
            yield pd.DataFrame(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flatten S3 JSON files to DataFrame")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--max-files", type=int, default=100)
    parser.add_argument("--nesting", type=int, default=None)
    parser.add_argument("--suffix", type=str, default=".json")
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, help="Save to parquet file")
    args = parser.parse_args()

    df = read_s3(
        bucket=args.bucket,
        prefix=args.prefix,
        max_files=args.max_files,
        max_nesting=args.nesting,
        max_workers=args.workers,
        suffix=args.suffix,
        profile_name=args.profile,
    )

    print(f"\nDataFrame: {len(df)} rows, {len(df.columns)} columns")

    if args.output:
        df.to_parquet(args.output, index=False)
        print(f"Saved to {args.output}")
    else:
        print(df.head())
