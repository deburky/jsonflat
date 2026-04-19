from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pandas as pd
import pytest

from jsonflat.integrations.aws.s3 import read_s3_async


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_s3_body(data: dict) -> AsyncMock:
    body = AsyncMock()
    body.read.return_value = orjson.dumps(data)
    return body


def _make_paginator(keys: list[str]):
    """Create a paginator whose .paginate() returns an async iterator of pages."""
    page = {"Contents": [{"Key": k} for k in keys]}

    class AsyncPages:
        def __init__(self):
            self._pages = [page]
            self._idx = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._idx >= len(self._pages):
                raise StopAsyncIteration
            p = self._pages[self._idx]
            self._idx += 1
            return p

    paginator = MagicMock()
    paginator.paginate.return_value = AsyncPages()
    return paginator


def _make_s3_client(keys: list[str], responses: list[dict]):
    """Create a mock S3 client with sync get_paginator and async get_object."""
    call_count = 0

    # Use MagicMock so get_paginator is sync (matches real aioboto3)
    client = MagicMock()
    client.get_paginator.return_value = _make_paginator(keys)

    async def get_object(**kwargs):
        nonlocal call_count
        idx = call_count
        call_count += 1
        return responses[idx]

    client.get_object = get_object
    return client


class FakeSessionClientCtx:
    """Async context manager wrapping a mock S3 client."""

    def __init__(self, client_factory):
        self._factory = client_factory

    async def __aenter__(self):
        return self._factory()

    async def __aexit__(self, *args):
        pass


RECORDS = [
    {"id": 1, "info": {"name": "Alice"}},
    {"id": 2, "info": {"name": "Bob"}},
    {"id": 3, "info": {"name": "Carol"}},
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestReadS3Async:
    @pytest.mark.asyncio
    @patch("jsonflat.integrations.aws.s3.aioboto3")
    async def test_basic_read(self, mock_aioboto3):
        keys = ["data/1.json", "data/2.json", "data/3.json"]
        responses = [{"Body": _make_s3_body(r)} for r in RECORDS]

        def make_client():
            return _make_s3_client(keys, responses)

        mock_aioboto3.Session.return_value.client.return_value = FakeSessionClientCtx(make_client)

        batches = []
        async for df in read_s3_async(bucket="b", prefix="data/", max_nesting=2, batch_size=10):
            batches.append(df)

        assert len(batches) == 1
        df = batches[0]
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "info__name" in df.columns
        assert "_source_file" in df.columns

    @pytest.mark.asyncio
    @patch("jsonflat.integrations.aws.s3.aioboto3")
    async def test_batching(self, mock_aioboto3):
        keys = [f"data/{i}.json" for i in range(5)]
        responses = [{"Body": _make_s3_body({"id": i})} for i in range(5)]

        def make_client():
            return _make_s3_client(keys, responses)

        mock_aioboto3.Session.return_value.client.return_value = FakeSessionClientCtx(make_client)

        batches = []
        async for df in read_s3_async(bucket="b", prefix="data/", batch_size=2):
            batches.append(df)

        # 5 keys / batch_size 2 = 3 batches (2, 2, 1)
        assert len(batches) == 3
        sizes = [len(b) for b in batches]
        assert sizes == [2, 2, 1]

    @pytest.mark.asyncio
    @patch("jsonflat.integrations.aws.s3.aioboto3")
    async def test_suffix_filter(self, mock_aioboto3):
        all_keys = ["a.json", "b.csv", "c.json"]
        json_keys = ["a.json", "c.json"]
        responses = [{"Body": _make_s3_body({"id": i})} for i in range(2)]

        def make_client():
            # Paginator lists all keys; suffix filtering happens in read_s3_async
            client = _make_s3_client(json_keys, responses)
            client.get_paginator.return_value = _make_paginator(all_keys)
            return client

        mock_aioboto3.Session.return_value.client.return_value = FakeSessionClientCtx(make_client)

        batches = []
        async for df in read_s3_async(bucket="b", prefix="", suffix=".json", batch_size=10):
            batches.append(df)

        assert len(batches) == 1
        assert len(batches[0]) == 2

    @pytest.mark.asyncio
    @patch("jsonflat.integrations.aws.s3.aioboto3")
    async def test_filter_fn(self, mock_aioboto3):
        keys = ["1.json", "2.json"]
        responses = [
            {"Body": _make_s3_body({"id": 1, "status": "complete"})},
            {"Body": _make_s3_body({"id": 2, "status": "pending"})},
        ]

        def make_client():
            return _make_s3_client(keys, responses)

        mock_aioboto3.Session.return_value.client.return_value = FakeSessionClientCtx(make_client)

        batches = []
        async for df in read_s3_async(
            bucket="b",
            prefix="",
            filter_fn=lambda d: d.get("status") == "complete",
            batch_size=10,
        ):
            batches.append(df)

        assert len(batches) == 1
        assert len(batches[0]) == 1
        assert batches[0].iloc[0]["id"] == 1

    @pytest.mark.asyncio
    @patch("jsonflat.integrations.aws.s3.aioboto3")
    async def test_empty_bucket(self, mock_aioboto3):
        def make_client():
            client = _make_s3_client([], [])
            client.get_paginator.return_value = _make_paginator([])
            return client

        mock_aioboto3.Session.return_value.client.return_value = FakeSessionClientCtx(make_client)

        batches = []
        async for df in read_s3_async(bucket="b", prefix="empty/"):
            batches.append(df)

        assert batches == []

    @pytest.mark.asyncio
    @patch("jsonflat.integrations.aws.s3.aioboto3")
    async def test_fetch_error_skipped(self, mock_aioboto3):
        keys = ["ok.json", "bad.json"]

        def make_client():
            client = MagicMock()
            client.get_paginator.return_value = _make_paginator(keys)

            call_idx = 0

            async def get_object(**kwargs):
                nonlocal call_idx
                idx = call_idx
                call_idx += 1
                if idx == 1:
                    raise Exception("network error")
                return {"Body": _make_s3_body({"id": 1})}

            client.get_object = get_object
            return client

        mock_aioboto3.Session.return_value.client.return_value = FakeSessionClientCtx(make_client)

        batches = []
        async for df in read_s3_async(bucket="b", prefix="", batch_size=10):
            batches.append(df)

        assert len(batches) == 1
        assert len(batches[0]) == 1
