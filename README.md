# jsonflat

<p align="center">
  <img src="ims/jsonflat.png" alt="jsonflat" width="600">
</p>

Flatten nested JSON into DataFrames with controlled depth.

## Install

```bash
pip install .               # core (stdlib only, zero dependencies)
pip install ".[dataframe]"  # + pandas for to_dataframe()
pip install ".[s3]"         # + boto3/orjson for S3 integration
pip install ".[s3-async]"   # + aioboto3/orjson for async S3 streaming
pip install ".[sqs]"        # + boto3 for SQS integration
pip install ".[dynamodb]"   # + boto3 for DynamoDB integration
pip install ".[cloudwatch]" # + boto3 for CloudWatch Logs
pip install ".[eventbridge]" # + boto3 for EventBridge
pip install ".[stepfunctions]" # + boto3 for Step Functions
pip install ".[sklearn]"    # + scikit-learn for pipeline integration
```

## Usage

```python
from jsonflat import flatten, normalize_json, to_dataframe

data = {
    "id": "order-123",
    "customer": {
        "name": "Alice",
        "items": [
            {"product": "Widget", "qty": 2},
            {"product": "Gadget", "qty": 1},
        ],
    },
    "metadata": {"source": {"system": {"version": "2.1"}}},
}

# Flatten to a single dict
flatten(data, max_nesting=2)
# {'id': 'order-123', 'customer__name': 'Alice',
#  'customer__items': [{'product': 'Widget', ...}, ...],
#  'metadata__source__system': {'version': '2.1'}}

# Flatten with no depth limit
flatten(data, max_nesting=None)

# Normalize into parent + child tables
tables = normalize_json(data, max_nesting=3)
# tables["main"]            -> [{'id': 'order-123', 'customer__name': 'Alice', ...}]
# tables["customer.items"]  -> [{'product': 'Widget', 'qty': 2}, ...]

# Straight to DataFrame
df = to_dataframe(data, max_nesting=3)
df_items = to_dataframe(data, max_nesting=3, table="customer.items")
```

### Nesting control

```python
from jsonflat import flatten

data = {"a": {"b": {"c": {"d": 1}}}}

flatten(data, max_nesting=0)     # {'a': {'b': {'c': {'d': 1}}}}
flatten(data, max_nesting=1)     # {'a__b': {'c': {'d': 1}}}
flatten(data, max_nesting=None)  # {'a__b__c__d': 1}
```

### Multiple records

```python
from jsonflat import normalize_json, to_dataframe

records = [
    {"id": 1, "info": {"name": "Alice", "age": 30}},
    {"id": 2, "info": {"name": "Bob", "age": 25}},
]

tables = normalize_json(records, max_nesting=3)
# tables["main"] -> [{'id': 1, 'info__name': 'Alice', ...}, ...]

df = to_dataframe(records, max_nesting=3)
# DataFrame with columns: id, info__name, info__age
```

### Child tables

```python
from jsonflat import normalize_json, to_dataframe

data = {
    "order_id": "A1",
    "items": [
        {"sku": "W1", "qty": 2},
        {"sku": "G1", "qty": 1},
    ],
}

tables = normalize_json(data, max_nesting=3)
# tables["main"]  -> [{'order_id': 'A1'}]
# tables["items"] -> [{'sku': 'W1', 'qty': 2}, {'sku': 'G1', 'qty': 1}]

df_items = to_dataframe(data, max_nesting=3, table="items")
```

### CLI

```bash
# From file
jsonflat data.json --nesting 3

# From stdin
cat data.json | jsonflat --nesting 3
```

## Integrations

### S3

Read JSON files from S3 in parallel, flatten, return a DataFrame.

```python
from jsonflat.aws.s3 import read_s3

df = read_s3(
    bucket="my-bucket",
    prefix="events/2025/",
    max_files=1000,
    max_nesting=3,
)
```

CLI:
```bash
python -m jsonflat.integrations.aws.s3 \
    --bucket my-bucket --prefix events/ \
    --max-files 100 --nesting 3 --output result.parquet
```

### S3 (async)

Stream JSON files from S3 as DataFrame batches with bounded memory usage.

```python
from jsonflat.aws.s3 import read_s3_async

async for df_batch in read_s3_async(
    bucket="my-bucket",
    prefix="events/2025/",
    max_nesting=3,
    batch_size=100,
):
    process(df_batch)
```

### SQS

Consume JSON messages from SQS, flatten, and return a DataFrame.

```python
from jsonflat.aws.sqs import read_sqs, stream_sqs

# Poll and flatten up to 100 messages
df = read_sqs(
    queue_url="https://sqs.us-east-1.amazonaws.com/123/my-queue",
    max_messages=100,
    max_nesting=3,
)

# Stream as batches
for df_batch in stream_sqs(
    queue_url="https://sqs.us-east-1.amazonaws.com/123/my-queue",
    batch_size=10,
    max_nesting=3,
):
    process(df_batch)
```

### DynamoDB

Scan a DynamoDB table and consume DynamoDB Streams.

```python
from jsonflat.aws.dynamodb import read_dynamodb, read_stream, stream_records

# Scan table
df = read_dynamodb(table_name="my-table", max_nesting=3)

# Read stream (new images)
df = read_stream(table_name="my-table", max_nesting=3, image="new")

# Stream as batches, including old and new images
for df_batch in stream_records(table_name="my-table", image="both"):
    process(df_batch)
```

### Bedrock

Flatten Bedrock converse responses and conversation histories.

```python
from jsonflat.aws.bedrock import (
    flatten_response,
    flatten_conversations,
    read_bedrock_history,
)

# Flatten a single converse() response
response = bedrock.converse(modelId="...", messages=[...])
flat = flatten_response(response)
# {'usage__inputTokens': 17, 'usage__outputTokens': 5, 'metrics__latencyMs': 218, ...}

# Conversation-level summary (one row per session)
df = flatten_conversations(conversation_log)

# Expand messages into rows, with optional role filter
df = read_bedrock_history(conversation_log, role="assistant")
```

### CloudWatch Logs

Read log events, parse JSON, flatten into a DataFrame.

```python
from jsonflat.aws.cloudwatch import read_logs

df = read_logs(
    log_group="/aws/lambda/my-function",
    max_nesting=3,
    max_events=1000,
    filter_fn=lambda d: d.get("level") == "ERROR",
)
```

### EventBridge

Flatten EventBridge events consumed via an SQS target queue.

```python
from jsonflat.aws.eventbridge import flatten_event, read_events

# Flatten a single event
flat = flatten_event(event)
# {'source': 'myapp', 'detail-type': 'OrderCreated', 'detail__order_id': 'o1', ...}

# Read events from SQS target queue
df = read_events(
    queue_url="https://sqs.us-east-1.amazonaws.com/123/eb-target",
    max_nesting=3,
)
```

### Step Functions

Read execution summaries and per-step event history.

```python
from jsonflat.aws.stepfunctions import read_executions, read_execution_history

# All executions with flattened input/output
df = read_executions(
    state_machine_arn="arn:aws:states:...:stateMachine:my-machine",
    max_nesting=3,
)
# Columns: execution_arn, status, input__order_id, output__result__status, ...

# Event history for a single execution
df = read_execution_history(execution_arn="arn:aws:states:...:execution:...")
```

### Scikit-learn

Use `JsonFlattener` as a preprocessing step in sklearn pipelines.

```python
from jsonflat.sklearn import JsonFlattener
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

pipe = Pipeline([
    ("flatten", JsonFlattener(max_nesting=3)),
    ("model", DecisionTreeClassifier()),
])

records = [
    {"info": {"age": 30, "score": 90}},
    {"info": {"age": 25, "score": 60}},
]
pipe.fit(records, [1, 0])
```

Accepts list of dicts or a DataFrame with a JSON column (`column="payload"`).

### Lambda

AWS Lambda handler that flattens JSON from S3 and writes parquet back.
Supports S3 event triggers and direct invocation.

```python
# Entry point: jsonflat.aws.lambda_handler.handler

# Direct invoke:
import boto3
lambda_client = boto3.client("lambda")
lambda_client.invoke(
    FunctionName="jsonflat",
    Payload='{"bucket": "my-bucket", "prefix": "events/", "max_nesting": 3}',
)
```

### ibis / DuckDB

jsonflat pairs well with ibis for SQL queries on flattened JSON.

```python
import ibis
import pandas as pd
from jsonflat.core import flatten

flat = [flatten(r) for r in records]
con = ibis.duckdb.connect()
t = con.create_table("orders", pd.DataFrame(flat))

# ibis API
t.filter(t.user__address__city == "NYC").to_pandas()

# Raw SQL
con.sql("SELECT * FROM orders WHERE order__total > 100")
```

## How it works

Nested dicts are flattened with `__` separators up to `max_nesting` depth. Beyond that depth, values are stored as-is (JSON blobs).

| `max_nesting` | Behavior |
|---|---|
| `None` | Flatten everything, no limit |
| `3` | Flatten 3 levels deep, store deeper as JSON blobs |
| `0` | No flattening, top-level keys only |

Lists of dicts become child tables in `normalize_json()` / `to_dataframe()`. Lists of scalars and empty lists are kept as values.

## Structure

```
jsonflat/
├── pyproject.toml
├── README.md
├── .pre-commit-config.yaml
├── jsonflat/
│   ├── __init__.py              # re-exports flatten, normalize_json, to_dataframe
│   ├── core.py                  # core functions
│   ├── __main__.py              # CLI entry point
│   ├── aws/                     # short import aliases (from jsonflat.aws.s3 import ...)
│   ├── sklearn.py               # short import alias (from jsonflat.sklearn import ...)
│   └── integrations/
│       ├── sklearn.py           # scikit-learn transformer
│       └── aws/
│           ├── s3.py            # S3 sync + async reader
│           ├── sqs.py           # SQS consumer
│           ├── dynamodb.py      # DynamoDB scan + streams
│           ├── bedrock.py       # Bedrock conversation flattening
│           ├── cloudwatch.py    # CloudWatch Logs reader
│           ├── eventbridge.py   # EventBridge event flattening
│           ├── stepfunctions.py # Step Functions execution reader
│           └── lambda_handler.py # AWS Lambda handler
└── tests/
    ├── test_jsonflat.py         # 24 core tests
    ├── test_s3.py               # 6 sync S3 tests
    ├── test_s3_async.py         # 6 async S3 tests
    ├── test_sqs.py              # 8 SQS tests
    ├── test_dynamodb.py         # 5 DynamoDB scan tests
    ├── test_dynamodb_streams.py # 8 DynamoDB streams tests
    ├── test_bedrock.py          # 10 Bedrock tests
    ├── test_cloudwatch.py       # 6 CloudWatch tests
    ├── test_eventbridge.py      # 7 EventBridge tests
    └── test_stepfunctions.py    # 6 Step Functions tests
```

## Tests

```bash
pip install -e ".[dataframe,s3,s3-async,sqs,dynamodb,sklearn]"
pytest tests/ -v
```
