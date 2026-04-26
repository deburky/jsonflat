# jsonflat

<p align="center">
  <img src="ims/jsonflat.png" alt="jsonflat" width="900">
</p>

**jsonflat** is a normalizer that handles the parent-child splitting automatically.

## Install

```bash
uv add git+https://github.com/deburky/jsonflat.git        # core (stdlib only)
uv add "git+https://github.com/deburky/jsonflat.git[dataframe]"  # + pandas
uv add "git+https://github.com/deburky/jsonflat.git[s3-async]"   # + aioboto3/orjson for @aio with S3
uv add "git+https://github.com/deburky/jsonflat.git[sklearn]"    # + scikit-learn for pipelines
```

## Main functions

### Flatten

`flatten(record)`: one dict in, one flat dict out. Use when you have a single record and want to inspect it or build your own DataFrame manually.

```python
from jsonflat import flatten

# Top level: bank statement transaction
pd.DataFrame([flatten(json.loads(df_bs["transactions"].iloc[0]))])

# No depth limit: all nesting resolved
record_level_df = pd.DataFrame([flatten(client, max_nesting=None)])
```

Depth is controlled by `max_nesting`. Beyond that depth, dicts are stored as-is (JSON blobs).

| `max_nesting` | Behavior |
|---|---|
| `None` | Flatten everything, no limit |
| `3` | Flatten 3 levels deep, store deeper as blobs |
| `0` | No flattening, top-level keys only |

```python
data = {"a": {"b": {"c": {"d": 1}}}}

flatten(data, max_nesting=0)     # {'a': {'b': {'c': {'d': 1}}}}
flatten(data, max_nesting=1)     # {'a__b': {'c': {'d': 1}}}
flatten(data, max_nesting=None)  # {'a__b__c__d': 1}
```

### Unflatten

`unflatten(row)` reconstructs a nested dict from a flat dict: the dual of `flatten`. Use when round-tripping through columnar storage (parquet, CSV, DataFrame columns with `__` separators) back to nested JSON.

```python
from jsonflat import flatten, unflatten

data = {"user": {"name": "Alice", "address": {"city": "NYC"}}, "amount": 42.5}

flat = flatten(data, max_nesting=None)
# {'user__name': 'Alice', 'user__address__city': 'NYC', 'amount': 42.5}

unflatten(flat) == data  # True
```

Pass `separator="."` to unflatten keys that use a different delimiter. Conflicting paths (a key used both as a leaf and as a parent, e.g. `{"a": 1, "a__b": 2}`) raise `ValueError`.

### Normalize JSON

`normalize_json(data)` is the main workhorse. It takes a record or list of records and returns a dict of tables: `main` for scalar fields, plus one child table per list-of-dicts it finds. Use when your JSON has arrays (transactions, documents, accounts) that need to become separate DataFrames.

```python
from jsonflat import normalize_json

data = {
    "order_id": "A1",
    "items": [
        {"sku": "W1", "qty": 2},
        {"sku": "G1", "qty": 1},
    ],
}

tables = normalize_json(data)
# tables["main"]  -> [{'order_id': 'A1'}]
# tables["items"] -> [{'sku': 'W1', 'qty': 2}, {'sku': 'G1', 'qty': 1}]
```

> [!NOTE]
> The `key` parameter copies a field from each parent row into its child rows as a foreign key. If the key doesn't exist in the parent, a `KeyError` is raised. If it already exists in a child row with the same value, a `UserWarning` is issued. If it exists with a different value, a `ValueError` is raised. Resolve by renaming the conflicting field or using a different key.

```python
tables = normalize_json([client1, client2], key="client_id")
# client_id is copied into every child row as a join key
```

#### `hoist`: ID-keyed dicts

When a dict's keys are IDs, `hoist` promotes those IDs to a column instead of baking them into table names.

```python
# Without hoist, loan IDs bleed into table names:
# tables["loans.<loan_id>.details.items"]

# With hoist, clean loan_id column:
tables = normalize_json(records, hoist=[("loans", "loan_id")])
# tables["loans.details.items"]
# each row has a loan_id column

# String shorthand, uses f"{prefix}_id" as the column name:
tables = normalize_json(records, hoist=["loans"])
# each row gets a loans_id column
```

### To DataFrame

`to_dataframe(data)` is a convenience wrapper around `normalize_json` that returns just the `main` table as a DataFrame. Use when there are no meaningful arrays, or you only care about the parent-level fields and want to skip the table dict entirely.

> [!NOTE]
> `to_dataframe` is just a convenience shortcut: it calls `normalize_json` internally and returns `tables["main"]` as a DataFrame directly.
> ```python
> # Two steps
> tables = normalize_json(data)
> pd.DataFrame(tables["main"])
>
> # One step (table="main" is the default)
> to_dataframe(data)
> ```

```python
from jsonflat import to_dataframe

df = to_dataframe(data, max_nesting=3)
df_items = to_dataframe(data, max_nesting=3, table="items")
```

### `@aio`: concurrent async I/O

`aio` is a decorator that runs an async function over a list of items concurrently. Pass `service="s3"` (and optionally `profile`, `region`) to get an `aioboto3` connection pool injected automatically: one shared pool for all calls, no per-call TLS overhead.

```python
import orjson
from jsonflat import aio, normalize_json

@aio(workers=32, service="s3", profile="my-profile")
async def fetch(key, s3):
    resp = await s3.get_object(Bucket="my-bucket", Key=key)
    return orjson.loads(await resp["Body"].read())

records = fetch(keys)                    # blocks; returns list in order
tables = normalize_json(records)
```

Without a service, use your own async context manager via `pool=`:

```python
import aioboto3
from aiobotocore.config import AioConfig

session = aioboto3.Session()
pool = lambda: session.client("s3", config=AioConfig(max_pool_connections=32))

@aio(workers=32, pool=pool)
async def fetch(key, s3):
    ...
```

Or use without a pool for non-I/O async work, with a semaphore limiting concurrency:

```python
@aio(workers=8)
async def process(item):
    ...

results = process(items)
```

### CLI

```bash
# From file
jsonflat data.json --nesting 3

# From stdin
cat data.json | jsonflat --nesting 3
```

## Integrations

### DynamoDB

`decode` converts a DynamoDB item (typed value dicts from `GetItem`, `Scan`, or stream images) to a plain Python dict, ready for `flatten()` or `normalize_json()`.

```python
from jsonflat.aws.dynamodb import decode
from jsonflat import flatten, normalize_json

# Decode a single item
item = {"id": {"S": "A1"}, "qty": {"N": "3"}, "active": {"BOOL": True}}
flatten(decode(item))
# {'id': 'A1', 'qty': 3, 'active': True}

# Decode a batch (e.g. from a Lambda event or Scan response)
records = [decode(item) for item in response["Items"]]
tables = normalize_json(records)
```

For table scans and stream consumption, use Lambda event source mappings or AWS Glue — they handle pagination, checkpointing, and fan-out better than a library wrapper.

### Scikit-learn

`JsonFlattener` is a stateful sklearn transformer. `fit` learns the column schema from training data; `transform` reindexes to that schema, filling missing columns with `NaN` and dropping unseen ones — guaranteeing consistent shape across fit/transform splits.

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

#### Joining child tables

Use `join`, `on`, and `how` to merge child tables onto the spine in a single transformer step. Child columns are prefixed as `<child_name>__<col>` to avoid collisions.

```python
records = [
    {"client_id": "C1", "name": "Alice", "transactions": [{"amount": 10.0}, {"amount": 20.0}]},
    {"client_id": "C2", "name": "Bob",   "transactions": [{"amount": 5.0}]},
]

JsonFlattener(join=["transactions"], on="client_id").fit_transform(records)
#    client_id   name  transactions__amount
# 0        C1  Alice                  10.0
# 1        C1  Alice                  20.0
# 2        C2    Bob                   5.0
```

### ibis / DuckDB

jsonflat pairs well with ibis for SQL queries on flattened JSON.

```python
import ibis
from jsonflat import flatten

flat = [flatten(r) for r in records]
con = ibis.duckdb.connect()
t = con.create_table("orders", pd.DataFrame(flat))

t.filter(t.user__address__city == "NYC").to_pandas()
con.sql("SELECT * FROM orders WHERE order__total > 100")
```

## Structure

```
jsonflat/
├── pyproject.toml
├── README.md
├── jsonflat/
│   ├── __init__.py      # re-exports flatten, normalize_json, to_dataframe, aio
│   ├── core.py          # flatten, unflatten, normalize_json, to_dataframe, aio
│   ├── __main__.py      # CLI entry point
│   ├── sklearn.py       # JsonFlattener — stateful sklearn transformer
│   └── aws/
│       └── dynamodb.py  # decode — DynamoDB JSON → plain Python
└── tests/
```

## Tests

```bash
uv run pytest tests/ -v
```
