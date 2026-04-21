"""CLI entry point for jsonflat."""

from __future__ import annotations

import argparse
import json
import sys

from jsonflat.core import normalize_json


def main() -> None:
    """Flatten nested JSON from file or stdin and print table summaries."""
    parser = argparse.ArgumentParser(description="Flatten nested JSON")
    parser.add_argument("file", nargs="?", help="JSON file (stdin if omitted)")
    parser.add_argument("--nesting", type=int, default=None, help="Max nesting depth (default: unlimited)")
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            data = json.load(f)
    else:
        data = json.load(sys.stdin)

    records = data if isinstance(data, list) else [data]
    tables = normalize_json(records, max_nesting=args.nesting)
    for name, rows in tables.items():
        cols = len(rows[0]) if rows else 0
        print(f"\n{name}: {len(rows)} rows, {cols} columns")
        if rows:
            for k in sorted(rows[0].keys()):
                v = rows[0][k]
                vstr = str(v)[:60]
                print(f"{k}: ({type(v).__name__}) {vstr}")


if __name__ == "__main__":
    main()
