"""Tests for the ``jsonflat`` CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run(args: list[str], stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    """Invoke the jsonflat CLI as a subprocess and return the completed process."""
    return subprocess.run(
        [sys.executable, "-m", "jsonflat", *args],
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_from_file(tmp_path: Path) -> None:
    """Reading JSON from a file path prints a main-table summary with flattened keys."""
    data = {"user": {"name": "Alice", "address": {"city": "NYC"}}, "score": 90}
    path = tmp_path / "data.json"
    path.write_text(json.dumps(data))

    result = _run([str(path), "--nesting", "3"])
    assert result.returncode == 0, result.stderr
    assert "main: 1 rows" in result.stdout
    assert "user__name" in result.stdout
    assert "user__address__city" in result.stdout


def test_cli_from_stdin() -> None:
    """Reading JSON from stdin produces a main row plus one child table per list."""
    data = {"order_id": "A1", "items": [{"sku": "W1"}, {"sku": "G1"}]}
    result = _run(["--nesting", "3"], stdin=json.dumps(data))
    assert result.returncode == 0, result.stderr
    assert "main: 1 rows" in result.stdout
    assert "items: 2 rows" in result.stdout


def test_cli_list_input(tmp_path: Path) -> None:
    """A top-level JSON array produces one main-table row per record."""
    records = [{"id": 1, "city": "NYC"}, {"id": 2, "city": "SF"}]
    path = tmp_path / "list.json"
    path.write_text(json.dumps(records))

    result = _run([str(path)])
    assert result.returncode == 0, result.stderr
    assert "main: 2 rows" in result.stdout


def test_cli_missing_file() -> None:
    """A non-existent file path exits with a non-zero status."""
    result = _run(["/no/such/file.json"])
    assert result.returncode != 0
