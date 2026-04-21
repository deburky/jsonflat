"""Tests for the ``__version__`` fallback path in ``jsonflat/__init__.py``."""

from __future__ import annotations

import importlib
from unittest.mock import patch

from importlib.metadata import PackageNotFoundError

import jsonflat


def test_version_is_populated():
    """With the package installed, __version__ is the real metadata version."""
    assert jsonflat.__version__
    assert jsonflat.__version__ != "0.0.0+unknown"


def test_version_falls_back_when_metadata_missing():
    """When metadata lookup raises, __version__ falls back to the sentinel."""
    with patch("importlib.metadata.version", side_effect=PackageNotFoundError):
        reloaded = importlib.reload(jsonflat)
        try:
            assert reloaded.__version__ == "0.0.0+unknown"
        finally:
            importlib.reload(jsonflat)  # restore real version for other tests
