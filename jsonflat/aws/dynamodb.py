"""Shortcut: ``from jsonflat.aws.dynamodb import read_dynamodb``."""

from jsonflat.integrations.aws.dynamodb import read_dynamodb, read_stream, stream_records

__all__ = ["read_dynamodb", "read_stream", "stream_records"]
