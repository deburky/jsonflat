"""Shortcut: ``from jsonflat.aws.bedrock import flatten_response``."""

from jsonflat.integrations.aws.bedrock import (
    flatten_conversations,
    flatten_response,
    read_bedrock_history,
)

__all__ = ["flatten_response", "flatten_conversations", "read_bedrock_history"]
