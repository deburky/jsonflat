"""
Bedrock integration for jsonflat -- flatten conversation histories and model responses.

Usage:
    from jsonflat.aws.bedrock import flatten_response, flatten_conversations

    # Flatten a single Bedrock converse() response
    response = bedrock.converse(modelId="...", messages=[...])
    flat = flatten_response(response)

    # Flatten a list of conversation records into a DataFrame
    df = flatten_conversations(conversation_log)
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from jsonflat.core import flatten


def flatten_response(
    response: dict[str, Any],
    max_nesting: int | None = None,
) -> dict[str, Any]:
    """Flatten a Bedrock converse/invoke_model response into a flat dict.

    Args:
        response: raw response dict from bedrock.converse() or invoke_model().
        max_nesting: flatten depth (None = unlimited).

    Returns:
        Flat dict with __ separated keys.
    """
    return flatten(response, max_nesting)


def flatten_conversations(
    conversations: list[dict[str, Any]],
    max_nesting: int | None = None,
) -> pd.DataFrame:
    """Flatten a list of conversation records into a DataFrame.

    Each record is expected to be a dict with nested fields like messages,
    usage, metadata, etc. All records are flattened and returned as rows.

    Args:
        conversations: list of conversation record dicts.
        max_nesting: flatten depth (None = unlimited).

    Returns:
        pandas DataFrame with one row per conversation.
    """
    records = [flatten(conv, max_nesting) for conv in conversations]
    return pd.DataFrame(records)


def read_bedrock_history(
    conversations: list[dict[str, Any]],
    max_nesting: int | None = None,
    role: str | None = None,
) -> pd.DataFrame:
    """Extract individual messages from conversation histories into a DataFrame.

    Expands each conversation's messages list into separate rows,
    preserving conversation-level metadata.

    Args:
        conversations: list of conversation dicts, each containing a
            "messages" key with a list of message dicts.
        max_nesting: flatten depth for message content (None = unlimited).
        role: filter to only this role ("user", "assistant", None = all).

    Returns:
        pandas DataFrame with one row per message.
    """
    records: list[dict[str, Any]] = []

    for conv_idx, conv in enumerate(conversations):
        messages = conv.get("messages", [])
        # Flatten conversation-level metadata (everything except messages)
        conv_meta = {k: v for k, v in conv.items() if k != "messages"}
        flat_meta = flatten(conv_meta, max_nesting)

        for msg_idx, msg in enumerate(messages):
            if role and msg.get("role") != role:
                continue

            flat_msg = flatten(msg, max_nesting)
            row = {
                "_conversation_idx": conv_idx,
                "_message_idx": msg_idx,
                **flat_meta,
                **flat_msg,
            }
            records.append(row)

    return pd.DataFrame(records)
