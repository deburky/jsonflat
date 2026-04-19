from __future__ import annotations

import pandas as pd

from jsonflat.integrations.bedrock import (
    flatten_conversations,
    flatten_response,
    read_bedrock_history,
)

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------
CONVERSE_RESPONSE = {
    "output": {
        "message": {
            "role": "assistant",
            "content": [{"text": "Hello! How can I help you?"}],
        }
    },
    "usage": {"inputTokens": 10, "outputTokens": 8},
    "metrics": {"latencyMs": 250},
    "stopReason": "end_turn",
}

CONVERSATIONS = [
    {
        "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
        "session_id": "s-001",
        "messages": [
            {"role": "user", "content": [{"text": "What is 2+2?"}]},
            {"role": "assistant", "content": [{"text": "4"}]},
        ],
        "usage": {"inputTokens": 5, "outputTokens": 1},
    },
    {
        "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
        "session_id": "s-002",
        "messages": [
            {"role": "user", "content": [{"text": "Hi"}]},
            {"role": "assistant", "content": [{"text": "Hello!"}]},
            {"role": "user", "content": [{"text": "Bye"}]},
            {"role": "assistant", "content": [{"text": "Goodbye!"}]},
        ],
        "usage": {"inputTokens": 10, "outputTokens": 5},
    },
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestFlattenResponse:
    def test_basic(self):
        flat = flatten_response(CONVERSE_RESPONSE)
        assert flat["usage__inputTokens"] == 10
        assert flat["usage__outputTokens"] == 8
        assert flat["metrics__latencyMs"] == 250
        assert flat["stopReason"] == "end_turn"
        assert flat["output__message__role"] == "assistant"

    def test_max_nesting(self):
        flat = flatten_response(CONVERSE_RESPONSE, max_nesting=1)
        assert "output__message" in flat
        assert isinstance(flat["output__message"], dict)


class TestFlattenConversations:
    def test_basic(self):
        df = flatten_conversations(CONVERSATIONS)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "model_id" in df.columns
        assert "session_id" in df.columns
        assert "usage__inputTokens" in df.columns

    def test_empty(self):
        df = flatten_conversations([])
        assert len(df) == 0


class TestReadBedrockHistory:
    def test_all_messages(self):
        df = read_bedrock_history(CONVERSATIONS)
        # s-001: 2 messages, s-002: 4 messages = 6 total
        assert len(df) == 6
        assert "_conversation_idx" in df.columns
        assert "_message_idx" in df.columns
        assert "role" in df.columns
        assert "session_id" in df.columns

    def test_filter_by_role(self):
        df = read_bedrock_history(CONVERSATIONS, role="assistant")
        # s-001: 1 assistant, s-002: 2 assistant = 3
        assert len(df) == 3
        assert all(df["role"] == "assistant")

    def test_user_messages_only(self):
        df = read_bedrock_history(CONVERSATIONS, role="user")
        assert len(df) == 3
        assert all(df["role"] == "user")

    def test_preserves_conversation_metadata(self):
        df = read_bedrock_history(CONVERSATIONS)
        s001 = df[df["session_id"] == "s-001"]
        assert len(s001) == 2
        assert all(s001["model_id"] == "anthropic.claude-3-haiku-20240307-v1:0")

    def test_empty_conversations(self):
        df = read_bedrock_history([])
        assert len(df) == 0

    def test_no_messages_key(self):
        df = read_bedrock_history([{"model_id": "test"}])
        assert len(df) == 0
