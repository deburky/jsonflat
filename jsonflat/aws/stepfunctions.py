"""Shortcut: ``from jsonflat.aws.stepfunctions import read_executions``."""

from jsonflat.integrations.aws.stepfunctions import read_execution_history, read_executions

__all__ = ["read_executions", "read_execution_history"]
