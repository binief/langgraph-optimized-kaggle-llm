"""Helpers for normalizing OpenAI chat messages for Qwen's Jinja template."""

import json
from typing import Any, Dict, List


def keep_latest_user_in_tail(
    messages: List[Dict[str, Any]], split_index: int
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return older/tail segments while keeping the latest user turn in tail."""
    latest_user_index = max(
        (index for index, message in enumerate(messages) if message.get("role") == "user"),
        default=None,
    )
    if latest_user_index is not None and split_index > latest_user_index:
        split_index = latest_user_index
    return messages[:split_index], messages[split_index:]


def merge_system_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Move system messages to the front and combine them into one message.

    Qwen's chat template requires a single system message at the beginning of
    the conversation. API clients and gateway middleware may otherwise insert
    system/developer instructions after user, assistant, or tool messages.
    Non-system message order is preserved.
    """
    system_messages: List[Dict[str, Any]] = []
    conversation: List[Dict[str, Any]] = []

    for message in messages:
        if message.get("role") == "system":
            system_messages.append(message)
        else:
            conversation.append(message)

    if not system_messages:
        return conversation

    first_system = dict(system_messages[0])
    system_content = []
    for message in system_messages:
        content = message.get("content")
        if content is None:
            continue
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        if content:
            system_content.append(content)

    first_system["role"] = "system"
    first_system["content"] = "\n\n".join(system_content)
    return [first_system, *conversation]
