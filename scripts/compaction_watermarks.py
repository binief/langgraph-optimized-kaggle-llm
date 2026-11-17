"""Derive character-count compaction thresholds from the solver context size."""

from typing import Dict


def calculate_watermark_chars(
    context_tokens: int,
    chars_per_token: float = 3.75,
    soft_ratio: float = 0.60,
    high_ratio: float = 0.75,
    low_ratio: float = 0.40,
) -> Dict[str, int]:
    """Estimate compaction thresholds as fractions of the configured context.

    The gateway uses character counts rather than the backend tokenizer, so
    ``chars_per_token`` is an estimate and can be adjusted for a deployment.
    """
    if context_tokens <= 0:
        raise ValueError("context_tokens must be positive")
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    if not 0 < low_ratio < soft_ratio < high_ratio < 1:
        raise ValueError("watermark ratios must satisfy 0 < low < soft < high < 1")

    estimated_context_chars = context_tokens * chars_per_token
    return {
        "soft": round(estimated_context_chars * soft_ratio),
        "high": round(estimated_context_chars * high_ratio),
        "low": round(estimated_context_chars * low_ratio),
    }
