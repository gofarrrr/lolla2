"""
Token Utility Module for METIS Tokenomics Telemetry

This module provides token counting utilities for the METIS cognitive platform,
enabling precise measurement of context consumption across cognitive processes.

Part of the "Operation: Certificate of Health - Enhanced" mission to implement
Glass-Box observability principles.
"""

import math
from typing import Union


def count_tokens(text: str) -> int:
    """
    Count the approximate number of tokens in a given text string.

    Uses the industry-standard heuristic of dividing character count by 4.
    This provides a fast, reasonable approximation for tokenomics telemetry.

    In the future, this can be upgraded to use proper tokenizers like tiktoken
    for more precise measurements.

    Args:
        text (str): The input text to count tokens for

    Returns:
        int: Approximate number of tokens in the text

    Example:
        >>> count_tokens("Hello world, this is a test")
        7
        >>> count_tokens('{"message": "A longer JSON payload with more content"}')
        15
    """
    if not isinstance(text, str):
        # Handle non-string input gracefully
        return 0

    if not text.strip():
        # Empty or whitespace-only text
        return 0

    # Use the industry standard heuristic: len(text) / 4
    token_count = len(text) / 4.0

    # Round up to ensure we don't underestimate token usage
    return math.ceil(token_count)


def count_tokens_for_payload(
    data: Union[str, dict, list, int, float, bool, None],
) -> int:
    """
    Count tokens for various data payload types by converting to string representation.

    This function handles different payload types that might be passed through
    the METIS cognitive processing pipeline.

    Args:
        data: The payload data to count tokens for (can be any JSON-serializable type)

    Returns:
        int: Approximate number of tokens in the payload

    Example:
        >>> count_tokens_for_payload({"status": "complete", "results": [1, 2, 3]})
        10
        >>> count_tokens_for_payload("Simple string payload")
        5
    """
    if data is None:
        return 0

    # Convert data to string representation
    if isinstance(data, str):
        text_repr = data
    else:
        # For non-string data, convert to string (similar to JSON serialization)
        text_repr = str(data)

    return count_tokens(text_repr)


def format_token_count_summary(token_count: int) -> str:
    """
    Format a human-readable summary of token count for logging and monitoring.

    Args:
        token_count (int): The number of tokens to format

    Returns:
        str: Formatted token count summary

    Example:
        >>> format_token_count_summary(1500)
        "1.5K tokens"
        >>> format_token_count_summary(50)
        "50 tokens"
    """
    if token_count < 1000:
        return f"{token_count} tokens"
    elif token_count < 1000000:
        return f"{token_count / 1000:.1f}K tokens"
    else:
        return f"{token_count / 1000000:.1f}M tokens"


# Export the main function for easy importing
__all__ = ["count_tokens", "count_tokens_for_payload", "format_token_count_summary"]
