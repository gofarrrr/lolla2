#!/usr/bin/env python3
"""
T-13 Unicode Hardening: String Utilities
Safe string manipulation utilities to prevent Unicode corruption.

This module provides Unicode-aware string operations that prevent the creation
of invalid Unicode sequences (like dangling surrogates) that can corrupt JSON
and cause API request failures.
"""

import logging
from typing import Union


logger = logging.getLogger(__name__)


def safe_truncate(text: str, max_bytes: int) -> str:
    """
    Truncates a string to a maximum number of bytes without creating invalid Unicode.
    
    This function prevents the "dangling surrogate" Unicode error that occurs when
    naive string slicing cuts a multi-byte Unicode character (like emojis) in half.
    
    The fix: Encode to UTF-8, truncate the byte string, then decode back to a string,
    ignoring any errors from a partial character at the end.
    
    Args:
        text: The string to truncate
        max_bytes: Maximum number of UTF-8 bytes allowed
        
    Returns:
        str: The safely truncated string without invalid Unicode sequences
        
    Example:
        >>> # This might create invalid Unicode:
        >>> bad_truncate = "Hello 😀 World"[:10]  # Could cut emoji in half
        
        >>> # This is safe:
        >>> safe_result = safe_truncate("Hello 😀 World", max_bytes=50)
        
        >>> # Even when truncating at a dangerous boundary:
        >>> emoji_text = "Text with emoji 😀 more text"
        >>> safe_result = safe_truncate(emoji_text, max_bytes=20)  # Won't break emoji
    """
    if not isinstance(text, str):
        logger.warning(f"safe_truncate received non-string input: {type(text)}")
        return str(text)
    
    if max_bytes <= 0:
        return ""
    
    try:
        # Encode to UTF-8, truncate the byte string, then decode back to a string,
        # ignoring any errors from a partial character at the end.
        truncated_bytes = text.encode('utf-8')[:max_bytes]
        return truncated_bytes.decode('utf-8', errors='ignore')
        
    except Exception as e:
        logger.error(f"T-13 UNICODE HARDENING: safe_truncate failed for text length {len(text)}: {e}")
        # Fallback: return empty string rather than risk corruption
        return ""


def safe_substring(text: str, start: int, end: int = None) -> str:
    """
    Extract a substring safely without creating invalid Unicode sequences.
    
    This is a Unicode-aware alternative to text[start:end] that prevents 
    cutting multi-byte characters in half.
    
    Args:
        text: The source string
        start: Starting character position (not byte position)
        end: Ending character position (optional)
        
    Returns:
        str: The safely extracted substring
    """
    if not isinstance(text, str):
        logger.warning(f"safe_substring received non-string input: {type(text)}")
        return str(text)
    
    if start < 0:
        start = 0
        
    if end is None:
        return text[start:]
    
    if end <= start:
        return ""
    
    try:
        # Use character-based slicing (which Python handles correctly)
        # Then ensure the result doesn't have trailing incomplete sequences
        substring = text[start:end]
        
        # Double-check by encoding/decoding to catch any edge cases
        test_bytes = substring.encode('utf-8')
        return test_bytes.decode('utf-8', errors='ignore')
        
    except Exception as e:
        logger.error(f"T-13 UNICODE HARDENING: safe_substring failed: {e}")
        return ""


def is_safe_unicode(text: str) -> bool:
    """
    Check if a string contains valid Unicode without dangling surrogates.
    
    Args:
        text: The string to validate
        
    Returns:
        bool: True if the string is safe, False if it contains invalid Unicode
    """
    if not isinstance(text, str):
        return False
        
    try:
        # Try to encode and decode - this will fail if there are invalid sequences
        text.encode('utf-8').decode('utf-8')
        return True
        
    except UnicodeError:
        return False


def sanitize_for_json(text: str, max_bytes: int = None) -> str:
    """
    Sanitize a string to be safe for JSON serialization.
    
    This function combines safe truncation with other JSON safety measures.
    
    Args:
        text: The string to sanitize
        max_bytes: Optional maximum byte length
        
    Returns:
        str: A JSON-safe string
    """
    if not isinstance(text, str):
        text = str(text)
    
    # First, ensure valid Unicode
    try:
        # Remove any existing invalid sequences
        clean_text = text.encode('utf-8', errors='ignore').decode('utf-8')
    except Exception:
        logger.warning("T-13 UNICODE HARDENING: Failed to clean Unicode, using empty string")
        clean_text = ""
    
    # Apply safe truncation if requested
    if max_bytes is not None and max_bytes > 0:
        clean_text = safe_truncate(clean_text, max_bytes)
    
    # Additional JSON safety: escape or remove problematic characters
    # (Add more JSON-specific sanitization here if needed)
    
    return clean_text


# Backward compatibility aliases
safe_slice = safe_substring
unicode_safe_truncate = safe_truncate