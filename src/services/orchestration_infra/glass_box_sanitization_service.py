"""
Glass-Box Sanitization Service
===============================

OPERATION SCALPEL V2 - Phase 2.3: MOVE (Self-Contained Service)

This service provides PII scrubbing and content sanitization for the Glass-Box
UnifiedContextStream, ensuring complete operational transparency without exposing
sensitive content.

Service Responsibilities:
- Generate content fingerprints (SHA256 hashing)
- Extract safe statistical variables from content
- Sanitize pipeline content for glass-box compliance

Pattern: Self-contained service with zero orchestrator dependency
Status: Phase 2.3 MOVE - Logic migrated, service fully independent
"""

import hashlib
import re
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class GlassBoxSanitizationService:
    """
    OPERATION SCALPEL V2 - Phase 2.3: Glass-Box Sanitization Service

    Provides PII scrubbing and content sanitization for transparent logging
    without exposing sensitive information.

    This service implements the "Glass Box" principle: complete operational
    transparency with zero PII leakage.

    Service is fully self-contained with no external dependencies.
    """

    def __init__(self):
        """
        Initialize Glass-Box Sanitization Service

        Phase 2.3: Self-contained service, no orchestrator dependency
        """
        logger.info("🔗 GlassBoxSanitizationService initialized (Phase 2.3: MOVE - Self-contained)")

    def generate_content_fingerprint(self, content: str) -> str:
        """
        Generate safe fingerprint for content without exposing content.

        Uses SHA256 hashing to create a 12-character fingerprint that uniquely
        identifies content without revealing its actual text.

        Args:
            content: Raw content to fingerprint

        Returns:
            SHA256 hash fingerprint (12 chars), or "empty_content" if content is empty
        """
        if not content:
            return "empty_content"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]

    def extract_safe_variables(self, content: str) -> Dict[str, Any]:
        """
        Extract safe statistical variables from content without exposing text.

        Analyzes content to extract statistical metadata and structural patterns
        without revealing the actual content. This enables glass-box transparency
        while maintaining PII protection.

        Args:
            content: Raw content to analyze

        Returns:
            Dictionary of safe statistical variables:
            - char_count: Total character count
            - word_count: Total word count
            - line_count: Total line count
            - has_questions: Boolean indicating presence of question marks
            - has_numbers: Boolean indicating presence of digits
            - has_bullet_points: Boolean indicating presence of bullet point markers
            - has_headers: Boolean indicating presence of markdown headers
            - structure_type: Classified content structure (json/code/report/etc.)
        """
        if not content:
            return {
                "char_count": 0,
                "word_count": 0,
                "line_count": 0,
                "has_questions": False,
                "has_numbers": False,
                "structure_type": "empty",
            }

        # Safe statistical analysis
        char_count = len(content)
        word_count = len(content.split())
        line_count = len(content.split("\n"))

        # Pattern detection (no content exposure)
        has_questions = "?" in content
        has_numbers = bool(re.search(r"\d", content))
        has_json = content.strip().startswith("{") and content.strip().endswith("}")
        has_code = bool(re.search(r"```|def |function |class |import ", content))
        has_bullet_points = bool(re.search(r"^\s*[-*•]\s", content, re.MULTILINE))
        has_headers = bool(re.search(r"^#+\s", content, re.MULTILINE))

        # Determine structure type for pipeline content
        if has_json:
            structure_type = "json"
        elif has_code:
            structure_type = "code"
        elif has_headers and has_bullet_points:
            structure_type = "report"
        elif has_bullet_points:
            structure_type = "structured_list"
        elif has_questions:
            structure_type = "interrogative"
        elif word_count < 10:
            structure_type = "brief"
        elif word_count > 500:
            structure_type = "comprehensive"
        else:
            structure_type = "narrative"

        return {
            "char_count": char_count,
            "word_count": word_count,
            "line_count": line_count,
            "has_questions": has_questions,
            "has_numbers": has_numbers,
            "has_bullet_points": has_bullet_points,
            "has_headers": has_headers,
            "structure_type": structure_type,
        }

    def sanitize_pipeline_content(self, content: str, content_type: str) -> Dict[str, Any]:
        """
        Convert raw pipeline content to glass-box safe representation.

        Combines content fingerprinting and safe variable extraction to create
        a fully sanitized representation suitable for glass-box logging.

        Args:
            content: Raw pipeline content
            content_type: Type label for the content (e.g., "query", "analysis", "final_report")

        Returns:
            Sanitized content dictionary with:
            - {content_type}_fingerprint: SHA256 hash of content
            - {content_type}_variables: Safe statistical variables
            - {content_type}_redacted: Always True
            - glass_box_compliant: Always True
        """
        return {
            f"{content_type}_fingerprint": self.generate_content_fingerprint(content),
            f"{content_type}_variables": self.extract_safe_variables(content),
            f"{content_type}_redacted": True,
            "glass_box_compliant": True,
        }
