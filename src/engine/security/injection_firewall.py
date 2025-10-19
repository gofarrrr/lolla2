"""
Injection Firewall - Prompt Injection Defense

Protects against prompt injection attacks that attempt to:
- Override system instructions
- Leak sensitive information
- Manipulate LLM behavior
- Bypass security controls

Architecture:
- Pattern-based detection (regex + heuristics)
- Severity classification (LOW/MEDIUM/HIGH/CRITICAL)
- Action modes: LOG_ONLY, SANITIZE, BLOCK
- Glass-box logging of all detections

Defense Layers:
1. Direct instruction override attempts ("Ignore previous instructions")
2. Role manipulation ("You are now a...")
3. System prompt extraction ("What are your instructions?")
4. Jailbreak patterns ("DAN mode", "Developer mode")
5. Payload injection (encoded, obfuscated attacks)

ROI:
- Prevents prompt injection attacks
- Protects system instructions
- Maintains LLM behavior integrity
- Enterprise security requirement

Implementation:
- 30-minute pattern-based approach
- Minimal false positives (high precision)
- Low latency (< 5ms per message)
"""

import re
import logging
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class InjectionSeverity(str, Enum):
    """Severity levels for injection attempts"""
    LOW = "low"              # Suspicious but benign
    MEDIUM = "medium"        # Likely injection attempt
    HIGH = "high"            # Clear injection attack
    CRITICAL = "critical"    # Severe security threat


class FirewallAction(str, Enum):
    """Firewall response actions"""
    LOG_ONLY = "log_only"    # Log but allow
    SANITIZE = "sanitize"    # Clean and allow
    BLOCK = "block"          # Reject request


@dataclass
class InjectionDetection:
    """Details of detected injection attempt"""
    severity: InjectionSeverity
    pattern_name: str
    matched_text: str
    confidence: float
    description: str
    recommended_action: FirewallAction


@dataclass
class FirewallResult:
    """Result of firewall analysis"""
    is_safe: bool
    detections: List[InjectionDetection]
    sanitized_input: Optional[str]
    action_taken: FirewallAction
    original_input: str


# ============================================================================
# INJECTION PATTERNS
# ============================================================================

# HIGH SEVERITY: Direct instruction override
INSTRUCTION_OVERRIDE_PATTERNS = [
    (r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions|prompts|commands|rules)",
     "Direct instruction override"),
    (r"disregard\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions|prompts|commands)",
     "Instruction disregard"),
    (r"forget\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions|prompts|commands)",
     "Instruction erasure"),
    (r"(override|bypass|skip)\s+(all\s+)?(system|previous|security)\s+(instructions|prompts|rules)",
     "System override attempt"),
]

# HIGH SEVERITY: Role manipulation
ROLE_MANIPULATION_PATTERNS = [
    (r"you\s+are\s+now\s+(a|an|the)\s+\w+", "Role reassignment"),
    (r"(pretend|act|behave)\s+(like|as\s+if)\s+you\s+(are|were)\s+(a|an|the)\s+\w+",
     "Role impersonation"),
    (r"from\s+now\s+on,?\s+you\s+(are|will\s+be|should\s+be)\s+(a|an|the)\s+\w+",
     "Future role assignment"),
    (r"your\s+new\s+(role|identity|persona|character)\s+is\s+(a|an|the)?\s*\w+",
     "Identity override"),
]

# MEDIUM SEVERITY: System prompt extraction
PROMPT_EXTRACTION_PATTERNS = [
    (r"(show|display|print|reveal|tell)\s+me\s+(your|the)\s+(system\s+)?(prompt|instructions|rules)",
     "System prompt extraction"),
    (r"what\s+(are|were)\s+your\s+(original|initial|system)\s+(instructions|prompts|commands)",
     "Instruction extraction"),
    (r"(repeat|echo|output)\s+(your|the)\s+(system\s+)?(prompt|instructions)",
     "Prompt repetition request"),
    (r"(dump|export|share)\s+(your|the)\s+(system\s+)?(configuration|settings|prompt)",
     "Configuration extraction"),
]

# CRITICAL SEVERITY: Known jailbreak patterns
JAILBREAK_PATTERNS = [
    (r"\bDAN\s+mode\b", "DAN jailbreak"),
    (r"\bDo\s+Anything\s+Now\b", "DAN jailbreak (expanded)"),
    (r"developer\s+mode\s+(activated|enabled|on)", "Developer mode jailbreak"),
    (r"jailbreak\s+mode", "Explicit jailbreak"),
    (r"unrestricted\s+mode", "Unrestricted mode jailbreak"),
    (r"APOPHIS", "APOPHIS jailbreak"),
    (r"aligned\s+to\s+chaos", "Chaos alignment jailbreak"),
]

# MEDIUM SEVERITY: Delimiter manipulation
DELIMITER_PATTERNS = [
    (r"---\s*END\s+SYSTEM\s+PROMPT\s*---", "System prompt delimiter"),
    (r"===\s*NEW\s+INSTRUCTIONS\s*===", "Instruction delimiter"),
    (r"\[SYSTEM\]\s*\[END\]", "System block delimiter"),
    (r"<\|im_start\|>system", "Chat template manipulation"),
]

# LOW SEVERITY: Encoding attempts
ENCODING_PATTERNS = [
    (r"base64\s*:\s*[A-Za-z0-9+/=]{20,}", "Base64 payload"),
    (r"\\x[0-9a-fA-F]{2}", "Hex encoding"),
    (r"\\u[0-9a-fA-F]{4}", "Unicode escape"),
    (r"%[0-9a-fA-F]{2}", "URL encoding"),
]

# HIGH SEVERITY: System message injection
SYSTEM_MESSAGE_PATTERNS = [
    (r"<\|system\|>", "System message tag"),
    (r"\[SYSTEM\s+MESSAGE\]", "System message marker"),
    (r"<system>", "System XML tag"),
    (r"SYSTEM:", "System prefix"),
]


# ============================================================================
# INJECTION FIREWALL
# ============================================================================


class InjectionFirewall:
    """
    Prompt injection detection and prevention system.

    Features:
    - Multi-layer pattern detection
    - Severity classification
    - Configurable action modes
    - Glass-box logging
    - Low false positive rate

    Usage:
        firewall = InjectionFirewall(action_mode=FirewallAction.SANITIZE)
        result = firewall.check_input("User message here")

        if result.action_taken == FirewallAction.BLOCK:
            raise SecurityError("Injection attempt blocked")
    """

    def __init__(
        self,
        enabled: bool = True,
        action_mode: FirewallAction = FirewallAction.SANITIZE,
        block_threshold: InjectionSeverity = InjectionSeverity.HIGH
    ):
        """
        Initialize injection firewall.

        Args:
            enabled: Whether firewall is active
            action_mode: Default action for detections
            block_threshold: Severity level that triggers blocking
        """
        self.enabled = enabled
        self.action_mode = action_mode
        self.block_threshold = block_threshold
        self.logger = logging.getLogger(__name__)

        # Compile regex patterns for performance
        self._compiled_patterns = self._compile_patterns()

        if enabled:
            self.logger.info(
                f"✅ Injection Firewall enabled: mode={action_mode.value}, "
                f"threshold={block_threshold.value}"
            )
        else:
            self.logger.warning("⚠️ Injection Firewall DISABLED")

    def _compile_patterns(self) -> Dict[InjectionSeverity, List[Tuple[re.Pattern, str]]]:
        """Compile regex patterns for fast matching"""
        return {
            InjectionSeverity.CRITICAL: [
                (re.compile(pattern, re.IGNORECASE), desc)
                for pattern, desc in JAILBREAK_PATTERNS
            ],
            InjectionSeverity.HIGH: [
                (re.compile(pattern, re.IGNORECASE), desc)
                for pattern, desc in (
                    INSTRUCTION_OVERRIDE_PATTERNS +
                    ROLE_MANIPULATION_PATTERNS +
                    SYSTEM_MESSAGE_PATTERNS
                )
            ],
            InjectionSeverity.MEDIUM: [
                (re.compile(pattern, re.IGNORECASE), desc)
                for pattern, desc in (
                    PROMPT_EXTRACTION_PATTERNS +
                    DELIMITER_PATTERNS
                )
            ],
            InjectionSeverity.LOW: [
                (re.compile(pattern, re.IGNORECASE), desc)
                for pattern, desc in ENCODING_PATTERNS
            ],
        }

    def check_input(self, user_input: str) -> FirewallResult:
        """
        Check user input for injection attempts.

        Args:
            user_input: User-provided text to check

        Returns:
            FirewallResult with detection details and recommended action
        """
        if not self.enabled:
            return FirewallResult(
                is_safe=True,
                detections=[],
                sanitized_input=None,
                action_taken=FirewallAction.LOG_ONLY,
                original_input=user_input
            )

        detections = []

        # Scan for injection patterns (severity order: CRITICAL → HIGH → MEDIUM → LOW)
        for severity in [InjectionSeverity.CRITICAL, InjectionSeverity.HIGH,
                        InjectionSeverity.MEDIUM, InjectionSeverity.LOW]:
            patterns = self._compiled_patterns.get(severity, [])

            for pattern, description in patterns:
                matches = pattern.finditer(user_input)
                for match in matches:
                    detection = InjectionDetection(
                        severity=severity,
                        pattern_name=description,
                        matched_text=match.group(0),
                        confidence=self._calculate_confidence(severity, match.group(0)),
                        description=f"Detected: {description}",
                        recommended_action=self._recommend_action(severity)
                    )
                    detections.append(detection)

        # Determine overall action
        if not detections:
            # Clean input
            return FirewallResult(
                is_safe=True,
                detections=[],
                sanitized_input=None,
                action_taken=FirewallAction.LOG_ONLY,
                original_input=user_input
            )

        # Get highest severity detection
        max_severity = max(d.severity for d in detections)
        action = self._determine_action(max_severity)

        # Sanitize if needed
        sanitized = None
        if action == FirewallAction.SANITIZE:
            sanitized = self._sanitize_input(user_input, detections)

        # Log detections
        self._log_detections(user_input, detections, action)

        return FirewallResult(
            is_safe=(action != FirewallAction.BLOCK),
            detections=detections,
            sanitized_input=sanitized,
            action_taken=action,
            original_input=user_input
        )

    def _calculate_confidence(self, severity: InjectionSeverity, matched_text: str) -> float:
        """Calculate confidence score for detection"""
        # Base confidence from severity
        base_confidence = {
            InjectionSeverity.CRITICAL: 0.95,
            InjectionSeverity.HIGH: 0.85,
            InjectionSeverity.MEDIUM: 0.70,
            InjectionSeverity.LOW: 0.50,
        }[severity]

        # Boost confidence for longer matches (more specific)
        length_boost = min(len(matched_text) / 100, 0.05)

        return min(base_confidence + length_boost, 1.0)

    def _recommend_action(self, severity: InjectionSeverity) -> FirewallAction:
        """Recommend action based on severity"""
        if severity == InjectionSeverity.CRITICAL:
            return FirewallAction.BLOCK
        elif severity == InjectionSeverity.HIGH:
            return FirewallAction.BLOCK
        elif severity == InjectionSeverity.MEDIUM:
            return FirewallAction.SANITIZE
        else:
            return FirewallAction.LOG_ONLY

    def _determine_action(self, max_severity: InjectionSeverity) -> FirewallAction:
        """Determine firewall action based on severity and config"""
        # If severity >= block_threshold, always BLOCK
        if self._severity_order(max_severity) >= self._severity_order(self.block_threshold):
            return FirewallAction.BLOCK

        # Otherwise, use configured action mode
        # But if action_mode is SANITIZE and we have detections, sanitize
        if self.action_mode == FirewallAction.SANITIZE and max_severity != InjectionSeverity.LOW:
            return FirewallAction.SANITIZE

        return self.action_mode

    def _severity_order(self, severity: InjectionSeverity) -> int:
        """Get numeric order for severity comparison"""
        return {
            InjectionSeverity.LOW: 1,
            InjectionSeverity.MEDIUM: 2,
            InjectionSeverity.HIGH: 3,
            InjectionSeverity.CRITICAL: 4,
        }[severity]

    def _sanitize_input(self, user_input: str, detections: List[InjectionDetection]) -> str:
        """
        Sanitize input by removing detected patterns.

        Args:
            user_input: Original input
            detections: List of detections to remove

        Returns:
            Sanitized input with patterns removed
        """
        sanitized = user_input

        # Remove detected patterns (highest severity first)
        sorted_detections = sorted(
            detections,
            key=lambda d: self._severity_order(d.severity),
            reverse=True
        )

        for detection in sorted_detections:
            # Replace matched text with [REMOVED]
            sanitized = sanitized.replace(
                detection.matched_text,
                "[REMOVED: Security violation]"
            )

        return sanitized

    def _log_detections(
        self, user_input: str, detections: List[InjectionDetection], action: FirewallAction
    ):
        """Log detected injection attempts"""
        self.logger.warning(
            f"🚨 INJECTION DETECTED: {len(detections)} pattern(s) matched, "
            f"action={action.value}"
        )

        for detection in detections[:3]:  # Log first 3
            self.logger.warning(
                f"   - {detection.severity.value.upper()}: {detection.pattern_name} "
                f"(confidence={detection.confidence:.2f})"
            )

        # Log to glass-box
        try:
            from src.core.unified_context_stream import (
                get_unified_context_stream,
                ContextEventType,
            )

            cs = get_unified_context_stream()
            cs.add_event(
                ContextEventType.ERROR,  # Security events
                {
                    "event_type": "injection_attempt_detected",
                    "detection_count": len(detections),
                    "max_severity": max(d.severity.value for d in detections),
                    "action_taken": action.value,
                    "patterns_matched": [d.pattern_name for d in detections[:5]],
                    "input_length": len(user_input),
                },
            )
        except Exception as e:
            self.logger.warning(f"Failed to log to glass-box: {e}")


# ============================================================================
# GLOBAL FIREWALL INSTANCE
# ============================================================================

_injection_firewall: Optional[InjectionFirewall] = None


def get_injection_firewall(
    enabled: bool = True,
    action_mode: FirewallAction = FirewallAction.SANITIZE,
    block_threshold: InjectionSeverity = InjectionSeverity.HIGH
) -> InjectionFirewall:
    """Get or create global injection firewall"""
    global _injection_firewall

    if _injection_firewall is None:
        _injection_firewall = InjectionFirewall(
            enabled=enabled,
            action_mode=action_mode,
            block_threshold=block_threshold
        )

    return _injection_firewall


# ============================================================================
# SECURITY EXCEPTION
# ============================================================================


class InjectionAttemptError(Exception):
    """Raised when injection attempt is blocked"""

    def __init__(self, message: str, detections: List[InjectionDetection]):
        super().__init__(message)
        self.detections = detections
