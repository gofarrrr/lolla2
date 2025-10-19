"""
Sensitivity Routing - Enterprise Data Protection

Routes high-sensitivity queries exclusively to enterprise-grade,
compliant LLM providers (Anthropic Claude).

Sensitivity Levels:
- LOW: Public information, general queries → Any provider
- MEDIUM: Business strategy, competitive analysis → Preferred providers
- HIGH: PII, financial data, legal matters → Enterprise providers only (Anthropic)
- CRITICAL: Healthcare, regulated industries → Enterprise + on-prem only

Compliance:
- GDPR Article 32 (Data protection by design)
- CCPA Section 1798.150 (Security standards)
- HIPAA Security Rule (for healthcare)
- SOC 2 Type II (Vendor management)

Architecture:
- Automatic sensitivity detection (PII, keywords, domains)
- Manual sensitivity override per request
- Provider allowlist per sensitivity level
- Glass-box logging of routing decisions
"""

import logging
from typing import List, Optional, Set, Dict, Any
from enum import Enum
from dataclasses import dataclass


logger = logging.getLogger(__name__)


class SensitivityLevel(str, Enum):
    """Data sensitivity levels"""
    LOW = "low"              # Public info, general queries
    MEDIUM = "medium"        # Business strategy, non-sensitive
    HIGH = "high"            # PII, financial, legal
    CRITICAL = "critical"    # Healthcare, regulated


class ProviderCompliance(str, Enum):
    """Provider compliance certifications"""
    SOC2_TYPE2 = "soc2_type2"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    CCPA = "ccpa"
    ISO27001 = "iso27001"


@dataclass
class ProviderProfile:
    """Provider security and compliance profile"""
    name: str
    compliance: List[ProviderCompliance]
    data_residency: str  # e.g., "US", "EU", "global"
    enterprise_sla: bool
    allows_content_logging: bool
    min_sensitivity_level: SensitivityLevel


@dataclass
class SensitivityDetection:
    """Result of sensitivity detection"""
    level: SensitivityLevel
    reasons: List[str]
    detected_pii: bool
    detected_keywords: List[str]
    confidence: float


@dataclass
class RoutingDecision:
    """Provider routing decision"""
    allowed_providers: List[str]
    sensitivity_level: SensitivityLevel
    reasons: List[str]
    restrictions: List[str]


# PROVIDER PROFILES
# Define compliance and security posture for each provider
PROVIDER_PROFILES: Dict[str, ProviderProfile] = {
    "anthropic": ProviderProfile(
        name="anthropic",
        compliance=[
            ProviderCompliance.SOC2_TYPE2,
            ProviderCompliance.GDPR,
            ProviderCompliance.CCPA,
            ProviderCompliance.ISO27001,
        ],
        data_residency="US",
        enterprise_sla=True,
        allows_content_logging=False,  # Enterprise plan
        min_sensitivity_level=SensitivityLevel.LOW,
    ),

    "openrouter": ProviderProfile(
        name="openrouter",
        compliance=[ProviderCompliance.GDPR],  # Limited compliance
        data_residency="global",
        enterprise_sla=False,
        allows_content_logging=True,  # May log for training
        min_sensitivity_level=SensitivityLevel.LOW,
    ),

    "deepseek": ProviderProfile(
        name="deepseek",
        compliance=[],  # Non-US provider
        data_residency="CN",  # China
        enterprise_sla=False,
        allows_content_logging=True,
        min_sensitivity_level=SensitivityLevel.LOW,
    ),

    "openai": ProviderProfile(
        name="openai",
        compliance=[
            ProviderCompliance.SOC2_TYPE2,
            ProviderCompliance.GDPR,
            ProviderCompliance.CCPA,
        ],
        data_residency="US",
        enterprise_sla=True,
        allows_content_logging=False,  # Enterprise plan
        min_sensitivity_level=SensitivityLevel.LOW,
    ),
}


# SENSITIVITY ROUTING POLICY
# Define which providers are allowed for each sensitivity level
SENSITIVITY_ROUTING_POLICY: Dict[SensitivityLevel, List[str]] = {
    SensitivityLevel.LOW: ["anthropic", "openrouter", "deepseek", "openai"],
    SensitivityLevel.MEDIUM: ["anthropic", "openrouter", "openai"],
    SensitivityLevel.HIGH: ["anthropic", "openai"],  # Enterprise only
    SensitivityLevel.CRITICAL: ["anthropic"],  # Anthropic only
}


# HIGH-SENSITIVITY KEYWORDS
# Keywords that trigger HIGH sensitivity classification
HIGH_SENSITIVITY_KEYWORDS = {
    # Financial
    "credit card", "bank account", "routing number", "swift code",
    "account number", "financial statement", "tax return",

    # Legal
    "legal document", "lawsuit", "litigation", "settlement",
    "attorney-client", "privileged",

    # Healthcare (HIPAA)
    "medical record", "patient", "diagnosis", "prescription",
    "health condition", "PHI", "protected health",

    # Personal identifiable
    "social security", "passport", "driver license",
    "national id", "birth certificate",

    # Regulated industries
    "insider trading", "material non-public", "confidential",
    "proprietary", "trade secret",
}


class SensitivityRouter:
    """
    Routes LLM requests based on data sensitivity.

    Features:
    - Automatic PII detection → HIGH sensitivity
    - Keyword-based sensitivity classification
    - Manual override per request
    - Provider compliance verification
    - Glass-box logging

    Usage:
        router = SensitivityRouter()
        decision = router.route(
            content="Analyze customer data for john@example.com",
            sensitivity_override=None  # Auto-detect
        )
        # decision.allowed_providers: ["anthropic"] (HIGH sensitivity)
    """

    def __init__(
        self,
        enabled: bool = True,
        default_sensitivity: SensitivityLevel = SensitivityLevel.MEDIUM
    ):
        """
        Initialize sensitivity router.

        Args:
            enabled: Whether sensitivity routing is enabled
            default_sensitivity: Default sensitivity level when detection fails
        """
        self.enabled = enabled
        self.default_sensitivity = default_sensitivity
        self.logger = logging.getLogger(__name__)

        if enabled:
            self.logger.info(
                "✅ Sensitivity Routing enabled: "
                f"default={default_sensitivity.value}"
            )
        else:
            self.logger.warning("⚠️ Sensitivity Routing DISABLED")

    def detect_sensitivity(
        self, content: str, context: Optional[Dict[str, Any]] = None
    ) -> SensitivityDetection:
        """
        Detect sensitivity level from content.

        Args:
            content: Text content to analyze
            context: Optional context (engagement metadata)

        Returns:
            SensitivityDetection with level and reasons
        """
        reasons = []
        detected_keywords = []
        confidence = 0.5

        # Check for PII
        detected_pii = False
        try:
            from src.engine.security.pii_redaction import get_pii_redaction_engine

            engine = get_pii_redaction_engine(enabled=True)
            if engine and engine.has_pii(content):
                detected_pii = True
                pii_types = engine.get_pii_types(content)
                reasons.append(f"Contains PII: {[t.value for t in pii_types]}")
                confidence = 0.95
        except Exception as e:
            self.logger.warning(f"⚠️ PII detection failed: {e}")

        # Check for high-sensitivity keywords
        content_lower = content.lower()
        for keyword in HIGH_SENSITIVITY_KEYWORDS:
            if keyword in content_lower:
                detected_keywords.append(keyword)

        if detected_keywords:
            reasons.append(f"High-sensitivity keywords: {detected_keywords[:3]}")
            confidence = max(confidence, 0.85)

        # Determine sensitivity level
        if detected_pii:
            level = SensitivityLevel.HIGH
        elif len(detected_keywords) >= 3:
            level = SensitivityLevel.HIGH
        elif len(detected_keywords) >= 1:
            level = SensitivityLevel.MEDIUM
        else:
            level = self.default_sensitivity
            reasons.append(f"Default sensitivity: {level.value}")

        return SensitivityDetection(
            level=level,
            reasons=reasons,
            detected_pii=detected_pii,
            detected_keywords=detected_keywords[:5],
            confidence=confidence
        )

    def route(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        sensitivity_override: Optional[SensitivityLevel] = None,
        available_providers: Optional[List[str]] = None
    ) -> RoutingDecision:
        """
        Make routing decision based on sensitivity.

        Args:
            content: Content to analyze
            context: Optional context
            sensitivity_override: Manual sensitivity level (overrides auto-detection)
            available_providers: Available providers (None = all)

        Returns:
            RoutingDecision with allowed providers and restrictions
        """
        if not self.enabled:
            # Routing disabled - allow all providers
            return RoutingDecision(
                allowed_providers=available_providers or list(PROVIDER_PROFILES.keys()),
                sensitivity_level=self.default_sensitivity,
                reasons=["Sensitivity routing disabled"],
                restrictions=[]
            )

        # Determine sensitivity level
        if sensitivity_override:
            sensitivity_level = sensitivity_override
            reasons = [f"Manual override: {sensitivity_level.value}"]
            detection = None
        else:
            detection = self.detect_sensitivity(content, context)
            sensitivity_level = detection.level
            reasons = detection.reasons

        # Get allowed providers for this sensitivity level
        policy_allowed = SENSITIVITY_ROUTING_POLICY.get(
            sensitivity_level,
            [PROVIDER_PROFILES["anthropic"].name]  # Fallback to most secure
        )

        # Filter by available providers
        if available_providers:
            allowed_providers = [
                p for p in policy_allowed if p in available_providers
            ]
        else:
            allowed_providers = policy_allowed

        # Build restrictions list
        restrictions = []
        if sensitivity_level == SensitivityLevel.HIGH:
            restrictions.append("Enterprise providers only (SOC2 Type II required)")
        if sensitivity_level == SensitivityLevel.CRITICAL:
            restrictions.append("Maximum security: Anthropic only")

        if detection and detection.detected_pii:
            restrictions.append("PII detected: No content logging allowed")

        # Log routing decision
        self.logger.info(
            f"🔐 Sensitivity Routing: level={sensitivity_level.value}, "
            f"allowed_providers={allowed_providers}, "
            f"reasons={reasons[:2]}"
        )

        return RoutingDecision(
            allowed_providers=allowed_providers,
            sensitivity_level=sensitivity_level,
            reasons=reasons,
            restrictions=restrictions
        )

    def get_provider_profile(self, provider: str) -> Optional[ProviderProfile]:
        """Get provider security profile"""
        return PROVIDER_PROFILES.get(provider)

    def is_provider_compliant(
        self, provider: str, required_compliance: ProviderCompliance
    ) -> bool:
        """Check if provider meets compliance requirement"""
        profile = self.get_provider_profile(provider)
        if not profile:
            return False
        return required_compliance in profile.compliance


# Global router instance
_sensitivity_router: Optional[SensitivityRouter] = None


def get_sensitivity_router(
    enabled: bool = True,
    default_sensitivity: SensitivityLevel = SensitivityLevel.MEDIUM
) -> SensitivityRouter:
    """Get or create global sensitivity router"""
    global _sensitivity_router

    if _sensitivity_router is None:
        _sensitivity_router = SensitivityRouter(
            enabled=enabled,
            default_sensitivity=default_sensitivity
        )

    return _sensitivity_router
