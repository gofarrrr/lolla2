"""
Week 4.1a: Unified Configuration Management System
Centralizes all engine configurations and system settings

Features:
- Single source of truth for all configurations
- Environment-based configuration overrides
- Configuration validation and schema enforcement
- Hot reloading for development
- Configuration drift detection
- Structured configuration documentation
"""

import os
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseSettings, validator, Field

from src.core.structured_logging import get_logger

logger = get_logger(__name__, component="configuration")


class EnvironmentType(str, Enum):
    """Deployment environment types"""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class CacheBackend(str, Enum):
    """Cache backend options"""

    MEMORY = "memory"
    REDIS = "redis"
    DISTRIBUTED = "distributed"


@dataclass
class LLMProviderConfig:
    """Configuration for LLM providers"""

    name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4000
    temperature: float = 0.7
    timeout_seconds: int = 30
    max_retries: int = 3
    enabled: bool = True
    cost_per_1k_tokens: float = 0.002

    # Provider-specific settings
    model_name: Optional[str] = None
    context_window: int = 8192
    supports_streaming: bool = True
    supports_function_calling: bool = False

    # Week 3.2 distributed caching settings
    cache_responses: bool = True
    cache_ttl_seconds: int = 3600


@dataclass
class CognitiveEngineConfig:
    """Configuration for cognitive processing engines"""

    enable_hmw_generation: bool = True
    enable_assumption_challenging: bool = True
    enable_devil_advocate: bool = True
    enable_research_grounding: bool = True
    enable_synthesis_arbitration: bool = True

    # Model selection settings
    max_models_per_phase: int = 5
    confidence_threshold: float = 0.8
    diversity_weight: float = 0.3
    synergy_weight: float = 0.4

    # Processing timeouts
    phase_timeout_seconds: int = 300  # 5 minutes
    synthesis_timeout_seconds: int = 180  # 3 minutes
    research_timeout_seconds: int = 45

    # Memory management (Week 3.1)
    max_reasoning_steps: int = 50
    max_integration_calls: int = 100
    enable_memory_limits: bool = True
    max_contract_size_mb: float = 50.0

    # Quality gates
    min_confidence_for_synthesis: float = 0.7
    required_challenger_consensus: int = 2


@dataclass
class DatabaseConfig:
    """Database configuration settings"""

    # Supabase settings
    supabase_url: Optional[str] = None
    supabase_anon_key: Optional[str] = None
    supabase_service_key: Optional[str] = None

    # Connection pooling
    max_connections: int = 20
    min_connections: int = 5
    connection_timeout_seconds: int = 30

    # Performance settings (Week 3.3)
    enable_query_optimization: bool = True
    index_maintenance_enabled: bool = True
    slow_query_threshold_ms: float = 1000.0

    # Data retention
    engagement_retention_days: int = 365
    audit_log_retention_days: int = 90
    performance_metrics_retention_days: int = 30


@dataclass
class CacheConfig:
    """Caching configuration (Week 3.2)"""

    backend: CacheBackend = CacheBackend.DISTRIBUTED
    redis_url: Optional[str] = None

    # Cache levels
    l1_memory_size: int = 1000
    l1_ttl_seconds: int = 300  # 5 minutes
    l2_redis_ttl_seconds: int = 3600  # 1 hour

    # Performance settings
    enable_compression: bool = True
    compression_threshold_bytes: int = 1024
    enable_circuit_breaker: bool = True

    # Cache key versioning for deployment safety
    cache_version: str = "v3.2"
    enable_cache_warming: bool = True


@dataclass
class TelemetryConfig:
    """Telemetry and monitoring configuration (Week 4.1)"""

    level: str = "standard"  # disabled, basic, standard, comprehensive
    collection_interval_seconds: int = 30

    # Exporters
    enable_prometheus: bool = True
    prometheus_port: int = 9090
    enable_opentelemetry: bool = True
    otlp_endpoint: Optional[str] = None

    # Storage
    metrics_retention_hours: int = 24
    trace_retention_hours: int = 6
    max_events_in_memory: int = 10000

    # Alerting
    enable_anomaly_detection: bool = True
    cpu_threshold_percent: float = 80.0
    memory_threshold_mb: float = 4000.0
    error_rate_threshold_percent: float = 5.0


@dataclass
class SecurityConfig:
    """Security configuration settings"""

    # Authentication
    jwt_secret_key: Optional[str] = None
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 1440  # 24 hours

    # API security
    enable_rate_limiting: bool = True
    rate_limit_requests_per_minute: int = 100
    enable_cors: bool = True
    allowed_origins: List[str] = field(
        default_factory=lambda: ["http://localhost:3001"]
    )

    # Data protection
    encrypt_sensitive_data: bool = True
    enable_audit_logging: bool = True
    mask_pii_in_logs: bool = True


class UnifiedConfiguration(BaseSettings):
    """
    Unified configuration system for all METIS components

    Week 4.1a Implementation:
    - Single source of truth for all configurations
    - Environment-based overrides
    - Validation and type safety
    - Hot reloading support
    """

    # Environment settings
    environment: EnvironmentType = Field(default=EnvironmentType.DEVELOPMENT)
    debug_mode: bool = Field(default=True)
    log_level: LogLevel = Field(default=LogLevel.INFO)

    # Core system settings
    system_name: str = Field(default="METIS Cognitive Platform")
    version: str = Field(default="4.1.0")
    instance_id: Optional[str] = Field(default=None)

    # LLM Providers (Week 1 integration)
    primary_llm_provider: str = Field(default="deepseek")
    fallback_llm_provider: str = Field(default="claude")

    llm_providers: Dict[str, LLMProviderConfig] = Field(
        default_factory=lambda: {
            "deepseek": LLMProviderConfig(
                name="DeepSeek V3.1",
                model_name="deepseek-reasoner",
                context_window=32768,
                max_tokens=8000,
                temperature=0.7,
                cost_per_1k_tokens=0.0014,
                supports_function_calling=True,
            ),
            "claude": LLMProviderConfig(
                name="Claude 3.5 Sonnet",
                model_name="claude-3-5-sonnet-20241022",
                context_window=200000,
                max_tokens=4000,
                temperature=0.7,
                cost_per_1k_tokens=0.003,
                supports_function_calling=True,
            ),
            "openai": LLMProviderConfig(
                name="GPT-4 Turbo",
                model_name="gpt-4-turbo-preview",
                context_window=128000,
                max_tokens=4000,
                temperature=0.7,
                cost_per_1k_tokens=0.01,
                supports_function_calling=True,
                enabled=False,  # Disabled by default
            ),
        }
    )

    # Cognitive engine settings
    cognitive_engine: CognitiveEngineConfig = Field(
        default_factory=CognitiveEngineConfig
    )

    # Database settings
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)

    # Caching settings (Week 3.2)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    # Telemetry settings (Week 4.1)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)

    # Security settings
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    # API settings
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_prefix: str = Field(default="/api/v2")
    enable_api_docs: bool = Field(default=True)

    # Frontend settings
    frontend_url: str = Field(default="http://localhost:3001")
    enable_cors: bool = Field(default=True)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"  # Support COGNITIVE_ENGINE__MAX_MODELS_PER_PHASE
        case_sensitive = False

    @validator("llm_providers", pre=True)
    def load_llm_api_keys(cls, v):
        """Load API keys from environment variables"""
        if isinstance(v, dict):
            for provider_name, config in v.items():
                if isinstance(config, dict):
                    # Convert dict to LLMProviderConfig if needed
                    config = LLMProviderConfig(**config)

                # Load API key from environment
                env_key = f"{provider_name.upper()}_API_KEY"
                api_key = os.getenv(env_key)
                if api_key:
                    config.api_key = api_key

                v[provider_name] = config
        return v

    @validator("database", pre=True)
    def load_database_credentials(cls, v):
        """Load database credentials from environment"""
        if isinstance(v, dict):
            v = DatabaseConfig(**v)
        elif not isinstance(v, DatabaseConfig):
            v = DatabaseConfig()

        # Load Supabase credentials
        v.supabase_url = os.getenv("SUPABASE_URL", v.supabase_url)
        v.supabase_anon_key = os.getenv("SUPABASE_ANON_KEY", v.supabase_anon_key)
        v.supabase_service_key = os.getenv(
            "SUPABASE_SERVICE_KEY", v.supabase_service_key
        )

        return v

    @validator("cache", pre=True)
    def load_cache_settings(cls, v):
        """Load cache settings from environment"""
        if isinstance(v, dict):
            v = CacheConfig(**v)
        elif not isinstance(v, CacheConfig):
            v = CacheConfig()

        # Load Redis URL
        v.redis_url = os.getenv("REDIS_URL", v.redis_url or "redis://localhost:6379")

        return v

    @validator("security", pre=True)
    def load_security_settings(cls, v):
        """Load security settings from environment"""
        if isinstance(v, dict):
            v = SecurityConfig(**v)
        elif not isinstance(v, SecurityConfig):
            v = SecurityConfig()

        # Load JWT secret
        v.jwt_secret_key = os.getenv("JWT_SECRET_KEY", v.jwt_secret_key)
        if not v.jwt_secret_key:
            import secrets

            v.jwt_secret_key = secrets.token_urlsafe(32)
            logger.warning(
                "🔐 Generated random JWT secret key - set JWT_SECRET_KEY in production"
            )

        return v

    def get_llm_config(self, provider_name: str) -> Optional[LLMProviderConfig]:
        """Get configuration for specific LLM provider"""
        return self.llm_providers.get(provider_name)

    def get_enabled_llm_providers(self) -> List[str]:
        """Get list of enabled LLM providers"""
        return [name for name, config in self.llm_providers.items() if config.enabled]

    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == EnvironmentType.PRODUCTION

    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == EnvironmentType.DEVELOPMENT

    def get_redis_url(self) -> str:
        """Get Redis URL with fallback"""
        return self.cache.redis_url or "redis://localhost:6379"

    def export_config(self, mask_secrets: bool = True) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        config_dict = self.dict()

        if mask_secrets:
            # Mask sensitive information
            for provider_name, provider_config in config_dict.get(
                "llm_providers", {}
            ).items():
                if "api_key" in provider_config and provider_config["api_key"]:
                    provider_config["api_key"] = "***masked***"

            if "database" in config_dict:
                db_config = config_dict["database"]
                for key in ["supabase_anon_key", "supabase_service_key"]:
                    if key in db_config and db_config[key]:
                        db_config[key] = "***masked***"

            if "security" in config_dict:
                sec_config = config_dict["security"]
                if "jwt_secret_key" in sec_config and sec_config["jwt_secret_key"]:
                    sec_config["jwt_secret_key"] = "***masked***"

        return config_dict

    def save_to_file(self, file_path: Union[str, Path], format: str = "yaml"):
        """Save configuration to file"""
        config_dict = self.export_config(mask_secrets=False)

        file_path = Path(file_path)

        if format.lower() == "yaml":
            import yaml

            with open(file_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif format.lower() == "json":
            with open(file_path, "w") as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"📁 Configuration saved to {file_path}")

    def validate_configuration(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []

        # Check LLM provider configurations
        enabled_providers = self.get_enabled_llm_providers()
        if not enabled_providers:
            issues.append("No LLM providers are enabled")

        for provider_name in enabled_providers:
            config = self.get_llm_config(provider_name)
            if not config.api_key:
                issues.append(
                    f"LLM provider '{provider_name}' is enabled but has no API key"
                )

        # Check database configuration
        if self.is_production() and not self.database.supabase_url:
            issues.append("Supabase URL is required in production")

        # Check cache configuration
        if self.cache.backend == CacheBackend.REDIS and not self.cache.redis_url:
            issues.append("Redis URL is required when using Redis cache backend")

        # Check security configuration
        if self.is_production() and self.debug_mode:
            issues.append("Debug mode should be disabled in production")

        return issues


# Global configuration instance
_config: Optional[UnifiedConfiguration] = None


def get_config() -> UnifiedConfiguration:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = UnifiedConfiguration()

        # Validate configuration
        issues = _config.validate_configuration()
        if issues:
            logger.warning(
                "⚠️ Configuration validation issues found",
                issues=issues,
                environment=_config.environment.value,
            )

        logger.info(
            "⚙️ Configuration loaded",
            environment=_config.environment.value,
            enabled_providers=_config.get_enabled_llm_providers(),
            cache_backend=_config.cache.backend.value,
            telemetry_level=_config.telemetry.level,
        )

    return _config


def reload_config() -> UnifiedConfiguration:
    """Reload configuration from environment"""
    global _config
    _config = None
    return get_config()


# Backward compatibility with legacy config
def get_settings() -> UnifiedConfiguration:
    """Backward compatibility alias"""
    return get_config()
