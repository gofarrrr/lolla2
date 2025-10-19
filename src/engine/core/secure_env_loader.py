"""
Secure Environment Variable Loader
Centralized, validated environment variable loading with security best practices
"""

import os
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    """Configuration class for environment variables"""

    anthropic_api_key: Optional[str] = None
    perplexity_api_key: Optional[str] = None
    supabase_url: Optional[str] = None
    supabase_anon_key: Optional[str] = None
    supabase_service_role_key: Optional[str] = None
    postgres_url: Optional[str] = None
    redis_url: Optional[str] = None


class SecureEnvLoader:
    """Secure environment variable loader with validation"""

    def __init__(self):
        self.config = EnvConfig()
        self._load_environment()

    def _load_environment(self):
        """Load environment variables with validation"""
        try:
            from dotenv import load_dotenv

            # Find .env file
            env_path = Path(__file__).parent.parent.parent / ".env"
            if env_path.exists():
                load_dotenv(
                    env_path, override=False
                )  # Don't override existing env vars
                logger.info(f"✅ Loaded .env file from {env_path}")
            else:
                logger.warning(f"⚠️ No .env file found at {env_path}")
        except ImportError:
            logger.warning("⚠️ python-dotenv not available")

        # Load and validate configuration
        self.config.anthropic_api_key = self._get_secure_env(
            "ANTHROPIC_API_KEY", required=False
        )
        self.config.perplexity_api_key = self._get_secure_env(
            "PERPLEXITY_API_KEY", required=False
        )
        self.config.supabase_url = self._get_secure_env(
            "NEXT_PUBLIC_SUPABASE_URL", required=False
        )
        self.config.supabase_anon_key = self._get_secure_env(
            "NEXT_PUBLIC_SUPABASE_ANON_KEY", required=False
        )
        self.config.supabase_service_role_key = self._get_secure_env(
            "SUPABASE_SERVICE_ROLE_KEY", required=False
        )
        self.config.postgres_url = self._get_secure_env("POSTGRES_URL", required=False)
        self.config.redis_url = self._get_secure_env("REDIS_URL", required=False)

    def _get_secure_env(
        self, key: str, required: bool = False, default: Optional[str] = None
    ) -> Optional[str]:
        """Securely get environment variable with validation"""
        value = os.getenv(key, default)

        if required and not value:
            logger.error(f"❌ Required environment variable {key} is not set")
            raise ValueError(f"Required environment variable {key} is not set")

        if value:
            # Validate API key format
            if "API_KEY" in key:
                if not self._validate_api_key(key, value):
                    logger.warning(f"⚠️ Invalid format for {key}")
                    return None

            # Don't log sensitive values
            masked_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            logger.info(f"✅ Loaded {key}: {masked_value}")

        return value

    def _validate_api_key(self, key: str, value: str) -> bool:
        """Validate API key format"""
        if key == "ANTHROPIC_API_KEY":
            return value.startswith("sk-ant-")
        elif key == "PERPLEXITY_API_KEY":
            return value.startswith("pplx-")
        elif "SUPABASE" in key and "KEY" in key:
            return len(value) > 20  # Basic length check for JWT tokens
        return True  # Default validation

    def get_config(self) -> EnvConfig:
        """Get loaded configuration"""
        return self.config

    def is_anthropic_available(self) -> bool:
        """Check if Anthropic API is available"""
        return bool(self.config.anthropic_api_key)

    def is_perplexity_available(self) -> bool:
        """Check if Perplexity API is available"""
        return bool(self.config.perplexity_api_key)

    def is_supabase_available(self) -> bool:
        """Check if Supabase configuration is available"""
        return bool(self.config.supabase_url and self.config.supabase_anon_key)


# Global instance
_env_loader = None


def get_env_loader() -> SecureEnvLoader:
    """Get global environment loader instance"""
    global _env_loader
    if _env_loader is None:
        _env_loader = SecureEnvLoader()
    return _env_loader


def get_env_config() -> EnvConfig:
    """Get environment configuration"""
    return get_env_loader().get_config()
