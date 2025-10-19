import asyncio
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import GammaConfig
from .exceptions import (
    GammaAPIError,
    RateLimitError,
    AuthenticationError,
    ValidationError,
)

logger = logging.getLogger(__name__)


class GammaAPIClient:
    """
    Gamma API Client for presentation generation
    Implements rate limiting and error handling
    """

    def __init__(self, config: Optional[GammaConfig] = None):
        self.config = config or GammaConfig()
        self.client = httpx.AsyncClient(
            base_url=self.config.api_base_url,
            headers={
                "X-API-KEY": self.config.api_key,
                "Content-Type": "application/json",
                "User-Agent": "METIS-Cognitive-Platform/1.0",
            },
            timeout=self.config.request_timeout,
        )
        self.generation_count = 0
        self._last_request_time = None
        self._request_count = 0
        self._minute_start = datetime.now()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def _rate_limit_check(self):
        """Implement rate limiting"""
        now = datetime.now()

        # Reset minute counter if needed
        if (now - self._minute_start).total_seconds() >= 60:
            self._request_count = 0
            self._minute_start = now

        # Check requests per minute limit
        if self._request_count >= self.config.max_requests_per_minute:
            wait_time = 60 - (now - self._minute_start).total_seconds()
            if wait_time > 0:
                logger.warning(f"⏸️ Rate limit reached, waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                self._request_count = 0
                self._minute_start = datetime.now()

        # Check delay between requests
        if self._last_request_time:
            elapsed = (now - self._last_request_time).total_seconds()
            if elapsed < self.config.rate_limit_delay:
                await asyncio.sleep(self.config.rate_limit_delay - elapsed)

        self._last_request_time = datetime.now()
        self._request_count += 1

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_presentation(
        self,
        input_text: str,
        text_mode: str = "generate",
        format: str = "presentation",
        theme_name: Optional[str] = None,
        num_cards: Optional[int] = None,
        additional_instructions: Optional[str] = None,
        text_options: Optional[Dict[str, Any]] = None,
        image_options: Optional[Dict[str, Any]] = None,
        export_as: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a presentation using Gamma API

        Args:
            input_text: Content to generate presentation from
            text_mode: "generate", "condense", or "preserve"
            format: "presentation", "document", or "social"
            theme_name: Name of the Gamma theme to use
            num_cards: Number of slides/cards to generate
            additional_instructions: Extra generation instructions
            text_options: Text generation options (tone, audience, etc.)
            image_options: Image generation options
            export_as: "pdf" or "pptx" for direct export

        Returns:
            Dictionary containing generation results and URLs
        """

        await self._rate_limit_check()

        # Check monthly generation limit
        if self.generation_count >= self.config.max_generations_per_month:
            raise RateLimitError("Monthly generation limit exceeded")

        # Validate input
        if not input_text.strip():
            raise ValidationError("Input text cannot be empty")

        if len(input_text) > 50000:  # Reasonable limit
            raise ValidationError("Input text too long (max 50,000 characters)")

        # Build request payload
        payload = {
            "inputText": input_text,
            "textMode": text_mode,
            "format": format,
            "themeName": theme_name or self.config.default_theme,
            "numCards": num_cards or self.config.default_num_cards,
            "cardSplit": "auto",
        }

        if additional_instructions:
            payload["additionalInstructions"] = additional_instructions

        if text_options:
            payload["textOptions"] = text_options
        else:
            payload["textOptions"] = {
                "amount": "detailed",
                "tone": "professional, insightful",
                "audience": "business professionals, decision makers",
                "language": self.config.default_language,
            }

        if image_options:
            payload["imageOptions"] = image_options
        else:
            payload["imageOptions"] = {
                "source": self.config.default_image_source,
                "model": self.config.default_image_model,
                "style": self.config.default_image_style,
            }

        if export_as:
            payload["exportAs"] = export_as

        # Sharing options for enterprise use
        payload["sharingOptions"] = {
            "workspaceAccess": self.config.workspace_access,
            "externalAccess": self.config.external_access,
        }

        try:
            logger.info(f"🎨 Generating {format} with Gamma API...")
            logger.debug(f"📝 Payload: {json.dumps(payload, indent=2)[:500]}...")

            response = await self.client.post("/generations", json=payload)

            # Handle different HTTP status codes
            if response.status_code == 401:
                raise AuthenticationError("Invalid API key")
            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                raise RateLimitError("Rate limit exceeded", retry_after)
            elif response.status_code == 400:
                error_data = response.json() if response.content else {}
                raise ValidationError(
                    error_data.get("message", "Request validation failed"),
                    error_data.get("field"),
                )
            elif response.status_code >= 500:
                raise GammaAPIError(f"Server error: {response.status_code}")

            response.raise_for_status()

            result = response.json()
            self.generation_count += 1

            logger.info(f"✅ Generation successful: {result.get('id', 'unknown')}")

            # Add metadata
            result["_metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "generation_count": self.generation_count,
                "input_length": len(input_text),
                "theme_used": theme_name or self.config.default_theme,
            }

            return result

        except httpx.HTTPStatusError as e:
            logger.error(f"❌ HTTP error: {e.response.status_code}")
            if e.response.status_code == 429:
                raise RateLimitError("Rate limit exceeded")
            raise GammaAPIError(f"API request failed: {e}")
        except httpx.RequestError as e:
            logger.error(f"❌ Request error: {e}")
            raise GammaAPIError(f"Request failed: {e}")
        except Exception as e:
            logger.error(f"❌ Gamma API error: {e}")
            raise GammaAPIError(str(e))

    async def get_generation_status(self, generation_id: str) -> Dict[str, Any]:
        """Check the status of a generation"""
        try:
            response = await self.client.get(f"/generations/{generation_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"❌ Failed to get generation status: {e}")
            raise GammaAPIError(f"Status check failed: {e}")

    async def download_export(self, export_url: str) -> bytes:
        """Download exported presentation file"""
        try:
            response = await self.client.get(export_url)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"❌ Failed to download export: {e}")
            raise GammaAPIError(f"Download failed: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Check API health and connection"""
        try:
            # Try a minimal request to test connectivity
            response = await self.client.get("/health", timeout=10.0)

            return {
                "status": "healthy" if response.status_code == 200 else "degraded",
                "status_code": response.status_code,
                "generation_count": self.generation_count,
                "rate_limit_remaining": max(
                    0, self.config.max_generations_per_month - self.generation_count
                ),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.warning(f"⚠️ Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "generation_count": self.generation_count,
                "timestamp": datetime.now().isoformat(),
            }

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        return {
            "total_generations": self.generation_count,
            "remaining_generations": max(
                0, self.config.max_generations_per_month - self.generation_count
            ),
            "requests_this_minute": self._request_count,
            "last_request": (
                self._last_request_time.isoformat() if self._last_request_time else None
            ),
        }
