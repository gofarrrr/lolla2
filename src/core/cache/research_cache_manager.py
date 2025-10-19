"""
Research Cache Manager for Monte Carlo Calibration Loop.

This module provides deterministic caching of LLM and research API responses
to enable consistent A/B testing across different code versions.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ResearchCacheManager:
    """
    Manages caching of research data for deterministic Monte Carlo testing.

    Key Features:
    - Per-LLM-call granular caching
    - Hash-based cache keys for deterministic retrieval
    - Versioned cache storage for Golden Standards
    - RECORD/PLAYBACK mode support
    """

    def __init__(
        self,
        cache_root: str = "tests/benchmarks/golden_standards",
        version: str = "v2.1",
    ):
        """
        Initialize the Research Cache Manager.

        Args:
            cache_root: Root directory for cache storage
            version: Version identifier for cache isolation
        """
        self.cache_root = Path(cache_root)
        self.version = version
        self.cache_dir = self.cache_root / version / "research_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.mode = "PLAYBACK"  # Default to PLAYBACK mode
        self.hit_count = 0
        self.miss_count = 0

        logger.info(
            f"ResearchCacheManager initialized: {self.cache_dir}, mode: {self.mode}"
        )

    def set_mode(self, mode: str) -> None:
        """
        Set cache operation mode.

        Args:
            mode: Either 'RECORD' or 'PLAYBACK'
        """
        if mode not in ["RECORD", "PLAYBACK"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'RECORD' or 'PLAYBACK'")

        self.mode = mode
        logger.info(f"Cache mode set to: {mode}")

    def _generate_cache_key(
        self, prompt: str, provider: str, model: str, **kwargs
    ) -> str:
        """
        Generate deterministic cache key from LLM call parameters.

        Args:
            prompt: The complete prompt sent to the LLM
            provider: LLM provider (e.g., 'deepseek', 'claude', 'perplexity')
            model: Model identifier
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            SHA-256 hash of normalized parameters
        """
        # Create normalized parameter dict
        cache_params = {
            "prompt": prompt.strip(),
            "provider": provider.lower(),
            "model": model.lower(),
            **{k: v for k, v in sorted(kwargs.items()) if v is not None},
        }

        # Generate deterministic JSON string
        cache_string = json.dumps(cache_params, sort_keys=True, separators=(",", ":"))

        # Create hash
        return hashlib.sha256(cache_string.encode("utf-8")).hexdigest()

    def _get_cache_file_path(self, cache_key: str) -> Path:
        """Get file path for cache entry."""
        return self.cache_dir / f"{cache_key}.json"

    def get_cached_response(
        self, prompt: str, provider: str, model: str, **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached response for LLM call.

        Args:
            prompt: The complete prompt sent to the LLM
            provider: LLM provider identifier
            model: Model identifier
            **kwargs: Additional LLM parameters

        Returns:
            Cached response if found, None otherwise
        """
        cache_key = self._generate_cache_key(prompt, provider, model, **kwargs)
        cache_file = self._get_cache_file_path(cache_key)

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)

                self.hit_count += 1
                logger.debug(f"Cache HIT: {cache_key[:16]}...")
                return cached_data["response"]

            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Cache corruption detected for {cache_key}: {e}")
                return None

        self.miss_count += 1
        logger.debug(f"Cache MISS: {cache_key[:16]}...")
        return None

    def store_response(
        self, prompt: str, provider: str, model: str, response: Dict[str, Any], **kwargs
    ) -> str:
        """
        Store LLM response in cache.

        Args:
            prompt: The complete prompt sent to the LLM
            provider: LLM provider identifier
            model: Model identifier
            response: LLM response to cache
            **kwargs: Additional LLM parameters

        Returns:
            Cache key for the stored response
        """
        if self.mode != "RECORD":
            logger.warning(
                f"Attempted to store response in {self.mode} mode - ignoring"
            )
            return ""

        cache_key = self._generate_cache_key(prompt, provider, model, **kwargs)
        cache_file = self._get_cache_file_path(cache_key)

        cache_entry = {
            "cache_key": cache_key,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "provider": provider,
                "model": model,
                "prompt_length": len(prompt),
                "version": self.version,
                **kwargs,
            },
            "response": response,
        }

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_entry, f, indent=2, ensure_ascii=False)

            logger.debug(f"Response cached: {cache_key[:16]}...")
            return cache_key

        except Exception as e:
            logger.error(f"Failed to cache response {cache_key}: {e}")
            return ""

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.

        Returns:
            Dictionary with hit/miss ratios and cache info
        """
        total_requests = self.hit_count + self.miss_count
        hit_ratio = (self.hit_count / total_requests) if total_requests > 0 else 0.0

        # Count cache files
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "mode": self.mode,
            "version": self.version,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_ratio": hit_ratio,
            "total_cache_entries": len(cache_files),
            "total_cache_size_mb": total_size / (1024 * 1024),
            "cache_directory": str(self.cache_dir),
        }

    def clear_cache(self) -> int:
        """
        Clear all cache entries for current version.

        Returns:
            Number of files deleted
        """
        cache_files = list(self.cache_dir.glob("*.json"))
        deleted_count = 0

        for cache_file in cache_files:
            try:
                cache_file.unlink()
                deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to delete cache file {cache_file}: {e}")

        # Reset counters
        self.hit_count = 0
        self.miss_count = 0

        logger.info(f"Cache cleared: {deleted_count} files deleted")
        return deleted_count

    def export_cache_manifest(self) -> Dict[str, Any]:
        """
        Export cache manifest for Golden Standards documentation.

        Returns:
            Manifest with cache metadata and statistics
        """
        cache_files = list(self.cache_dir.glob("*.json"))

        manifest = {
            "version": self.version,
            "generated_at": datetime.now().isoformat(),
            "cache_statistics": self.get_cache_stats(),
            "entries": [],
        }

        for cache_file in cache_files:
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)

                manifest["entries"].append(
                    {
                        "cache_key": cache_data.get("cache_key", "unknown"),
                        "timestamp": cache_data.get("timestamp", "unknown"),
                        "provider": cache_data.get("metadata", {}).get(
                            "provider", "unknown"
                        ),
                        "model": cache_data.get("metadata", {}).get("model", "unknown"),
                        "prompt_length": cache_data.get("metadata", {}).get(
                            "prompt_length", 0
                        ),
                    }
                )

            except Exception as e:
                logger.error(f"Failed to read cache file {cache_file}: {e}")

        return manifest


# Global cache manager instance
_global_cache_manager: Optional[ResearchCacheManager] = None


def get_research_cache_manager(version: str = "v2.1") -> ResearchCacheManager:
    """
    Get global research cache manager instance.

    Args:
        version: Cache version identifier

    Returns:
        ResearchCacheManager instance
    """
    global _global_cache_manager

    if _global_cache_manager is None or _global_cache_manager.version != version:
        _global_cache_manager = ResearchCacheManager(version=version)

    return _global_cache_manager


def set_cache_mode(mode: str) -> None:
    """
    Set global cache mode.

    Args:
        mode: Either 'RECORD' or 'PLAYBACK'
    """
    cache_manager = get_research_cache_manager()
    cache_manager.set_mode(mode)
