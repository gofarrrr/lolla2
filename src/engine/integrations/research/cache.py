"""
Research caching system with TTL and LRU eviction
"""

import hashlib
import json
import re
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging

from .models import ResearchResult


class ResearchCache:
    """Simple in-memory cache with TTL and LRU eviction"""

    def __init__(self, max_entries: int = 500, ttl_hours: int = 24):
        self.max_entries = max_entries
        self.ttl_delta = timedelta(hours=ttl_hours)
        self.cache = OrderedDict()  # LRU with OrderedDict
        self.logger = logging.getLogger(__name__)

    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent caching"""
        return re.sub(r"\s+", " ", query.lower().strip())

    def _hash_context(self, context: Dict[str, Any]) -> str:
        """Create stable hash of context"""
        if not context:
            return "no_context"

        # Sort keys for stable hashing
        stable_context = json.dumps(context, sort_keys=True, default=str)
        return hashlib.md5(stable_context.encode()).hexdigest()[:16]

    def _make_key(self, query: str, context: Dict[str, Any]) -> str:
        """Generate cache key"""
        normalized_query = self._normalize_query(query)
        context_hash = self._hash_context(context)
        return f"{normalized_query}::{context_hash}"

    def _evict_expired(self):
        """Remove expired entries"""
        now = datetime.utcnow()
        expired_keys = []

        for key, entry in self.cache.items():
            if now > entry["expires_at"]:
                expired_keys.append(key)

        for key in expired_keys:
            del self.cache[key]

    def _enforce_lru_limit(self):
        """Evict oldest entries if over limit"""
        while len(self.cache) > self.max_entries:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

    def get(self, query: str, context: Dict[str, Any]) -> Optional[ResearchResult]:
        """Get cached result if available and not expired"""
        self._evict_expired()

        key = self._make_key(query, context)

        if key in self.cache:
            entry = self.cache[key]
            # Move to end (most recently accessed)
            self.cache.move_to_end(key)
            entry["last_accessed"] = datetime.utcnow()

            # Mark as cache hit
            result = entry["result"]
            result.cache_hit = True

            self.logger.debug(f"Cache hit for query: {query[:50]}...")
            return result

        return None

    def put(self, query: str, context: Dict[str, Any], result: ResearchResult):
        """Store result in cache"""
        key = self._make_key(query, context)

        entry = {
            "result": result,
            "expires_at": datetime.utcnow() + self.ttl_delta,
            "last_accessed": datetime.utcnow(),
        }

        self.cache[key] = entry
        self._enforce_lru_limit()

        self.logger.debug(f"Cached result for query: {query[:50]}...")
