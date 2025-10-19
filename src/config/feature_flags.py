"""
Feature flagging utilities (beyond env toggles)
- Supports env vars (FF_*), with optional override via Supabase system_config table
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

try:
    # Optional import; fail open if unavailable
    from src.storage.supabase_store import SupabaseStore  # type: ignore
except Exception:  # pragma: no cover - optional
    SupabaseStore = None  # type: ignore


class FeatureFlags:
    def __init__(self, supabase: Optional[SupabaseStore] = None):  # type: ignore
        self.supabase = supabase

    def _env_bool(self, key: str, default: bool) -> bool:
        v = os.getenv(key)
        if v is None:
            return default
        return v.lower() in ("1", "true", "yes", "on")

    async def get(self, key: str, default: Any = None) -> Any:
        # 1) Supabase override
        if self.supabase is not None:
            try:
                cfg = await self.supabase.get_system_config(key)
                if cfg and isinstance(cfg, dict):
                    val = cfg.get("config_value")
                    if val is not None:
                        return val
            except Exception:
                pass  # Ignore Supabase failures
        # 2) Env fallback
        return os.getenv(key, default)

    async def enabled(self, key: str, default: bool = False) -> bool:
        # Try Supabase config first
        val = await self.get(key, None)
        if val is None:
            # FF_ prefix for env flags
            return self._env_bool(f"FF_{key}", default)
        # Normalize truthy values
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return val != 0
        return str(val).lower() in ("1", "true", "yes", "on")


# Singleton
_flags: Optional[FeatureFlags] = None


def get_feature_flags() -> FeatureFlags:
    global _flags
    if _flags is None:
        supabase = None
        try:
            supabase = SupabaseStore() if SupabaseStore else None
        except Exception:
            supabase = None
        _flags = FeatureFlags(supabase)
    return _flags
