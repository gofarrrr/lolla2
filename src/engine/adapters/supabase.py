"""Supabase adapter - bridges src.core.supabase_platform to src.engine"""

from src.core.supabase_platform import SupabasePlatform, get_supabase_client

__all__ = ["SupabasePlatform", "get_supabase_client"]
