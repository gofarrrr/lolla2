"""
Faculty Showcase API - METIS V5 Public-Facing Consultant Registry
REST API endpoints for the public-facing consultant showcase system

This API provides:
1. GET /api/consultants - List all consultant profiles with heavy caching
2. GET /api/consultants/{agent_id} - Get single consultant profile by ID with heavy caching

Designed for public consumption with IP protection and performance optimization.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime, timedelta
import asyncio

# Import our contracts
from src.contracts.personas import ConsultantProfile

# Supabase integration
try:
    from supabase import create_client, Client
    import os

    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Faculty Showcase"])

# Cache configuration
CACHE_DURATION_HOURS = 24  # 24-hour cache for consultant profiles
CACHE_DURATION_SECONDS = CACHE_DURATION_HOURS * 3600

# In-memory cache
_consultant_profiles_cache: Optional[List[ConsultantProfile]] = None
_cache_timestamp: Optional[datetime] = None
_cache_lock = asyncio.Lock()

# Supabase client
_supabase_client: Optional[Client] = None


def get_supabase_client() -> Optional[Client]:
    """Get or create Supabase client"""
    global _supabase_client

    if not SUPABASE_AVAILABLE:
        return None

    if _supabase_client is None:
        supabase_url = os.getenv("NEXT_PUBLIC_SUPABASE_URL") or os.getenv(
            "SUPABASE_URL"
        )
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        if supabase_url and supabase_key:
            _supabase_client = create_client(supabase_url, supabase_key)
            logger.info("📊 Supabase client initialized for Faculty Showcase API")
        else:
            logger.warning(
                "⚠️ Supabase credentials not available for Faculty Showcase API"
            )

    return _supabase_client


def is_cache_valid() -> bool:
    """Check if cache is still valid"""
    if _cache_timestamp is None or _consultant_profiles_cache is None:
        return False

    cache_expiry = _cache_timestamp + timedelta(seconds=CACHE_DURATION_SECONDS)
    return datetime.now() < cache_expiry


async def refresh_consultant_profiles_cache() -> List[ConsultantProfile]:
    """Refresh consultant profiles from database"""
    global _consultant_profiles_cache, _cache_timestamp

    async with _cache_lock:
        # Double-check cache validity under lock
        if is_cache_valid():
            return _consultant_profiles_cache

        logger.info("🔄 Refreshing consultant profiles cache from database...")

        # Get Supabase client
        supabase = get_supabase_client()
        if not supabase:
            raise HTTPException(status_code=503, detail="Database service unavailable")

        try:
            # Fetch all consultant profiles from database
            result = (
                supabase.table("consultant_profiles")
                .select("*")
                .order("matrix_position", desc=False)
                .execute()
            )

            if not result.data:
                logger.warning("⚠️ No consultant profiles found in database")
                _consultant_profiles_cache = []
                _cache_timestamp = datetime.now()
                return []

            # Convert to ConsultantProfile objects with validation
            validated_profiles = []
            for profile_data in result.data:
                try:
                    # Remove database-specific fields
                    clean_data = {
                        k: v
                        for k, v in profile_data.items()
                        if k not in ["id", "created_at", "updated_at"]
                    }

                    profile = ConsultantProfile(**clean_data)
                    validated_profiles.append(profile)
                except Exception as e:
                    logger.error(
                        f"❌ Failed to validate profile {profile_data.get('agent_id', 'unknown')}: {e}"
                    )
                    continue

            _consultant_profiles_cache = validated_profiles
            _cache_timestamp = datetime.now()

            logger.info(f"✅ Cached {len(validated_profiles)} consultant profiles")
            return validated_profiles

        except Exception as e:
            logger.error(f"❌ Failed to refresh consultant profiles cache: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to load consultant profiles"
            )


async def get_cached_consultant_profiles() -> List[ConsultantProfile]:
    """Get consultant profiles from cache, refreshing if needed"""
    if is_cache_valid():
        logger.debug("✅ Using cached consultant profiles")
        return _consultant_profiles_cache

    return await refresh_consultant_profiles_cache()


# Response Models
class ConsultantProfileResponse(BaseModel):
    """Response model for individual consultant profile"""

    profile: ConsultantProfile
    cached: bool = True
    cache_age_minutes: Optional[int] = None


class ConsultantsListResponse(BaseModel):
    """Response model for consultant profiles list"""

    consultants: List[ConsultantProfile]
    count: int
    cached: bool = True
    cache_age_minutes: Optional[int] = None


def calculate_cache_age() -> Optional[int]:
    """Calculate cache age in minutes"""
    if _cache_timestamp:
        delta = datetime.now() - _cache_timestamp
        return int(delta.total_seconds() / 60)
    return None


# API Endpoints


@router.get(
    "/consultants",
    response_model=ConsultantsListResponse,
    summary="Get All Consultant Profiles",
    description="Retrieve all consultant profiles for the Faculty Showcase. Results are heavily cached for performance.",
)
async def get_all_consultants(
    background_tasks: BackgroundTasks,
) -> ConsultantsListResponse:
    """Get all consultant profiles with heavy caching"""
    try:
        profiles = await get_cached_consultant_profiles()

        # Schedule background cache refresh if cache is getting old
        cache_age = calculate_cache_age()
        if cache_age and cache_age > (
            CACHE_DURATION_HOURS * 60 * 0.8
        ):  # 80% of cache duration
            background_tasks.add_task(refresh_consultant_profiles_cache)

        return ConsultantsListResponse(
            consultants=profiles,
            count=len(profiles),
            cached=is_cache_valid(),
            cache_age_minutes=cache_age,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in get_all_consultants: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/consultants/{agent_id}",
    response_model=ConsultantProfileResponse,
    summary="Get Consultant Profile by ID",
    description="Retrieve a specific consultant profile by agent_id. Results are heavily cached for performance.",
)
async def get_consultant_by_id(
    agent_id: str, background_tasks: BackgroundTasks
) -> ConsultantProfileResponse:
    """Get specific consultant profile by agent_id with heavy caching"""
    try:
        profiles = await get_cached_consultant_profiles()

        # Find the requested profile
        target_profile = None
        for profile in profiles:
            if profile.agent_id == agent_id:
                target_profile = profile
                break

        if not target_profile:
            raise HTTPException(
                status_code=404, detail=f"Consultant profile not found: {agent_id}"
            )

        # Schedule background cache refresh if cache is getting old
        cache_age = calculate_cache_age()
        if cache_age and cache_age > (
            CACHE_DURATION_HOURS * 60 * 0.8
        ):  # 80% of cache duration
            background_tasks.add_task(refresh_consultant_profiles_cache)

        return ConsultantProfileResponse(
            profile=target_profile, cached=is_cache_valid(), cache_age_minutes=cache_age
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in get_consultant_by_id: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/consultants/refresh-cache",
    summary="Refresh Consultant Profiles Cache",
    description="Force refresh of consultant profiles cache. Admin endpoint for cache management.",
)
async def refresh_cache():
    """Force refresh of consultant profiles cache - admin endpoint"""
    try:
        profiles = await refresh_consultant_profiles_cache()

        return {
            "message": "Cache refreshed successfully",
            "profiles_loaded": len(profiles),
            "cache_timestamp": (
                _cache_timestamp.isoformat() if _cache_timestamp else None
            ),
        }

    except Exception as e:
        logger.error(f"❌ Error refreshing cache: {e}")
        raise HTTPException(status_code=500, detail=f"Cache refresh failed: {str(e)}")


@router.get(
    "/consultants/system/health",
    summary="Faculty Showcase Health Check",
    description="Health check endpoint for the Faculty Showcase system",
)
async def health_check():
    """Health check for Faculty Showcase system"""

    # Check cache status
    cache_valid = is_cache_valid()
    cache_age = calculate_cache_age()
    profile_count = len(_consultant_profiles_cache) if _consultant_profiles_cache else 0

    # Check database connectivity
    db_status = "available" if get_supabase_client() else "unavailable"

    status = {
        "status": "healthy" if cache_valid and profile_count >= 12 else "degraded",
        "cache": {
            "valid": cache_valid,
            "age_minutes": cache_age,
            "profiles_cached": profile_count,
            "expected_profiles": 12,
        },
        "database": {"status": db_status, "supabase_available": SUPABASE_AVAILABLE},
        "timestamp": datetime.now().isoformat(),
    }

    return status
