"""
Feature Flag Decorators for METIS API Endpoints
===============================================

Provides decorators to wrap API endpoints with feature flag checks,
enabling gradual rollout and A/B testing of the Red Team Council features.

Usage:
    @feature_flag_required(FeatureFlag.ENABLE_PARALLEL_VALIDATION)
    async def red_team_endpoint(...):
        ...

    @ab_test_assignment(FeatureFlag.ENABLE_ENHANCED_ARBITRATION)
    async def arbitration_endpoint(...):
        # Endpoint gets user's A/B test group assignment
        ...
"""

import functools
from typing import Callable, Optional, Dict, Any
from uuid import UUID

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

from src.engine.adapters.core.feature_flags import (
    FeatureFlag,
    get_feature_flag_service,
    get_experiment_group,
)
from src.engine.adapters.core.structured_logging import get_logger

logger = get_logger(__name__, component="feature_flag_decorators")


def feature_flag_required(
    flag: FeatureFlag,
    fallback_response: Optional[Dict[str, Any]] = None,
    fallback_status_code: int = 404,
):
    """
    Decorator that requires a feature flag to be enabled for the endpoint.

    Args:
        flag: The feature flag that must be enabled
        fallback_response: Optional custom response when flag is disabled
        fallback_status_code: HTTP status code for fallback (default: 404)

    Example:
        @feature_flag_required(FeatureFlag.ENABLE_PARALLEL_VALIDATION)
        async def red_team_validation(engagement_id: str):
            # This endpoint only works when the flag is enabled
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from args/kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if request is None:
                # Try to find request in kwargs
                request = kwargs.get("request")

            # Get feature flag service
            flag_service = get_feature_flag_service()

            # Extract user context if available
            user_id = None
            org_id = None

            try:
                # Try to get user context from request headers or auth
                if request:
                    user_id = request.headers.get("X-User-ID")
                    org_id = request.headers.get("X-Org-ID")

                    # Convert to UUID if provided
                    if user_id:
                        user_id = UUID(user_id)
                    if org_id:
                        org_id = UUID(org_id)

            except (ValueError, TypeError):
                # Invalid UUID format, continue without user context
                user_id = None
                org_id = None

            # Check if feature flag is enabled
            is_enabled = await flag_service.is_enabled(
                flag=flag,
                user_id=user_id,
                org_id=org_id,
                request_context={"endpoint": func.__name__} if request else {},
            )

            if not is_enabled:
                logger.info(
                    "feature_flag_blocked_request",
                    flag=flag.value,
                    endpoint=func.__name__,
                    user_id=str(user_id) if user_id else None,
                    org_id=str(org_id) if org_id else None,
                )

                # Return fallback response
                if fallback_response is not None:
                    return JSONResponse(
                        status_code=fallback_status_code, content=fallback_response
                    )
                else:
                    # Default fallback: feature not available
                    raise HTTPException(
                        status_code=fallback_status_code,
                        detail="Feature not available",
                    )

            # Feature flag is enabled, proceed with original function
            logger.debug(
                "feature_flag_allowed_request",
                flag=flag.value,
                endpoint=func.__name__,
                user_id=str(user_id) if user_id else None,
            )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def ab_test_assignment(flag: FeatureFlag, inject_group: bool = True):
    """
    Decorator that assigns users to A/B test groups and optionally injects
    the group assignment into the endpoint.

    Args:
        flag: The feature flag that controls the A/B test
        inject_group: Whether to inject the A/B test group into kwargs

    Example:
        @ab_test_assignment(FeatureFlag.ENABLE_ENHANCED_ARBITRATION)
        async def arbitration_endpoint(engagement_id: str, ab_test_group: ABTestGroup):
            if ab_test_group == ABTestGroup.TREATMENT:
                # Use enhanced arbitration
                ...
            else:
                # Use control group behavior
                ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request and user context
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if request is None:
                request = kwargs.get("request")

            user_id = None
            org_id = None

            try:
                if request:
                    user_id = request.headers.get("X-User-ID")
                    org_id = request.headers.get("X-Org-ID")

                    if user_id:
                        user_id = UUID(user_id)
                    if org_id:
                        org_id = UUID(org_id)

            except (ValueError, TypeError):
                user_id = None
                org_id = None

            # Get A/B test group assignment
            ab_test_group = get_experiment_group(
                flag=flag, user_id=user_id, org_id=org_id
            )

            logger.debug(
                "ab_test_assignment",
                flag=flag.value,
                endpoint=func.__name__,
                user_id=str(user_id) if user_id else None,
                ab_test_group=ab_test_group.value,
            )

            # Inject group assignment if requested
            if inject_group:
                kwargs["ab_test_group"] = ab_test_group

            # Add A/B test info to response headers if possible
            result = await func(*args, **kwargs)

            # If result is a Response object, add headers
            if hasattr(result, "headers"):
                result.headers["X-AB-Test-Group"] = ab_test_group.value
                result.headers["X-Feature-Flag"] = flag.value

            return result

        return wrapper

    return decorator


def feature_flag_context(flags: list[FeatureFlag]):
    """
    Decorator that injects feature flag context into the endpoint.

    Args:
        flags: List of feature flags to check and inject

    Example:
        @feature_flag_context([
            FeatureFlag.ENABLE_PARALLEL_VALIDATION,
            FeatureFlag.ENABLE_USER_GENERATED_CRITIQUES
        ])
        async def engagement_endpoint(engagement_id: str, feature_context: Dict[str, bool]):
            if feature_context['enable_parallel_validation']:
                # Use Red Team Council
                ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user context
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            user_id = None
            org_id = None

            try:
                if request:
                    user_id = request.headers.get("X-User-ID")
                    org_id = request.headers.get("X-Org-ID")

                    if user_id:
                        user_id = UUID(user_id)
                    if org_id:
                        org_id = UUID(org_id)

            except (ValueError, TypeError):
                user_id = None
                org_id = None

            # Get feature flag service
            flag_service = get_feature_flag_service()

            # Check all feature flags
            feature_context = {}
            for flag in flags:
                is_enabled = await flag_service.is_enabled(
                    flag=flag, user_id=user_id, org_id=org_id
                )

                # Convert flag enum to snake_case key
                key = flag.value.lower()
                feature_context[key] = is_enabled

            # Inject feature context
            kwargs["feature_context"] = feature_context

            logger.debug(
                "feature_flag_context_injected",
                endpoint=func.__name__,
                feature_context=feature_context,
                user_id=str(user_id) if user_id else None,
            )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_beta_access(beta_feature_name: str = "Red Team Council"):
    """
    Decorator that requires beta user access.

    Args:
        beta_feature_name: Name of the beta feature for error messages

    Example:
        @require_beta_access("Enhanced Arbitration")
        async def beta_endpoint(...):
            # Only beta users can access this
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user context
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            user_id = None

            try:
                if request:
                    user_id = request.headers.get("X-User-ID")
                    is_beta_user = (
                        request.headers.get("X-Beta-User", "false").lower() == "true"
                    )

                    if not is_beta_user:
                        logger.info(
                            "beta_access_denied",
                            endpoint=func.__name__,
                            user_id=user_id,
                            beta_feature=beta_feature_name,
                        )

                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail=f"Beta access required for {beta_feature_name}",
                        )

            except HTTPException:
                raise
            except Exception:
                # If we can't determine beta status, deny access
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Unable to verify beta access",
                )

            logger.debug(
                "beta_access_granted",
                endpoint=func.__name__,
                user_id=user_id,
                beta_feature=beta_feature_name,
            )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


# Convenience decorators for common Red Team Council features
red_team_required = functools.partial(
    feature_flag_required,
    FeatureFlag.ENABLE_PARALLEL_VALIDATION,
    {"error": "Red Team Council feature not available", "code": "FEATURE_DISABLED"},
)

enhanced_arbitration_ab_test = functools.partial(
    ab_test_assignment, FeatureFlag.ENABLE_ENHANCED_ARBITRATION
)

user_critiques_beta = functools.partial(require_beta_access, "User-Generated Critiques")