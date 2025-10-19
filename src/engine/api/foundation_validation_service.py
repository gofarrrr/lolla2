"""
Foundation Validation Service
============================

Operation Chimera Phase 3 - Foundation Service Extraction

Validation service implementing business rules and constraints for the Foundation API.
Extracted from enhanced_foundation.py to separate validation concerns from routing logic.

Key Responsibilities:
- Engagement creation and update validation
- Cognitive analysis request validation  
- Mental model selection validation
- Access control and security validation
- ID format validation and sanitization
"""

import re
import uuid
import hashlib
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

from .foundation_contracts import (
    IFoundationValidationService,
    EngagementCreateRequest,
    CognitiveAnalysisRequest,
    FoundationServiceError,
    EngagementValidationError,
    CognitiveAnalysisError,
    MentalModelError,
)
from src.core.unified_context_stream import get_unified_context_stream, UnifiedContextStream


class FoundationValidationService(IFoundationValidationService):
    """
    Foundation Validation Service Implementation
    
    Provides centralized validation for all Foundation API operations,
    ensuring business rules are enforced consistently across the platform.
    """
    
    def __init__(self, context_stream: Optional[UnifiedContextStream] = None):
        """Initialize Foundation Validation Service"""
        self.context_stream = context_stream or get_unified_context_stream()
    
    async def validate_engagement_create(
        self,
        request: EngagementCreateRequest,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate engagement creation request
        
        Validates:
        - Problem statement format and content
        - Business context structure
        - User preferences validity
        - Compliance requirements
        """
        await self.context_stream.log_event(
            "FOUNDATION_ENGAGEMENT_VALIDATION_STARTED",
            {
                "request_type": "create_engagement",
                "problem_statement_length": len(request.problem_statement),
                "user_id": user_id,
                "has_business_context": bool(request.business_context),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "details": {}
        }
        
        try:
            # Validate problem statement
            problem_validation = await self._validate_problem_statement(request.problem_statement)
            if not problem_validation["is_valid"]:
                validation_result["is_valid"] = False
                validation_result["errors"].extend(problem_validation["errors"])
            
            # Validate business context structure
            context_validation = await self._validate_business_context(request.business_context)
            if not context_validation["is_valid"]:
                validation_result["warnings"].extend(context_validation["warnings"])
            
            # Validate user preferences
            prefs_validation = await self._validate_user_preferences(request.user_preferences)
            if not prefs_validation["is_valid"]:
                validation_result["warnings"].extend(prefs_validation["warnings"])
            
            # Validate compliance requirements
            compliance_validation = await self._validate_compliance_requirements(
                request.compliance_requirements
            )
            if not compliance_validation["is_valid"]:
                validation_result["errors"].extend(compliance_validation["errors"])
                validation_result["is_valid"] = False
            
            # Add validation details
            validation_result["details"] = {
                "problem_statement_length": len(request.problem_statement),
                "context_keys_count": len(request.business_context.keys()),
                "preferences_count": len(request.user_preferences.keys()),
                "compliance_rules_count": len(request.compliance_requirements.keys()),
                "estimated_complexity": self._estimate_problem_complexity(request.problem_statement)
            }
            
            await self.context_stream.log_event(
                "FOUNDATION_ENGAGEMENT_VALIDATION_COMPLETED",
                {
                    "validation_result": validation_result["is_valid"],
                    "errors_count": len(validation_result["errors"]),
                    "warnings_count": len(validation_result["warnings"]),
                    "complexity_estimate": validation_result["details"]["estimated_complexity"]
                }
            )
            
            return validation_result
        
        except Exception as e:
            await self.context_stream.log_event(
                "FOUNDATION_ENGAGEMENT_VALIDATION_ERROR",
                {
                    "error": str(e),
                    "error_type": e.__class__.__name__,
                    "request_details": {
                        "problem_length": len(request.problem_statement),
                        "context_provided": bool(request.business_context)
                    }
                }
            )
            raise EngagementValidationError(
                f"Engagement validation failed: {str(e)}",
                code="ENGAGEMENT_VALIDATION_ERROR",
                details={"original_error": str(e)}
            )
    
    async def validate_cognitive_analysis_request(
        self,
        request: CognitiveAnalysisRequest,
        engagement_id: str
    ) -> Dict[str, Any]:
        """
        Validate cognitive analysis request
        
        Validates:
        - Engagement ID format and existence check readiness
        - Rigor level compliance
        - Model selection validity
        - Analysis preferences structure
        """
        await self.context_stream.log_event(
            "FOUNDATION_ANALYSIS_VALIDATION_STARTED",
            {
                "engagement_id": engagement_id,
                "rigor_level": request.rigor_level,
                "forced_models": request.force_model_selection,
                "create_transparency": request.create_transparency_layers,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "details": {}
        }
        
        try:
            # Validate engagement ID format
            id_validation = await self._validate_engagement_id_format(engagement_id)
            if not id_validation["is_valid"]:
                validation_result["is_valid"] = False
                validation_result["errors"].extend(id_validation["errors"])
            
            # Validate rigor level
            rigor_validation = await self._validate_rigor_level(request.rigor_level)
            if not rigor_validation["is_valid"]:
                validation_result["is_valid"] = False
                validation_result["errors"].extend(rigor_validation["errors"])
            
            # Validate forced model selection if provided
            if request.force_model_selection:
                model_validation = await self._validate_model_selection(request.force_model_selection)
                if not model_validation["is_valid"]:
                    validation_result["warnings"].extend(model_validation["warnings"])
            
            # Validate analysis preferences structure
            prefs_validation = await self._validate_analysis_preferences(request.analysis_preferences)
            if not prefs_validation["is_valid"]:
                validation_result["warnings"].extend(prefs_validation["warnings"])
            
            # Add validation details
            validation_result["details"] = {
                "engagement_id_type": id_validation.get("id_type", "unknown"),
                "rigor_level_valid": rigor_validation["is_valid"],
                "forced_models_count": len(request.force_model_selection or []),
                "analysis_preferences_count": len(request.analysis_preferences.keys()),
                "transparency_requested": request.create_transparency_layers
            }
            
            await self.context_stream.log_event(
                "FOUNDATION_ANALYSIS_VALIDATION_COMPLETED",
                {
                    "validation_result": validation_result["is_valid"],
                    "errors_count": len(validation_result["errors"]),
                    "warnings_count": len(validation_result["warnings"]),
                    "rigor_level": request.rigor_level
                }
            )
            
            return validation_result
        
        except Exception as e:
            await self.context_stream.log_event(
                "FOUNDATION_ANALYSIS_VALIDATION_ERROR",
                {
                    "error": str(e),
                    "error_type": e.__class__.__name__,
                    "engagement_id": engagement_id,
                    "rigor_level": request.rigor_level
                }
            )
            raise CognitiveAnalysisError(
                f"Analysis validation failed: {str(e)}",
                code="ANALYSIS_VALIDATION_ERROR",
                details={"engagement_id": engagement_id, "original_error": str(e)}
            )
    
    async def validate_engagement_access(
        self,
        engagement_id: str,
        user_id: Optional[str] = None,
        action: str = "read"
    ) -> Dict[str, Any]:
        """
        Validate user access to engagement for given action
        
        Validates:
        - Engagement ID format
        - User authentication status
        - Action permissions
        - Access control rules
        """
        await self.context_stream.log_event(
            "FOUNDATION_ACCESS_VALIDATION_STARTED",
            {
                "engagement_id": engagement_id,
                "user_id": user_id,
                "action": action,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "details": {}
        }
        
        try:
            # Validate engagement ID format
            id_validation = await self._validate_engagement_id_format(engagement_id)
            if not id_validation["is_valid"]:
                validation_result["is_valid"] = False
                validation_result["errors"].extend(id_validation["errors"])
            
            # Validate action type
            action_validation = await self._validate_action_type(action)
            if not action_validation["is_valid"]:
                validation_result["is_valid"] = False
                validation_result["errors"].extend(action_validation["errors"])
            
            # For demo mode, allow access without user authentication
            # In production, this would enforce strict access controls
            if user_id is None:
                validation_result["warnings"].append("Demo mode: Access granted without authentication")
            
            validation_result["details"] = {
                "engagement_id_type": id_validation.get("id_type", "unknown"),
                "action": action,
                "user_authenticated": user_id is not None,
                "access_level": "demo" if user_id is None else "authenticated"
            }
            
            await self.context_stream.log_event(
                "FOUNDATION_ACCESS_VALIDATION_COMPLETED",
                {
                    "validation_result": validation_result["is_valid"],
                    "access_granted": validation_result["is_valid"],
                    "action": action,
                    "user_id": user_id
                }
            )
            
            return validation_result
        
        except Exception as e:
            await self.context_stream.log_event(
                "FOUNDATION_ACCESS_VALIDATION_ERROR",
                {
                    "error": str(e),
                    "error_type": e.__class__.__name__,
                    "engagement_id": engagement_id,
                    "action": action
                }
            )
            raise FoundationServiceError(
                f"Access validation failed: {str(e)}",
                code="ACCESS_VALIDATION_ERROR",
                details={"engagement_id": engagement_id, "action": action}
            )
    
    async def validate_mental_model_selection(
        self,
        selected_models: List[str],
        problem_context: str
    ) -> Dict[str, Any]:
        """
        Validate mental model selection for given context
        
        Validates:
        - Model names format and existence
        - Model compatibility with problem type
        - Selection count limits
        - Model combination effectiveness
        """
        await self.context_stream.log_event(
            "FOUNDATION_MODEL_VALIDATION_STARTED",
            {
                "selected_models_count": len(selected_models),
                "problem_context_length": len(problem_context),
                "models": selected_models,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "details": {}
        }
        
        try:
            # Validate model count limits
            if len(selected_models) > 10:
                validation_result["warnings"].append(
                    f"High model count ({len(selected_models)}) may impact performance"
                )
            
            if len(selected_models) == 0:
                validation_result["warnings"].append("No models specified - will use automatic selection")
            
            # Validate individual model names
            invalid_models = []
            for model in selected_models:
                if not await self._is_valid_model_name(model):
                    invalid_models.append(model)
            
            if invalid_models:
                validation_result["errors"].append(
                    f"Invalid model names: {', '.join(invalid_models)}"
                )
                validation_result["is_valid"] = False
            
            # Validate model compatibility with problem context
            compatibility_score = await self._calculate_model_compatibility(
                selected_models, problem_context
            )
            
            if compatibility_score < 0.3:
                validation_result["warnings"].append(
                    "Low compatibility between selected models and problem context"
                )
            
            validation_result["details"] = {
                "models_count": len(selected_models),
                "invalid_models": invalid_models,
                "compatibility_score": compatibility_score,
                "problem_complexity": self._estimate_problem_complexity(problem_context)
            }
            
            await self.context_stream.log_event(
                "FOUNDATION_MODEL_VALIDATION_COMPLETED",
                {
                    "validation_result": validation_result["is_valid"],
                    "models_count": len(selected_models),
                    "compatibility_score": compatibility_score,
                    "invalid_models_count": len(invalid_models)
                }
            )
            
            return validation_result
        
        except Exception as e:
            await self.context_stream.log_event(
                "FOUNDATION_MODEL_VALIDATION_ERROR",
                {
                    "error": str(e),
                    "error_type": e.__class__.__name__,
                    "selected_models": selected_models
                }
            )
            raise MentalModelError(
                f"Model validation failed: {str(e)}",
                code="MODEL_VALIDATION_ERROR",
                details={"selected_models": selected_models}
            )
    
    async def sanitize_engagement_id(self, engagement_id: str) -> UUID:
        """
        Sanitize and convert engagement ID to UUID
        
        Handles:
        - Standard UUID format validation
        - Legacy query format conversion (query_*)
        - Deterministic UUID generation for legacy IDs
        """
        await self.context_stream.log_event(
            "FOUNDATION_ID_SANITIZATION_STARTED",
            {
                "original_id": engagement_id,
                "id_length": len(engagement_id),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        try:
            # Try parsing as standard UUID first
            try:
                sanitized_uuid = UUID(engagement_id)
                await self.context_stream.log_event(
                    "FOUNDATION_ID_SANITIZATION_COMPLETED",
                    {
                        "original_id": engagement_id,
                        "sanitized_uuid": str(sanitized_uuid),
                        "conversion_type": "direct_uuid"
                    }
                )
                return sanitized_uuid
            
            except ValueError:
                # Handle legacy query format like query_1756965138054
                if engagement_id.startswith("query_"):
                    # Generate deterministic UUID from the ID
                    hash_object = hashlib.md5(engagement_id.encode())
                    hex_dig = hash_object.hexdigest()
                    
                    # Convert to UUID format
                    uuid_string = f"{hex_dig[:8]}-{hex_dig[8:12]}-{hex_dig[12:16]}-{hex_dig[16:20]}-{hex_dig[20:32]}"
                    sanitized_uuid = UUID(uuid_string)
                    
                    await self.context_stream.log_event(
                        "FOUNDATION_ID_SANITIZATION_COMPLETED",
                        {
                            "original_id": engagement_id,
                            "sanitized_uuid": str(sanitized_uuid),
                            "conversion_type": "legacy_query_format"
                        }
                    )
                    return sanitized_uuid
                
                # For other invalid formats, raise validation error
                await self.context_stream.log_event(
                    "FOUNDATION_ID_SANITIZATION_ERROR",
                    {
                        "original_id": engagement_id,
                        "error": "Invalid ID format",
                        "supported_formats": ["uuid", "query_*"]
                    }
                )
                raise EngagementValidationError(
                    f"Invalid engagement ID format: {engagement_id}",
                    code="INVALID_ENGAGEMENT_ID",
                    details={"provided_id": engagement_id}
                )
        
        except Exception as e:
            await self.context_stream.log_event(
                "FOUNDATION_ID_SANITIZATION_ERROR",
                {
                    "original_id": engagement_id,
                    "error": str(e),
                    "error_type": e.__class__.__name__
                }
            )
            raise EngagementValidationError(
                f"ID sanitization failed: {str(e)}",
                code="ID_SANITIZATION_ERROR",
                details={"provided_id": engagement_id}
            )
    
    # Private validation helper methods
    
    async def _validate_problem_statement(self, problem_statement: str) -> Dict[str, Any]:
        """Validate problem statement format and content"""
        result = {"is_valid": True, "errors": [], "warnings": []}
        
        # Check length constraints
        if len(problem_statement) < 10:
            result["is_valid"] = False
            result["errors"].append("Problem statement too short (minimum 10 characters)")
        
        if len(problem_statement) > 5000:
            result["is_valid"] = False
            result["errors"].append("Problem statement too long (maximum 5000 characters)")
        
        # Check for empty or whitespace-only content
        if not problem_statement.strip():
            result["is_valid"] = False
            result["errors"].append("Problem statement cannot be empty")
        
        # Check for potentially problematic content
        if len(problem_statement.split()) < 3:
            result["warnings"].append("Very short problem statement may limit analysis quality")
        
        return result
    
    async def _validate_business_context(self, business_context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate business context structure"""
        result = {"is_valid": True, "errors": [], "warnings": []}
        
        if not business_context:
            result["warnings"].append("No business context provided - analysis may be less targeted")
        
        # Check for recommended context fields
        recommended_fields = ["industry", "company_size", "timeline", "stakeholders"]
        missing_fields = [field for field in recommended_fields if field not in business_context]
        
        if missing_fields:
            result["warnings"].append(f"Recommended context fields missing: {', '.join(missing_fields)}")
        
        return result
    
    async def _validate_user_preferences(self, user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Validate user preferences structure"""
        result = {"is_valid": True, "errors": [], "warnings": []}
        
        # Check for valid preference keys
        valid_preference_keys = ["analysis_depth", "communication_style", "risk_tolerance", "time_horizon"]
        invalid_keys = [key for key in user_preferences.keys() if key not in valid_preference_keys]
        
        if invalid_keys:
            result["warnings"].append(f"Unknown preference keys: {', '.join(invalid_keys)}")
        
        return result
    
    async def _validate_compliance_requirements(self, compliance_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Validate compliance requirements"""
        result = {"is_valid": True, "errors": [], "warnings": []}
        
        # For now, accept any compliance requirements
        # In production, this would validate against known compliance frameworks
        if compliance_requirements:
            result["warnings"].append("Compliance requirements will be considered in analysis")
        
        return result
    
    async def _validate_engagement_id_format(self, engagement_id: str) -> Dict[str, Any]:
        """Validate engagement ID format"""
        result = {"is_valid": True, "errors": [], "id_type": "unknown"}
        
        # Check for UUID format
        try:
            UUID(engagement_id)
            result["id_type"] = "uuid"
            return result
        except ValueError:
            pass
        
        # Check for legacy query format
        if engagement_id.startswith("query_"):
            result["id_type"] = "legacy_query"
            return result
        
        # Invalid format
        result["is_valid"] = False
        result["errors"].append(f"Invalid engagement ID format: {engagement_id}")
        return result
    
    async def _validate_rigor_level(self, rigor_level: str) -> Dict[str, Any]:
        """Validate rigor level"""
        result = {"is_valid": True, "errors": []}
        
        valid_levels = ["L0", "L1", "L2", "L3"]
        if rigor_level not in valid_levels:
            result["is_valid"] = False
            result["errors"].append(f"Invalid rigor level: {rigor_level}. Must be one of {valid_levels}")
        
        return result
    
    async def _validate_model_selection(self, models: List[str]) -> Dict[str, Any]:
        """Validate forced model selection"""
        result = {"is_valid": True, "errors": [], "warnings": []}
        
        if len(models) > 10:
            result["warnings"].append("Large number of forced models may impact performance")
        
        # In production, this would validate against available models
        result["warnings"].append("Forced model selection will override automatic selection")
        
        return result
    
    async def _validate_analysis_preferences(self, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Validate analysis preferences structure"""
        result = {"is_valid": True, "errors": [], "warnings": []}
        
        # Check for valid preference structure
        if preferences and not isinstance(preferences, dict):
            result["warnings"].append("Analysis preferences should be a dictionary")
        
        return result
    
    async def _validate_action_type(self, action: str) -> Dict[str, Any]:
        """Validate action type for access control"""
        result = {"is_valid": True, "errors": []}
        
        valid_actions = ["read", "write", "delete", "analyze", "share"]
        if action not in valid_actions:
            result["is_valid"] = False
            result["errors"].append(f"Invalid action: {action}. Must be one of {valid_actions}")
        
        return result
    
    async def _is_valid_model_name(self, model_name: str) -> bool:
        """Check if model name is valid format"""
        # Basic validation - in production would check against model registry
        return bool(re.match(r'^[a-zA-Z0-9_-]+$', model_name)) and len(model_name) > 0
    
    async def _calculate_model_compatibility(self, models: List[str], context: str) -> float:
        """Calculate compatibility score between models and problem context"""
        # Simplified compatibility calculation
        # In production, this would use sophisticated matching algorithms
        if not models or not context:
            return 0.5
        
        # Simple heuristic based on model count and context length
        base_score = min(len(models) / 5.0, 1.0)  # Up to 5 models is optimal
        context_factor = min(len(context) / 100.0, 1.0)  # Longer context helps
        
        return (base_score + context_factor) / 2.0
    
    def _estimate_problem_complexity(self, problem_text: str) -> str:
        """Estimate problem complexity based on text analysis"""
        word_count = len(problem_text.split())
        
        if word_count < 10:
            return "low"
        elif word_count < 50:
            return "medium"
        elif word_count < 200:
            return "high"
        else:
            return "very_high"
