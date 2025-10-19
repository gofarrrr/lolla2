"""
Project Validation Service
==========================

Business rules validation service for project operations. Centralizes all
validation logic that was scattered throughout the monolithic project service.

Responsibilities:
- Project creation/update validation rules
- File upload constraints and security checks
- Analysis request validation
- User access permission validation
- Business rule enforcement

This service contains pure business logic without any database operations
or external dependencies beyond the validation rules themselves.
"""

import logging
import mimetypes
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from .specialized_contracts import (
    IProjectValidator,
    ValidationResult,
    ProjectValidationError,
)
from .contracts import (
    ProjectCreateRequest,
    ProjectUpdateRequest,
    AnalysisCreateRequest,
    MentalModelIngestionRequest,
)


class ProjectValidationService(IProjectValidator):
    """
    Centralized validation service for project operations
    
    Implements all business rules and constraints for project-related
    operations without any external dependencies or side effects.
    """
    
    def __init__(self, context_stream=None, strict_mode=True, enable_security_checks=True):
        """Initialize validation service"""
        self.logger = logging.getLogger(__name__)
        self.context_stream = context_stream
        self.strict_mode = strict_mode
        self.enable_security_checks = enable_security_checks
        self.logger.debug("🏗️ ProjectValidationService initialized")
        
        # Configuration constants
        self.MAX_PROJECT_NAME_LENGTH = 255
        self.MAX_DESCRIPTION_LENGTH = 2000
        self.MAX_FILE_SIZE_MB = 10
        self.ALLOWED_FILE_TYPES = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
            "text/markdown",
            "application/msword",
        ]
        self.ALLOWED_MENTAL_MODEL_EXTENSIONS = [".yaml", ".yml", ".json", ".md", ".txt"]
        self.VALID_PROJECT_STATUSES = ["active", "archived", "deleted"]
        self.VALID_ENGAGEMENT_TYPES = [
            "deep_dive", "quick_analysis", "strategic_review", 
            "market_research", "competitive_analysis", "feasibility_study"
        ]
        self.VALID_PRIORITIES = ["low", "normal", "high", "urgent"]
    
    async def validate_create_request(self, request: ProjectCreateRequest) -> ValidationResult:
        """Validate project creation request"""
        errors = []
        warnings = []
        details = {}
        
        try:
            self.logger.debug(f"Validating project creation request: {request.name}")
            
            # Name validation
            if not request.name or not request.name.strip():
                errors.append("Project name is required")
            elif len(request.name) > self.MAX_PROJECT_NAME_LENGTH:
                errors.append(f"Project name exceeds maximum length of {self.MAX_PROJECT_NAME_LENGTH} characters")
            elif len(request.name.strip()) < 3:
                errors.append("Project name must be at least 3 characters long")
            
            # Description validation
            if request.description and len(request.description) > self.MAX_DESCRIPTION_LENGTH:
                errors.append(f"Description exceeds maximum length of {self.MAX_DESCRIPTION_LENGTH} characters")
            
            # Organization ID validation
            if not request.organization_id or not request.organization_id.strip():
                errors.append("Organization ID is required")
            
            # Settings validation
            if request.settings:
                settings_validation = self._validate_project_settings(request.settings)
                errors.extend(settings_validation.get("errors", []))
                warnings.extend(settings_validation.get("warnings", []))
            
            # Security checks
            security_check = self._perform_security_validation(request.name, request.description)
            if security_check.get("suspicious"):
                warnings.append("Project contains potentially suspicious content - manual review recommended")
                details["security_flags"] = security_check.get("flags", [])
            
            self.logger.info(f"✅ Project creation validation completed: {len(errors)} errors, {len(warnings)} warnings")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                details=details
            )
            
        except Exception as e:
            self.logger.error(f"❌ Project creation validation failed: {e}")
            raise ProjectValidationError(
                [f"Validation service error: {str(e)}"],
                {"request": request.dict(), "original_error": str(e)}
            ) from e
    
    async def validate_update_request(self, project_id: str, request: ProjectUpdateRequest) -> ValidationResult:
        """Validate project update request"""
        errors = []
        warnings = []
        details = {}
        
        try:
            self.logger.debug(f"Validating project update request: {project_id}")
            
            # Project ID validation
            if not project_id or not project_id.strip():
                errors.append("Project ID is required")
            
            # Name validation (if being updated)
            if request.name is not None:
                if not request.name.strip():
                    errors.append("Project name cannot be empty")
                elif len(request.name) > self.MAX_PROJECT_NAME_LENGTH:
                    errors.append(f"Project name exceeds maximum length of {self.MAX_PROJECT_NAME_LENGTH} characters")
                elif len(request.name.strip()) < 3:
                    errors.append("Project name must be at least 3 characters long")
            
            # Description validation (if being updated)
            if request.description is not None and len(request.description) > self.MAX_DESCRIPTION_LENGTH:
                errors.append(f"Description exceeds maximum length of {self.MAX_DESCRIPTION_LENGTH} characters")
            
            # Status validation (if being updated)
            if request.status is not None and request.status not in self.VALID_PROJECT_STATUSES:
                errors.append(f"Invalid status. Must be one of: {', '.join(self.VALID_PROJECT_STATUSES)}")
            
            # Settings validation (if being updated)
            if request.settings is not None:
                settings_validation = self._validate_project_settings(request.settings)
                errors.extend(settings_validation.get("errors", []))
                warnings.extend(settings_validation.get("warnings", []))
            
            # Check if any actual changes are being made
            if all(getattr(request, field) is None for field in ["name", "description", "settings", "status"]):
                warnings.append("No changes detected in update request")
            
            self.logger.info(f"✅ Project update validation completed: {len(errors)} errors, {len(warnings)} warnings")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                details=details
            )
            
        except Exception as e:
            self.logger.error(f"❌ Project update validation failed: {e}")
            raise ProjectValidationError(
                [f"Validation service error: {str(e)}"],
                {"project_id": project_id, "request": request.dict(), "original_error": str(e)}
            ) from e
    
    async def validate_file_upload(self, file_info: Dict[str, Any]) -> ValidationResult:
        """Validate uploaded file constraints and security"""
        errors = []
        warnings = []
        details = {}
        
        try:
            self.logger.debug(f"Validating file upload: {file_info.get('filename')}")
            
            filename = file_info.get("filename")
            content_type = file_info.get("content_type")
            size = file_info.get("size")
            
            # Filename validation
            if not filename:
                errors.append("Filename is required")
            else:
                # Check for potentially dangerous filenames
                if any(char in filename for char in ["<", ">", ":", "\"", "|", "?", "*"]):
                    errors.append("Filename contains invalid characters")
                
                # Check for overly long filenames
                if len(filename) > 255:
                    errors.append("Filename is too long (maximum 255 characters)")
                
                # Check for suspicious extensions
                file_ext = Path(filename).suffix.lower()
                if file_ext in [".exe", ".bat", ".cmd", ".scr", ".js", ".vbs"]:
                    errors.append(f"File type '{file_ext}' is not allowed for security reasons")
            
            # Content type validation
            if not content_type:
                warnings.append("Content type not provided - attempting to detect from filename")
                if filename:
                    detected_type, _ = mimetypes.guess_type(filename)
                    content_type = detected_type
                    details["detected_content_type"] = detected_type
            
            if content_type and content_type not in self.ALLOWED_FILE_TYPES:
                errors.append(f"File type '{content_type}' is not supported. Allowed types: {', '.join(self.ALLOWED_FILE_TYPES)}")
            
            # File size validation
            if size is None:
                warnings.append("File size not provided - cannot validate size limits")
            elif size > self.MAX_FILE_SIZE_MB * 1024 * 1024:
                errors.append(f"File size ({size / 1024 / 1024:.1f}MB) exceeds maximum allowed size of {self.MAX_FILE_SIZE_MB}MB")
            elif size == 0:
                errors.append("File appears to be empty")
            
            # Security checks
            if filename:
                security_check = self._perform_file_security_check(filename)
                if security_check.get("suspicious"):
                    warnings.append("File shows potential security concerns - manual review recommended")
                    details["security_flags"] = security_check.get("flags", [])
            
            self.logger.info(f"✅ File upload validation completed: {len(errors)} errors, {len(warnings)} warnings")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                details=details
            )
            
        except Exception as e:
            self.logger.error(f"❌ File upload validation failed: {e}")
            raise ProjectValidationError(
                [f"File validation error: {str(e)}"],
                {"file_info": file_info, "original_error": str(e)}
            ) from e
    
    async def validate_analysis_request(self, request: AnalysisCreateRequest) -> ValidationResult:
        """Validate analysis creation request"""
        errors = []
        warnings = []
        details = {}
        
        try:
            self.logger.debug(f"Validating analysis request for project: {request.project_id}")
            
            # Project ID validation
            if not request.project_id or not request.project_id.strip():
                errors.append("Project ID is required")
            
            # Organization ID validation
            if not request.organization_id or not request.organization_id.strip():
                errors.append("Organization ID is required")
            
            # User query validation
            if not request.user_query or not request.user_query.strip():
                errors.append("User query is required")
            elif len(request.user_query.strip()) < 10:
                warnings.append("Query is very short - consider providing more detail for better analysis")
            elif len(request.user_query) > 5000:
                errors.append("Query is too long (maximum 5000 characters)")
            
            # Engagement type validation
            if request.engagement_type not in self.VALID_ENGAGEMENT_TYPES:
                errors.append(f"Invalid engagement type. Must be one of: {', '.join(self.VALID_ENGAGEMENT_TYPES)}")
            
            # Priority validation
            if request.priority not in self.VALID_PRIORITIES:
                errors.append(f"Invalid priority. Must be one of: {', '.join(self.VALID_PRIORITIES)}")
            
            # Consultants validation
            if request.consultants and len(request.consultants) > 10:
                warnings.append("Large number of consultants selected - this may increase analysis time and cost")
            
            # Context merge validation
            if request.merge_project_context:
                details["context_merge_enabled"] = True
                # Note: actual context availability would be checked by the orchestration service
            
            # Security checks for query content
            security_check = self._perform_security_validation(request.user_query)
            if security_check.get("suspicious"):
                warnings.append("Query contains potentially sensitive content - review recommended")
                details["security_flags"] = security_check.get("flags", [])
            
            self.logger.info(f"✅ Analysis request validation completed: {len(errors)} errors, {len(warnings)} warnings")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                details=details
            )
            
        except Exception as e:
            self.logger.error(f"❌ Analysis request validation failed: {e}")
            raise ProjectValidationError(
                [f"Analysis validation error: {str(e)}"],
                {"request": request.dict(), "original_error": str(e)}
            ) from e
    
    async def validate_project_access(self, project_id: str, user_id: Optional[str]) -> ValidationResult:
        """Validate user access to project (basic validation - actual auth is handled elsewhere)"""
        errors = []
        warnings = []
        details = {}
        
        try:
            self.logger.debug(f"Validating project access: {project_id}")
            
            # Project ID validation
            if not project_id or not project_id.strip():
                errors.append("Project ID is required")
            
            # User ID validation (if provided)
            if user_id is not None and not user_id.strip():
                warnings.append("User ID provided but empty - using anonymous access")
            
            # Note: Actual authorization logic would be implemented in a separate auth service
            # This validation only checks basic parameter validity
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                details=details
            )
            
        except Exception as e:
            self.logger.error(f"❌ Project access validation failed: {e}")
            raise ProjectValidationError(
                [f"Access validation error: {str(e)}"],
                {"project_id": project_id, "user_id": user_id, "original_error": str(e)}
            ) from e
    
    async def validate_mental_model_request(self, request: MentalModelIngestionRequest) -> ValidationResult:
        """Validate mental model ingestion request"""
        errors = []
        warnings = []
        details = {}
        
        try:
            self.logger.debug(f"Validating mental model ingestion request")
            
            # Directory path validation (if provided)
            if hasattr(request, 'directory_path') and request.directory_path:
                directory_path = Path(request.directory_path)
                if not directory_path.exists():
                    errors.append(f"Directory does not exist: {request.directory_path}")
                elif not directory_path.is_dir():
                    errors.append(f"Path is not a directory: {request.directory_path}")
                else:
                    # Check for mental model files
                    file_pattern = getattr(request, 'file_pattern', '*.yaml')
                    matching_files = list(directory_path.glob(file_pattern))
                    if not matching_files:
                        warnings.append(f"No files found matching pattern '{file_pattern}' in directory")
                    else:
                        details["files_found"] = len(matching_files)
                        
                        # Validate file extensions
                        for file_path in matching_files:
                            if file_path.suffix.lower() not in self.ALLOWED_MENTAL_MODEL_EXTENSIONS:
                                warnings.append(f"Potentially unsupported file type: {file_path.name}")
            
            # Project ID validation (if provided)
            if hasattr(request, 'project_id') and request.project_id:
                if not request.project_id.strip():
                    errors.append("Project ID cannot be empty")
            
            # Organization ID validation
            if hasattr(request, 'organization_id') and not request.organization_id:
                errors.append("Organization ID is required")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                details=details
            )
            
        except Exception as e:
            self.logger.error(f"❌ Mental model request validation failed: {e}")
            raise ProjectValidationError(
                [f"Mental model validation error: {str(e)}"],
                {"request": request.dict() if hasattr(request, 'dict') else str(request), "original_error": str(e)}
            ) from e
    
    # ============================================================
    # Private Helper Methods
    # ============================================================
    
    def _validate_project_settings(self, settings: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate project settings structure and values"""
        errors = []
        warnings = []
        
        # Required settings structure
        expected_keys = [
            "context_merging_enabled",
            "auto_rag_indexing", 
            "retention_policy",
            "privacy_settings"
        ]
        
        for key in expected_keys:
            if key not in settings:
                warnings.append(f"Missing recommended setting: {key}")
        
        # Validate specific settings
        if "retention_policy" in settings:
            valid_policies = ["standard", "extended", "minimal", "custom"]
            if settings["retention_policy"] not in valid_policies:
                errors.append(f"Invalid retention policy. Must be one of: {', '.join(valid_policies)}")
        
        if "privacy_settings" in settings:
            privacy = settings["privacy_settings"]
            if not isinstance(privacy, dict):
                errors.append("Privacy settings must be a dictionary")
            else:
                valid_classifications = ["public", "internal", "confidential", "restricted"]
                if "data_classification" in privacy:
                    if privacy["data_classification"] not in valid_classifications:
                        errors.append(f"Invalid data classification. Must be one of: {', '.join(valid_classifications)}")
        
        return {"errors": errors, "warnings": warnings}
    
    def _perform_security_validation(self, *texts) -> Dict[str, Any]:
        """Perform basic security validation on text content"""
        suspicious_patterns = [
            # Basic injection patterns
            "<script", "javascript:", "data:text/html",
            # Command injection patterns  
            "rm -rf", "del /", "format c:",
            # SQL injection patterns
            "union select", "drop table", "'; drop",
            # Path traversal
            "../../../", "..\\..\\..\\",
        ]
        
        flags = []
        combined_text = " ".join(str(text) for text in texts if text).lower()
        
        for pattern in suspicious_patterns:
            if pattern in combined_text:
                flags.append(f"Suspicious pattern detected: {pattern}")
        
        return {
            "suspicious": len(flags) > 0,
            "flags": flags
        }
    
    def _perform_file_security_check(self, filename: str) -> Dict[str, Any]:
        """Perform security checks on filename"""
        flags = []
        
        # Check for double extensions (common malware technique)
        if filename.count('.') > 2:
            flags.append("Multiple file extensions detected")
        
        # Check for suspicious filename patterns
        suspicious_names = ["autorun", "desktop.ini", "thumbs.db", ".htaccess"]
        if any(sus_name in filename.lower() for sus_name in suspicious_names):
            flags.append("Suspicious filename pattern")
        
        # Check for very long filenames (potential buffer overflow)
        if len(filename) > 200:
            flags.append("Unusually long filename")
        
        return {
            "suspicious": len(flags) > 0,
            "flags": flags
        }


# ============================================================
# Factory Function
# ============================================================

def get_project_validator() -> IProjectValidator:
    """Factory function for dependency injection"""
    return ProjectValidationService()