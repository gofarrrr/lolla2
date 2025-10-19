"""
Projects API - CRUD Operations
===============================

Project CRUD endpoints for creating, reading, updating, and deleting projects.
"""

import logging
from datetime import datetime, timezone
from typing import List, Dict
from fastapi import APIRouter, HTTPException, Query, Depends
from supabase import Client

from .models import ProjectCreateRequest, ProjectUpdateRequest, ProjectResponse
from .dependencies import get_supabase, validate_organization_access
from src.core.unified_context_stream import get_unified_context_stream

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()


@router.post("/", response_model=ProjectResponse)
async def create_project(
    request: ProjectCreateRequest,
    supabase: Client = Depends(get_supabase),
    org_id: str = Depends(validate_organization_access),
) -> ProjectResponse:
    """
    Create a new project with default settings and initialize knowledge base.
    This establishes the foundation for project-scoped analyses and RAG.
    """
    try:
        # Insert project into database
        project_data = {
            "organization_id": request.organization_id,
            "name": request.name,
            "description": request.description,
            "settings": request.settings,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        result = supabase.table("projects").insert(project_data).execute()

        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create project")

        project = result.data[0]

        # Initialize audit trail
        audit_stream = get_unified_context_stream()
        await audit_stream.log_event(
            "PROJECT_CREATED",
            {
                "project_id": project["project_id"],
                "project_name": project["name"],
                "organization_id": project["organization_id"],
                "settings": project["settings"],
            },
            metadata={"api_version": "v2", "operation": "project_creation"},
        )

        logger.info(f"✅ Project created: {project['name']} ({project['project_id']})")

        return ProjectResponse(
            **project, recent_analyses_count=0, rag_health_status="healthy"
        )

    except Exception as e:
        logger.error(f"❌ Project creation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Project creation failed: {str(e)}"
        )


@router.get("/", response_model=List[ProjectResponse])
async def list_projects(
    organization_id: str = Query(..., description="Organization UUID"),
    status: str = Query("active", description="Project status filter"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of projects"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    supabase: Client = Depends(get_supabase),
) -> List[ProjectResponse]:
    """
    List projects for an organization with dashboard statistics.
    Returns projects sorted by last accessed date.
    """
    try:
        # Query projects with dashboard statistics
        query = supabase.table("v2_project_dashboard").select("*")

        if organization_id:
            query = query.eq("organization_id", organization_id)
        if status != "all":
            query = query.eq("status", status)

        result = (
            query.order("last_accessed_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )

        if not result.data:
            return []

        projects = [ProjectResponse(**project) for project in result.data]
        logger.info(
            f"✅ Listed {len(projects)} projects for organization {organization_id}"
        )

        return projects

    except Exception as e:
        logger.error(f"❌ Project listing failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list projects: {str(e)}"
        )


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str, supabase: Client = Depends(get_supabase)
) -> ProjectResponse:
    """
    Get detailed project information including statistics and health status.
    """
    try:
        # Get project with statistics from dashboard view
        result = (
            supabase.table("v2_project_dashboard")
            .select("*")
            .eq("project_id", project_id)
            .execute()
        )

        if not result.data:
            raise HTTPException(status_code=404, detail="Project not found")

        project = result.data[0]

        # Update last accessed timestamp
        supabase.table("projects").update(
            {"last_accessed_at": datetime.now(timezone.utc).isoformat()}
        ).eq("project_id", project_id).execute()

        logger.info(f"✅ Retrieved project: {project['name']} ({project_id})")

        return ProjectResponse(**project)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Project retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get project: {str(e)}")


@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    request: ProjectUpdateRequest,
    supabase: Client = Depends(get_supabase),
) -> ProjectResponse:
    """
    Update project information and settings.
    """
    try:
        # Build update data
        update_data = {"updated_at": datetime.now(timezone.utc).isoformat()}

        if request.name is not None:
            update_data["name"] = request.name
        if request.description is not None:
            update_data["description"] = request.description
        if request.settings is not None:
            update_data["settings"] = request.settings
        if request.status is not None:
            update_data["status"] = request.status

        # Update project
        result = (
            supabase.table("projects")
            .update(update_data)
            .eq("project_id", project_id)
            .execute()
        )

        if not result.data:
            raise HTTPException(status_code=404, detail="Project not found")

        # Get updated project with statistics
        updated = (
            supabase.table("v2_project_dashboard")
            .select("*")
            .eq("project_id", project_id)
            .execute()
        )
        project = updated.data[0]

        logger.info(f"✅ Project updated: {project['name']} ({project_id})")

        return ProjectResponse(**project)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Project update failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update project: {str(e)}"
        )


@router.delete("/{project_id}")
async def delete_project(
    project_id: str,
    hard_delete: bool = Query(
        False, description="Permanently delete (vs mark as deleted)"
    ),
    supabase: Client = Depends(get_supabase),
) -> Dict[str, str]:
    """
    Delete or archive a project. Hard delete removes all associated data.
    """
    try:
        if hard_delete:
            # Hard delete - remove all related data
            # Note: CASCADE deletes will handle rag_documents and rag_text_chunks
            result = (
                supabase.table("projects")
                .delete()
                .eq("project_id", project_id)
                .execute()
            )
            action = "permanently deleted"
        else:
            # Soft delete - mark as deleted
            result = (
                supabase.table("projects")
                .update(
                    {
                        "status": "deleted",
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                .eq("project_id", project_id)
                .execute()
            )
            action = "marked as deleted"

        if not result.data:
            raise HTTPException(status_code=404, detail="Project not found")

        logger.info(f"✅ Project {action}: {project_id}")

        return {"message": f"Project {action} successfully", "project_id": project_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Project deletion failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete project: {str(e)}"
        )
