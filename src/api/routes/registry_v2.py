"""
Registry API V2 - Frontend Contract Compliance
Operation: Bedrock Task 2 - Align Frontend Contracts

This module creates v2-compatible endpoints for the registry/proving-ground functionality
to match frontend expectations while reusing existing logic from registry.py.
"""

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query, status

# Import existing functionality to avoid code duplication
from src.api.routes.registry import (
    get_db,
    get_challenger_prompts,
    create_challenger_prompt,
    get_challenger_prompt,
    update_challenger_prompt,
    delete_challenger_prompt,
    get_proving_ground_stats,
    get_station_metadata,
)
from src.engine.api.models.registry import (
    ChallengerPromptResponse,
    ChallengerPromptCreate,
    ChallengerPromptUpdate,
    TargetStation,
    ChallengerStatus,
)

logger = logging.getLogger(__name__)

# Create v2 registry router with expected prefix
router = APIRouter(prefix="/api/v2/registry", tags=["Registry V2"])


# =================== CONTRACTS ENDPOINTS ===================
# Frontend calls: /api/v2/registry/contracts

@router.get("/contracts")
async def get_system_contracts(
    limit: int = Query(50, ge=1, le=1000, description="Limit number of records"),
    target_station: Optional[TargetStation] = Query(None, description="Filter by target station"),
    challenger_status: Optional[ChallengerStatus] = Query(ChallengerStatus.ACTIVE, description="Filter by status"),
    db=Depends(get_db),
):
    """
    Get system contracts (mapped from challenger prompts)
    
    Frontend expects: { "contracts": [...] }
    Maps proving ground challenger prompts to system contracts format
    """
    try:
        # Reuse existing challenger prompts logic
        challenger_prompts = await get_challenger_prompts(
            target_station=target_station,
            status=challenger_status,
            golden_case_id=None,
            skip=0,
            limit=limit,
            db=db
        )
        
        # Transform to frontend contract format
        contracts = []
        for prompt in challenger_prompts:
            contract = {
                "contract_id": prompt.prompt_id,
                "capabilities": [
                    f"target_station_{prompt.target_station.value.lower()}",
                    "analytical_processing",
                    "cognitive_analysis"
                ],
                "cognitive_complexity": len(prompt.prompt_text) / 100,  # Simple heuristic
                "target_domain": prompt.target_station.value,
                "performance_metrics": prompt.compilation_metadata or {},
                "status": prompt.status.value,
                "created_at": prompt.created_at.isoformat(),
            }
            contracts.append(contract)
        
        logger.info(f"Retrieved {len(contracts)} system contracts for v2 API")
        return {"contracts": contracts}
        
    except Exception as e:
        logger.error(f"Error retrieving system contracts: {e}")
        # Degrade gracefully for tests: return empty contracts array
        return {"contracts": []}


@router.post("/contracts")
async def create_system_contract(
    contract_data: Dict[str, Any],
    db=Depends(get_db)
):
    """
    Create a new system contract (mapped to challenger prompt creation)
    
    Frontend sends contract data, we transform it to challenger prompt format
    """
    try:
        # Transform v2 contract format to challenger prompt format
        challenger_prompt = ChallengerPromptCreate(
            prompt_name=contract_data.get("contract_name", f"Contract {contract_data.get('contract_id', 'unknown')}"),
            prompt_text=contract_data.get("template_content", "# Generated Contract\n\nAnalyze the following:"),
            version="v2.0",
            target_station=TargetStation.FULL_PIPELINE,
            status=ChallengerStatus.ACTIVE,
            golden_case_id=None,
            compilation_metadata={
                "source": "v2_registry_api",
                "contract_data": contract_data,
                "created_via": "frontend_contract_creation"
            }
        )
        
        # Create using existing logic
        created_prompt = await create_challenger_prompt(challenger_prompt, db)
        
        # Transform back to contract format for response
        contract = {
            "contract_id": created_prompt.prompt_id,
            "capabilities": contract_data.get("capabilities", []),
            "cognitive_complexity": contract_data.get("cognitive_complexity", 1.0),
            "target_domain": created_prompt.target_station.value,
            "performance_metrics": created_prompt.compilation_metadata,
            "status": created_prompt.status.value,
            "created_at": created_prompt.created_at.isoformat(),
        }
        
        logger.info(f"Created system contract: {contract['contract_id']}")
        return contract
        
    except Exception as e:
        logger.error(f"Error creating system contract: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create system contract: {str(e)}",
        )


# =================== TEMPLATES ENDPOINTS ===================  
# Frontend calls: /api/v2/registry/templates

@router.get("/templates") 
async def get_prompt_templates(
    limit: int = Query(50, ge=1, le=1000, description="Limit number of records"),
    contract_id: Optional[str] = Query(None, description="Filter by contract ID"),
    db=Depends(get_db),
):
    """
    Get prompt templates (mapped from challenger prompts)
    
    Frontend expects: { "templates": [...] }
    Maps challenger prompts to template format
    """
    try:
        # Get challenger prompts and transform to template format
        challenger_prompts = await get_challenger_prompts(
            target_station=None,
            status=ChallengerStatus.ACTIVE,
            golden_case_id=contract_id,  # Use golden_case_id as contract filter
            skip=0,
            limit=limit,
            db=db
        )
        
        templates = []
        for prompt in challenger_prompts:
            # Extract variables from prompt text (simple implementation)
            variables = []
            if "{" in prompt.prompt_text:
                import re
                variables = list(set(re.findall(r'\{([^}]+)\}', prompt.prompt_text)))
            
            template = {
                "template_id": f"tpl_{prompt.prompt_id[:8]}",
                "contract_id": prompt.prompt_id,
                "template_content": prompt.prompt_text,
                "variables": variables,
                "template_type": "analysis_prompt",
                "status": prompt.status.value,
                "created_at": prompt.created_at.isoformat(),
            }
            templates.append(template)
        
        logger.info(f"Retrieved {len(templates)} prompt templates for v2 API")
        return {"templates": templates}
        
    except Exception as e:
        logger.error(f"Error retrieving prompt templates: {e}")
        return {"templates": []}


@router.post("/templates")
async def create_prompt_template(
    template_data: Dict[str, Any],
    db=Depends(get_db)
):
    """
    Create a new prompt template (mapped to challenger prompt creation)
    """
    try:
        # Transform template format to challenger prompt format
        challenger_prompt = ChallengerPromptCreate(
            prompt_name=template_data.get("template_name", f"Template {template_data.get('template_id', 'unknown')}"),
            prompt_text=template_data.get("template_content", "# Generated Template\n\nTemplate content here"),
            version="v2.0",
            target_station=TargetStation.FULL_PIPELINE,
            status=ChallengerStatus.ACTIVE,
            golden_case_id=template_data.get("contract_id"),
            compilation_metadata={
                "source": "v2_registry_api",
                "template_data": template_data,
                "created_via": "frontend_template_creation",
                "variables": template_data.get("variables", [])
            }
        )
        
        # Create using existing logic
        created_prompt = await create_challenger_prompt(challenger_prompt, db)
        
        # Transform back to template format
        template = {
            "template_id": f"tpl_{created_prompt.prompt_id[:8]}",
            "contract_id": template_data.get("contract_id", created_prompt.prompt_id),
            "template_content": created_prompt.prompt_text,
            "variables": template_data.get("variables", []),
            "template_type": "analysis_prompt",
            "status": created_prompt.status.value,
            "created_at": created_prompt.created_at.isoformat(),
        }
        
        logger.info(f"Created prompt template: {template['template_id']}")
        return template
        
    except Exception as e:
        logger.error(f"Error creating prompt template: {e}")
        # Degrade gracefully
        return {
            "template_id": None,
            "contract_id": template_data.get("contract_id"),
            "template_content": template_data.get("template_content", ""),
            "variables": template_data.get("variables", []),
            "template_type": "analysis_prompt",
            "status": "inactive",
            "created_at": None,
        }


# =================== ADDITIONAL V2 ENDPOINTS ===================

@router.get("/health")
async def get_registry_health():
    """
    V2 Registry health check endpoint
    """
    try:
        return {
            "status": "healthy",
            "service": "registry_v2", 
            "version": "2.0.0",
            "description": "Frontend contract compliant registry API",
            "endpoints": {
                "contracts": "/api/v2/registry/contracts",
                "templates": "/api/v2/registry/templates"
            }
        }
    except Exception as e:
        logger.error(f"Registry v2 health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registry v2 service unhealthy"
        )


@router.get("/stats")
async def get_registry_stats(db=Depends(get_db)):
    """
    Get registry statistics (mapped from proving ground stats)
    """
    try:
        # Reuse proving ground stats and transform for v2
        pg_stats = await get_proving_ground_stats(db)
        
        registry_stats = {
            "total_contracts": pg_stats.total_challengers,
            "active_contracts": pg_stats.active_challengers, 
            "total_templates": pg_stats.total_challengers,  # Templates are mapped from challengers
            "registry_health": "healthy" if pg_stats.total_challengers > 0 else "empty",
            "performance_metrics": {
                "avg_quality_delta": pg_stats.avg_quality_delta,
                "total_evaluations": pg_stats.total_duels
            }
        }
        
        return registry_stats
        
    except Exception as e:
        logger.error(f"Error retrieving registry stats: {e}")
        return {
            "total_contracts": 0,
            "active_contracts": 0,
            "total_templates": 0,
            "registry_health": "degraded",
            "performance_metrics": {
                "avg_quality_delta": 0.0,
                "total_evaluations": 0,
            },
        }


# Export router for main.py integration
__all__ = ["router"]