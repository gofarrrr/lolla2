"""
Lolla Proving Ground - Registry API Routes
FastAPI endpoints for managing challenger prompts and proving ground operations
"""

import logging
import sqlite3
from datetime import datetime, timezone
from typing import List, Optional, Dict
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends, Query, status

from src.engine.api.models.registry import (
    ChallengerPromptCreate,
    ChallengerPromptUpdate,
    ChallengerPromptResponse,
    DuelConfiguration,
    DuelResult,
    CompilationRequest,
    ProvingGroundStats,
    TargetStation,
    ChallengerStatus,
    get_station_display_name,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/proving-ground", tags=["Proving Ground"])


# Database dependency (placeholder - should be replaced with actual DB session)
def get_db():
    """
    Database session dependency
    This should be implemented to return actual database session
    For now, returns a mock connection for API structure
    """
    # This is a placeholder - implement proper database session management
    try:
        # Example SQLite connection - replace with actual session management
        conn = sqlite3.connect("evaluation_results.db")
        yield conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")
    finally:
        if "conn" in locals():
            conn.close()


# Challenger Prompt CRUD Operations
@router.get("/challenger-prompts", response_model=List[ChallengerPromptResponse])
async def get_challenger_prompts(
    target_station: Optional[TargetStation] = Query(
        None, description="Filter by target station"
    ),
    status: Optional[ChallengerStatus] = Query(
        ChallengerStatus.ACTIVE, description="Filter by status"
    ),
    golden_case_id: Optional[str] = Query(None, description="Filter by golden case ID"),
    skip: int = Query(0, ge=0, description="Skip number of records"),
    limit: int = Query(100, ge=1, le=1000, description="Limit number of records"),
    db=Depends(get_db),
):
    """
    Retrieve challenger prompts with optional filtering

    Query Parameters:
    - target_station: Filter by target station (FULL_PIPELINE, STATION_1, etc.)
    - status: Filter by status (active, archived, draft)
    - golden_case_id: Filter by associated golden case
    - skip: Pagination offset
    - limit: Maximum results to return
    """
    try:
        # Build query with filters
        query = """
        SELECT prompt_id, prompt_name, prompt_text, version, status, 
               target_station, golden_case_id, compilation_metadata,
               created_at, updated_at
        FROM challenger_prompts 
        WHERE 1=1
        """
        params = []

        if target_station:
            query += " AND target_station = ?"
            params.append(target_station.value)

        if status:
            query += " AND status = ?"
            params.append(status.value)

        if golden_case_id:
            query += " AND golden_case_id = ?"
            params.append(golden_case_id)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, skip])

        cursor = db.cursor()
        results = cursor.execute(query, params).fetchall()

        # Convert results to response models
        challenger_prompts = []
        for row in results:
            prompt_data = {
                "prompt_id": row[0],
                "prompt_name": row[1],
                "prompt_text": row[2],
                "version": row[3],
                "status": ChallengerStatus(row[4]),
                "target_station": TargetStation(row[5]),
                "golden_case_id": row[6],
                "compilation_metadata": eval(row[7]) if row[7] else None,
                "created_at": datetime.fromisoformat(row[8]),
                "updated_at": datetime.fromisoformat(row[9]),
            }
            challenger_prompts.append(ChallengerPromptResponse(**prompt_data))

        logger.info(f"Retrieved {len(challenger_prompts)} challenger prompts")
        return challenger_prompts

    except Exception as e:
        logger.error(f"Error retrieving challenger prompts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve challenger prompts: {str(e)}",
        )


@router.post("/challenger-prompts", response_model=ChallengerPromptResponse)
async def create_challenger_prompt(prompt: ChallengerPromptCreate, db=Depends(get_db)):
    """
    Create a new challenger prompt

    Request Body:
    - prompt_name: Descriptive name for the prompt
    - prompt_text: The full prompt text
    - version: Version identifier
    - target_station: Target station for comparison
    - golden_case_id: Associated golden case (optional)
    - compilation_metadata: Metadata about compilation process (optional)
    """
    try:
        # Generate unique ID
        prompt_id = str(uuid4())
        current_time = datetime.now(timezone.utc)

        # Insert into database
        cursor = db.cursor()
        cursor.execute(
            """
            INSERT INTO challenger_prompts 
            (prompt_id, prompt_name, prompt_text, version, status, target_station, 
             golden_case_id, compilation_metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                prompt_id,
                prompt.prompt_name,
                prompt.prompt_text,
                prompt.version,
                prompt.status.value,
                prompt.target_station.value,
                prompt.golden_case_id,
                (
                    str(prompt.compilation_metadata)
                    if prompt.compilation_metadata
                    else None
                ),
                current_time.isoformat(),
                current_time.isoformat(),
            ),
        )

        db.commit()

        # Return created prompt
        created_prompt = ChallengerPromptResponse(
            prompt_id=prompt_id,
            prompt_name=prompt.prompt_name,
            prompt_text=prompt.prompt_text,
            version=prompt.version,
            status=prompt.status,
            target_station=prompt.target_station,
            golden_case_id=prompt.golden_case_id,
            compilation_metadata=prompt.compilation_metadata,
            created_at=current_time,
            updated_at=current_time,
        )

        logger.info(f"Created challenger prompt: {prompt_id}")
        return created_prompt

    except Exception as e:
        logger.error(f"Error creating challenger prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create challenger prompt: {str(e)}",
        )


@router.get("/challenger-prompts/{prompt_id}", response_model=ChallengerPromptResponse)
async def get_challenger_prompt(prompt_id: str, db=Depends(get_db)):
    """
    Retrieve detailed information for a specific challenger prompt
    """
    try:
        cursor = db.cursor()
        result = cursor.execute(
            """
            SELECT prompt_id, prompt_name, prompt_text, version, status, 
                   target_station, golden_case_id, compilation_metadata,
                   created_at, updated_at
            FROM challenger_prompts 
            WHERE prompt_id = ?
        """,
            (prompt_id,),
        ).fetchone()

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Challenger prompt {prompt_id} not found",
            )

        prompt_data = {
            "prompt_id": result[0],
            "prompt_name": result[1],
            "prompt_text": result[2],
            "version": result[3],
            "status": ChallengerStatus(result[4]),
            "target_station": TargetStation(result[5]),
            "golden_case_id": result[6],
            "compilation_metadata": eval(result[7]) if result[7] else None,
            "created_at": datetime.fromisoformat(result[8]),
            "updated_at": datetime.fromisoformat(result[9]),
        }

        return ChallengerPromptResponse(**prompt_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving challenger prompt {prompt_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve challenger prompt: {str(e)}",
        )


@router.put("/challenger-prompts/{prompt_id}", response_model=ChallengerPromptResponse)
async def update_challenger_prompt(
    prompt_id: str, updates: ChallengerPromptUpdate, db=Depends(get_db)
):
    """
    Update an existing challenger prompt

    Path Parameters:
    - prompt_id: UUID of the prompt to update

    Request Body:
    - Any fields to update (prompt_text, status, version, etc.)
    """
    try:
        # Check if prompt exists
        cursor = db.cursor()
        existing = cursor.execute(
            "SELECT prompt_id FROM challenger_prompts WHERE prompt_id = ?", (prompt_id,)
        ).fetchone()

        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Challenger prompt {prompt_id} not found",
            )

        # Build update query dynamically
        update_fields = []
        params = []

        if updates.prompt_name is not None:
            update_fields.append("prompt_name = ?")
            params.append(updates.prompt_name)

        if updates.prompt_text is not None:
            update_fields.append("prompt_text = ?")
            params.append(updates.prompt_text)

        if updates.version is not None:
            update_fields.append("version = ?")
            params.append(updates.version)

        if updates.status is not None:
            update_fields.append("status = ?")
            params.append(updates.status.value)

        if updates.target_station is not None:
            update_fields.append("target_station = ?")
            params.append(updates.target_station.value)

        if updates.golden_case_id is not None:
            update_fields.append("golden_case_id = ?")
            params.append(updates.golden_case_id)

        if updates.compilation_metadata is not None:
            update_fields.append("compilation_metadata = ?")
            params.append(str(updates.compilation_metadata))

        if not update_fields:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid update fields provided",
            )

        # Add updated timestamp
        update_fields.append("updated_at = ?")
        params.append(datetime.utcnow().isoformat())
        params.append(prompt_id)  # For WHERE clause

        update_query = f"""
            UPDATE challenger_prompts 
            SET {', '.join(update_fields)}
            WHERE prompt_id = ?
        """

        cursor.execute(update_query, params)
        db.commit()

        # Return updated prompt
        return await get_challenger_prompt(prompt_id, db)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating challenger prompt {prompt_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update challenger prompt: {str(e)}",
        )


@router.delete("/challenger-prompts/{prompt_id}")
async def delete_challenger_prompt(prompt_id: str, db=Depends(get_db)):
    """
    Delete a challenger prompt
    """
    try:
        cursor = db.cursor()
        result = cursor.execute(
            "DELETE FROM challenger_prompts WHERE prompt_id = ?", (prompt_id,)
        )

        if result.rowcount == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Challenger prompt {prompt_id} not found",
            )

        db.commit()
        logger.info(f"Deleted challenger prompt: {prompt_id}")

        return {"message": f"Challenger prompt {prompt_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting challenger prompt {prompt_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete challenger prompt: {str(e)}",
        )


# Proving Ground Operations
@router.post("/compile-monolith", response_model=ChallengerPromptResponse)
async def compile_monolithic_challenger(
    request: CompilationRequest, db=Depends(get_db)
):
    """
    Compile a monolithic challenger prompt from a golden case

    This endpoint triggers the MonolithCompilerService to:
    1. Execute a dry run of the 8-station pipeline
    2. Extract cognitive DNA from each station
    3. Weave into a single monolithic prompt
    4. Save to the challenger_prompts table
    """
    try:
        # This is a placeholder implementation
        # The actual implementation should use the MonolithCompilerService
        logger.info(
            f"Compiling monolithic challenger for golden case: {request.golden_case_id}"
        )

        # Generate prompt name if not provided
        prompt_name = (
            request.prompt_name or f"Compiled Monolith - {request.golden_case_id[:8]}"
        )

        # Create placeholder compiled prompt
        compiled_prompt = ChallengerPromptCreate(
            prompt_name=prompt_name,
            prompt_text="""# COMPREHENSIVE ANALYSIS SYSTEM - COMPILED MONOLITH

## GLOBAL CONTEXT
This is a compiled monolithic prompt that combines all 8 stations of the Lolla pipeline into a single comprehensive analytical system.

## UNIFIED PERSONA
You are an advanced analytical system combining rapid analysis, deep thinking, conservative and bold perspectives, reality checking, synthesis capabilities, divergent thinking, and final convergence.

## SEQUENTIAL ANALYSIS TASKS

### TASK 1: QUICKTHINK - Initial Analysis
Perform rapid initial analysis of the input, identifying key themes and immediate insights.

### TASK 2: DEEPTHINK - Comprehensive Exploration  
Dive deep into the analysis, exploring nuances and complex relationships.

### TASK 3: BLUETHINK - Conservative Analysis
Apply conservative perspective, identifying risks and cautious approaches.

### TASK 4: REDTHINK - Bold Innovation
Think boldly and innovatively, considering revolutionary approaches.

### TASK 5: GREYTHINK - Reality Check
Apply practical reality checks and feasibility assessments.

### TASK 6: ULTRATHINK - Deep Synthesis
Synthesize all previous analyses into coherent insights.

### TASK 7: DIVERGENTTHINK - Alternative Perspectives
Explore alternative viewpoints and contrarian positions.

### TASK 8: CONVERGENTTHINK - Final Integration
Integrate all perspectives into final actionable recommendations.

## OUTPUT REQUIREMENTS
Provide comprehensive analysis with clear structure and actionable insights.

Now, analyze the following input comprehensively:
{input_data}""",
            version=request.version,
            target_station=TargetStation.FULL_PIPELINE,
            golden_case_id=request.golden_case_id,
            status=ChallengerStatus.ACTIVE,
            compilation_metadata={
                "compilation_type": "automated",
                "source_golden_case": request.golden_case_id,
                "compiled_at": datetime.utcnow().isoformat(),
                "stations_integrated": [
                    "STATION_1",
                    "STATION_2",
                    "STATION_3",
                    "STATION_4",
                    "STATION_5",
                    "STATION_6",
                    "STATION_7",
                    "STATION_8",
                ],
            },
        )

        # Create the challenger prompt
        result = await create_challenger_prompt(compiled_prompt, db)

        logger.info(f"Successfully compiled monolithic challenger: {result.prompt_id}")
        return result

    except Exception as e:
        logger.error(f"Error compiling monolithic challenger: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compile monolithic challenger: {str(e)}",
        )


@router.post("/launch-duel", response_model=DuelResult)
async def launch_proving_ground_duel(config: DuelConfiguration, db=Depends(get_db)):
    """
    Launch a head-to-head duel between Lolla pipeline and challenger

    Request Body:
    - golden_case_id: The golden case to test against
    - challenger_prompt_id: The challenger prompt to use
    - station_to_test: FULL_PIPELINE or specific station
    """
    try:
        logger.info(
            f"Launching duel: {config.challenger_prompt_id} vs Lolla on {config.station_to_test}"
        )

        # Generate unique duel ID
        duel_id = str(uuid4())

        # This is a placeholder implementation
        # The actual implementation should use the enhanced run_harness.py

        # Mock duel result
        mock_result = DuelResult(
            duel_id=duel_id,
            lolla_result={
                "method": (
                    "lolla_pipeline"
                    if config.station_to_test == "FULL_PIPELINE"
                    else f"lolla_{config.station_to_test.lower()}"
                ),
                "output": "Lolla pipeline analysis result...",
                "execution_time": 15.2,
                "station_outputs": (
                    {} if config.station_to_test == "FULL_PIPELINE" else None
                ),
            },
            challenger_result={
                "method": "challenger_monolith",
                "output": "Challenger monolith analysis result...",
                "execution_time": 8.7,
            },
            comparison={
                "lolla_score": 0.82,
                "challenger_score": 0.78,
                "delta": 0.04,
                "winner": "lolla",
            },
            execution_metadata={
                "golden_case_id": config.golden_case_id,
                "challenger_prompt_id": config.challenger_prompt_id,
                "test_scope": config.station_to_test,
                "executed_at": datetime.utcnow().isoformat(),
            },
        )

        logger.info(
            f"Duel {duel_id} completed - Winner: {mock_result.comparison['winner']}"
        )
        return mock_result

    except Exception as e:
        logger.error(f"Error launching duel: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to launch duel: {str(e)}",
        )


@router.get("/stats", response_model=ProvingGroundStats)
async def get_proving_ground_stats(db=Depends(get_db)):
    """
    Get statistics for the proving ground system
    """
    try:
        cursor = db.cursor()

        # Get challenger counts
        challenger_counts = cursor.execute(
            """
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN status = 'active' THEN 1 END) as active
            FROM challenger_prompts
        """
        ).fetchone()

        # Get duel statistics (mock for now)
        stats = ProvingGroundStats(
            total_challengers=challenger_counts[0] if challenger_counts else 0,
            active_challengers=challenger_counts[1] if challenger_counts else 0,
            total_duels=0,  # Will be implemented with actual duel tracking
            lolla_wins=0,
            challenger_wins=0,
            ties=0,
            avg_quality_delta=0.0,
            latest_duel=None,
        )

        return stats

    except Exception as e:
        logger.error(f"Error retrieving proving ground stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve stats: {str(e)}",
        )


@router.get("/stations", response_model=Dict[str, Dict[str, str]])
async def get_station_metadata():
    """
    Get metadata about all available stations
    """
    try:
        station_info = {}
        for station in TargetStation:
            station_info[station.value] = {
                "display_name": get_station_display_name(station),
                "description": f"Station {station.value.split('_')[1] if '_' in station.value else 'FULL'}",
            }

        return station_info

    except Exception as e:
        logger.error(f"Error retrieving station metadata: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve station metadata: {str(e)}",
        )
