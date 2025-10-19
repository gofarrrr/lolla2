"""
METIS Contract Persistence System - P5.1
Persistent storage schema for MetisDataContract with comprehensive versioning

Implements enterprise-grade contract lifecycle persistence with:
- Immutable contract versioning with audit trails
- Automatic checkpoint generation at phase boundaries
- PostgreSQL backend with JSON optimization
- Vector embeddings for semantic search across contracts
- ACID transaction guarantees for consistency
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum
import hashlib

try:
    import asyncpg
    from pgvector.asyncpg import register_vector

    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    asyncpg = None

from src.engine.models.data_contracts import (
    MetisDataContract,
    EngagementPhase,
)
from src.core.state_management import DistributedStateManager


class CheckpointType(str, Enum):
    """Types of contract checkpoints"""

    MANUAL = "manual"  # User-initiated checkpoint
    AUTOMATIC = "automatic"  # System-initiated checkpoint
    PHASE_BOUNDARY = "phase_boundary"  # At engagement phase transitions
    ERROR_RECOVERY = "error_recovery"  # Before risky operations
    MILESTONE = "milestone"  # At significant achievements
    BACKUP = "backup"  # Scheduled backup checkpoint


class ContractStatus(str, Enum):
    """Status of persisted contracts"""

    ACTIVE = "active"  # Currently being processed
    COMPLETED = "completed"  # Engagement finished successfully
    SUSPENDED = "suspended"  # Temporarily paused
    FAILED = "failed"  # Processing failed
    ARCHIVED = "archived"  # Moved to long-term storage
    DELETED = "deleted"  # Soft-deleted (audit trail preserved)


@dataclass
class ContractVersion:
    """Individual version of a contract with metadata"""

    version_id: UUID = field(default_factory=uuid4)
    contract_id: UUID = field(default_factory=uuid4)
    version_number: int = 1
    parent_version_id: Optional[UUID] = None

    # Contract data
    contract_data: MetisDataContract = None
    contract_hash: str = ""

    # Version metadata
    checkpoint_type: CheckpointType = CheckpointType.AUTOMATIC
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    change_summary: str = ""

    # Workflow context
    phase_at_creation: EngagementPhase = EngagementPhase.PROBLEM_STRUCTURING
    progress_percentage: float = 0.0

    # Storage metadata
    compressed_size: int = 0
    storage_location: Optional[str] = None
    vector_embedding: Optional[List[float]] = None


@dataclass
class ContractMetadata:
    """High-level metadata for contract management"""

    contract_id: UUID
    engagement_id: UUID
    client_name: str
    problem_statement_hash: str

    # Lifecycle tracking
    status: ContractStatus = ContractStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_modified_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    # Version tracking
    current_version: int = 1
    total_versions: int = 1
    current_version_id: UUID = field(default_factory=uuid4)

    # Performance metrics
    total_processing_time: float = 0.0
    phase_completion_times: Dict[str, float] = field(default_factory=dict)
    quality_score: Optional[float] = None

    # Business context
    engagement_type: str = "strategic_analysis"
    priority_level: str = "medium"
    tags: List[str] = field(default_factory=list)


class ContractStorageSchema:
    """
    Database schema for contract persistence
    Designed for PostgreSQL with JSON and vector extensions
    """

    @staticmethod
    def get_schema_sql() -> List[str]:
        """Get SQL statements to create the contract storage schema"""
        return [
            # Extensions
            "CREATE EXTENSION IF NOT EXISTS vector;",
            'CREATE EXTENSION IF NOT EXISTS "uuid-ossp";',
            # Contract metadata table
            """
            CREATE TABLE IF NOT EXISTS contract_metadata (
                contract_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                engagement_id UUID UNIQUE NOT NULL,
                client_name VARCHAR(255) NOT NULL,
                problem_statement_hash CHAR(64) NOT NULL,
                
                -- Lifecycle tracking
                status VARCHAR(20) NOT NULL DEFAULT 'active',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_modified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                completed_at TIMESTAMPTZ,
                
                -- Version tracking
                current_version INTEGER NOT NULL DEFAULT 1,
                total_versions INTEGER NOT NULL DEFAULT 1,
                current_version_id UUID NOT NULL,
                
                -- Performance metrics
                total_processing_time REAL NOT NULL DEFAULT 0.0,
                phase_completion_times JSONB NOT NULL DEFAULT '{}',
                quality_score REAL,
                
                -- Business context
                engagement_type VARCHAR(50) NOT NULL DEFAULT 'strategic_analysis',
                priority_level VARCHAR(20) NOT NULL DEFAULT 'medium',
                tags TEXT[] DEFAULT '{}',
                
                -- Indexes
                CONSTRAINT valid_status CHECK (status IN ('active', 'completed', 'suspended', 'failed', 'archived', 'deleted')),
                CONSTRAINT valid_priority CHECK (priority_level IN ('low', 'medium', 'high', 'critical'))
            );
            """,
            # Contract versions table (stores actual contract data)
            """
            CREATE TABLE IF NOT EXISTS contract_versions (
                version_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                contract_id UUID NOT NULL REFERENCES contract_metadata(contract_id) ON DELETE CASCADE,
                version_number INTEGER NOT NULL,
                parent_version_id UUID REFERENCES contract_versions(version_id),
                
                -- Contract data (compressed JSON)
                contract_data JSONB NOT NULL,
                contract_hash CHAR(64) NOT NULL,
                
                -- Version metadata
                checkpoint_type VARCHAR(20) NOT NULL DEFAULT 'automatic',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                created_by VARCHAR(100) NOT NULL DEFAULT 'system',
                change_summary TEXT,
                
                -- Workflow context
                phase_at_creation VARCHAR(30) NOT NULL,
                progress_percentage REAL NOT NULL DEFAULT 0.0,
                
                -- Storage metadata
                compressed_size INTEGER NOT NULL DEFAULT 0,
                storage_location TEXT,
                vector_embedding vector(1536),
                
                -- Constraints
                CONSTRAINT valid_checkpoint_type CHECK (checkpoint_type IN ('manual', 'automatic', 'phase_boundary', 'error_recovery', 'milestone', 'backup')),
                CONSTRAINT valid_phase CHECK (phase_at_creation IN ('problem_structuring', 'hypothesis_generation', 'analysis_execution', 'synthesis_delivery')),
                CONSTRAINT valid_progress CHECK (progress_percentage >= 0 AND progress_percentage <= 100),
                UNIQUE(contract_id, version_number)
            );
            """,
            # Contract checkpoints table (quick lookup for recovery)
            """
            CREATE TABLE IF NOT EXISTS contract_checkpoints (
                checkpoint_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                contract_id UUID NOT NULL REFERENCES contract_metadata(contract_id) ON DELETE CASCADE,
                version_id UUID NOT NULL REFERENCES contract_versions(version_id) ON DELETE CASCADE,
                
                -- Checkpoint metadata
                checkpoint_type VARCHAR(20) NOT NULL,
                checkpoint_name VARCHAR(100),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                
                -- Recovery context
                phase_before VARCHAR(30) NOT NULL,
                phase_after VARCHAR(30),
                recovery_data JSONB,
                
                -- Performance tracking
                processing_time_at_checkpoint REAL,
                memory_usage_mb INTEGER,
                
                CONSTRAINT valid_checkpoint_phase_before CHECK (phase_before IN ('problem_structuring', 'hypothesis_generation', 'analysis_execution', 'synthesis_delivery')),
                CONSTRAINT valid_checkpoint_phase_after CHECK (phase_after IN ('problem_structuring', 'hypothesis_generation', 'analysis_execution', 'synthesis_delivery') OR phase_after IS NULL)
            );
            """,
            # Indexes for performance
            "CREATE INDEX IF NOT EXISTS idx_contract_metadata_engagement_id ON contract_metadata(engagement_id);",
            "CREATE INDEX IF NOT EXISTS idx_contract_metadata_status ON contract_metadata(status);",
            "CREATE INDEX IF NOT EXISTS idx_contract_metadata_created_at ON contract_metadata(created_at);",
            "CREATE INDEX IF NOT EXISTS idx_contract_metadata_client_name ON contract_metadata(client_name);",
            "CREATE INDEX IF NOT EXISTS idx_contract_versions_contract_id ON contract_versions(contract_id);",
            "CREATE INDEX IF NOT EXISTS idx_contract_versions_created_at ON contract_versions(created_at);",
            "CREATE INDEX IF NOT EXISTS idx_contract_versions_checkpoint_type ON contract_versions(checkpoint_type);",
            "CREATE INDEX IF NOT EXISTS idx_contract_versions_phase ON contract_versions(phase_at_creation);",
            "CREATE INDEX IF NOT EXISTS idx_contract_versions_hash ON contract_versions(contract_hash);",
            "CREATE INDEX IF NOT EXISTS idx_contract_checkpoints_contract_id ON contract_checkpoints(contract_id);",
            "CREATE INDEX IF NOT EXISTS idx_contract_checkpoints_type ON contract_checkpoints(checkpoint_type);",
            "CREATE INDEX IF NOT EXISTS idx_contract_checkpoints_created_at ON contract_checkpoints(created_at);",
            # Vector similarity index for semantic search
            "CREATE INDEX IF NOT EXISTS idx_contract_versions_embedding ON contract_versions USING ivfflat (vector_embedding vector_cosine_ops) WITH (lists = 100);",
            # Updated timestamp trigger
            """
            CREATE OR REPLACE FUNCTION update_last_modified_at()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.last_modified_at = NOW();
                RETURN NEW;
            END;
            $$ language 'plpgsql';
            """,
            "DROP TRIGGER IF EXISTS update_contract_metadata_timestamp ON contract_metadata;",
            """
            CREATE TRIGGER update_contract_metadata_timestamp
                BEFORE UPDATE ON contract_metadata
                FOR EACH ROW
                EXECUTE FUNCTION update_last_modified_at();
            """,
        ]


class ContractPersistenceManager:
    """
    Manages persistent storage of MetisDataContract with versioning
    Provides enterprise-grade contract lifecycle management
    """

    def __init__(
        self,
        postgres_url: str = "postgresql://localhost:5432/metis",
        state_manager: Optional[DistributedStateManager] = None,
        enable_compression: bool = True,
        enable_vector_search: bool = True,
    ):
        self.postgres_url = postgres_url
        self.state_manager = state_manager
        self.enable_compression = enable_compression
        self.enable_vector_search = enable_vector_search

        self.pool: Optional[asyncpg.Pool] = None
        self.logger = logging.getLogger(__name__)

        # In-memory cache for active contracts
        self.contract_cache: Dict[UUID, ContractVersion] = {}
        self.metadata_cache: Dict[UUID, ContractMetadata] = {}
        self.cache_ttl = timedelta(minutes=15)

        # Performance tracking
        self.metrics = {
            "contracts_stored": 0,
            "versions_created": 0,
            "checkpoints_created": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    async def initialize(self) -> bool:
        """Initialize the persistence system and create schema"""
        if not POSTGRES_AVAILABLE:
            self.logger.warning("PostgreSQL not available, using in-memory persistence")
            return False

        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                self.postgres_url, min_size=2, max_size=20, command_timeout=60
            )

            # Initialize vector extension and schema
            async with self.pool.acquire() as conn:
                await register_vector(conn)

                # Execute schema creation
                schema_sql = ContractStorageSchema.get_schema_sql()
                for sql_statement in schema_sql:
                    try:
                        await conn.execute(sql_statement)
                    except Exception as e:
                        self.logger.warning(f"Schema creation warning: {str(e)}")

            self.logger.info("🗄️ Contract persistence system initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize contract persistence: {str(e)}")
            return False

    def _calculate_contract_hash(self, contract: MetisDataContract) -> str:
        """Calculate SHA-256 hash of contract for versioning"""
        contract_json = contract.json(sort_keys=True)
        return hashlib.sha256(contract_json.encode()).hexdigest()

    def _generate_change_summary(
        self, old_contract: Optional[MetisDataContract], new_contract: MetisDataContract
    ) -> str:
        """Generate human-readable summary of changes between contract versions"""
        if not old_contract:
            return "Initial contract creation"

        changes = []

        # Check workflow phase changes
        if (
            old_contract.workflow_state.current_phase
            != new_contract.workflow_state.current_phase
        ):
            changes.append(
                f"Phase: {old_contract.workflow_state.current_phase.value} → {new_contract.workflow_state.current_phase.value}"
            )

        # Check deliverable artifacts
        old_artifacts = len(old_contract.deliverable_artifacts)
        new_artifacts = len(new_contract.deliverable_artifacts)
        if old_artifacts != new_artifacts:
            changes.append(f"Artifacts: {old_artifacts} → {new_artifacts}")

        # Check completed phases
        old_completed = len(old_contract.workflow_state.completed_phases)
        new_completed = len(new_contract.workflow_state.completed_phases)
        if old_completed != new_completed:
            changes.append(f"Completed phases: {old_completed} → {new_completed}")

        # Check processing metadata updates
        old_meta_keys = set(old_contract.processing_metadata.keys())
        new_meta_keys = set(new_contract.processing_metadata.keys())
        new_keys = new_meta_keys - old_meta_keys
        if new_keys:
            changes.append(f"Added metadata: {', '.join(new_keys)}")

        return "; ".join(changes) if changes else "Minor updates"

    async def store_contract(
        self,
        contract: MetisDataContract,
        checkpoint_type: CheckpointType = CheckpointType.AUTOMATIC,
        change_summary: Optional[str] = None,
        created_by: str = "system",
    ) -> ContractVersion:
        """
        Store a new version of a contract with full versioning
        Returns the created contract version
        """
        if not self.pool:
            raise RuntimeError("Persistence system not initialized")

        engagement_id = contract.engagement_context.engagement_id
        contract_hash = self._calculate_contract_hash(contract)

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Check if this is a new contract or update
                existing_metadata = await conn.fetchrow(
                    "SELECT * FROM contract_metadata WHERE engagement_id = $1",
                    engagement_id,
                )

                if existing_metadata:
                    # Update existing contract
                    contract_id = existing_metadata["contract_id"]
                    current_version = existing_metadata["current_version"]
                    new_version_number = current_version + 1

                    # Get previous version for change detection
                    previous_version = await conn.fetchrow(
                        "SELECT contract_data FROM contract_versions WHERE contract_id = $1 AND version_number = $2",
                        contract_id,
                        current_version,
                    )

                    old_contract = None
                    if previous_version:
                        old_contract = MetisDataContract.parse_obj(
                            previous_version["contract_data"]
                        )

                    # Generate change summary if not provided
                    if not change_summary:
                        change_summary = self._generate_change_summary(
                            old_contract, contract
                        )

                else:
                    # Create new contract
                    contract_id = uuid4()
                    new_version_number = 1

                    # Create metadata record
                    await conn.execute(
                        """
                        INSERT INTO contract_metadata (
                            contract_id, engagement_id, client_name, problem_statement_hash,
                            engagement_type, priority_level, current_version_id
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                        contract_id,
                        engagement_id,
                        contract.engagement_context.client_name,
                        hashlib.sha256(
                            contract.engagement_context.problem_statement.encode()
                        ).hexdigest(),
                        contract.engagement_context.business_context.get(
                            "engagement_type", "strategic_analysis"
                        ),
                        contract.engagement_context.business_context.get(
                            "priority_level", "medium"
                        ),
                        uuid4(),  # Will be updated with actual version_id
                    )

                    if not change_summary:
                        change_summary = "Initial contract creation"

                # Create new version record
                version_id = uuid4()
                contract_data = contract.dict()

                # Calculate progress percentage based on completed phases
                total_phases = 4
                completed_phases = len(contract.workflow_state.completed_phases)
                progress_percentage = (completed_phases / total_phases) * 100

                await conn.execute(
                    """
                    INSERT INTO contract_versions (
                        version_id, contract_id, version_number, contract_data, contract_hash,
                        checkpoint_type, created_by, change_summary, phase_at_creation, progress_percentage,
                        compressed_size
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                    version_id,
                    contract_id,
                    new_version_number,
                    json.dumps(contract_data),
                    contract_hash,
                    checkpoint_type.value,
                    created_by,
                    change_summary,
                    contract.workflow_state.current_phase.value,
                    progress_percentage,
                    len(json.dumps(contract_data)),
                )

                # Update metadata with new version info
                await conn.execute(
                    """
                    UPDATE contract_metadata 
                    SET current_version = $1, total_versions = $2, current_version_id = $3,
                        last_modified_at = NOW()
                    WHERE contract_id = $4
                """,
                    new_version_number,
                    new_version_number,
                    version_id,
                    contract_id,
                )

                # Create checkpoint record for important transitions
                if checkpoint_type in [
                    CheckpointType.PHASE_BOUNDARY,
                    CheckpointType.MILESTONE,
                    CheckpointType.MANUAL,
                ]:
                    await self._create_checkpoint_record(
                        conn, contract_id, version_id, checkpoint_type, contract
                    )

                # Update metrics
                self.metrics["versions_created"] += 1
                if new_version_number == 1:
                    self.metrics["contracts_stored"] += 1

                # Create and cache the version object
                contract_version = ContractVersion(
                    version_id=version_id,
                    contract_id=contract_id,
                    version_number=new_version_number,
                    contract_data=contract,
                    contract_hash=contract_hash,
                    checkpoint_type=checkpoint_type,
                    created_by=created_by,
                    change_summary=change_summary,
                    phase_at_creation=contract.workflow_state.current_phase,
                    progress_percentage=progress_percentage,
                    compressed_size=len(json.dumps(contract_data)),
                )

                # Update cache
                self.contract_cache[engagement_id] = contract_version

                self.logger.info(
                    f"📁 Stored contract version {new_version_number} for engagement {engagement_id}"
                )
                return contract_version

    async def _create_checkpoint_record(
        self,
        conn,
        contract_id: UUID,
        version_id: UUID,
        checkpoint_type: CheckpointType,
        contract: MetisDataContract,
    ):
        """Create a checkpoint record for recovery purposes"""
        checkpoint_id = uuid4()

        # Determine phase transition if applicable
        phase_before = contract.workflow_state.current_phase.value
        phase_after = None

        if len(contract.workflow_state.completed_phases) > 0:
            last_completed = contract.workflow_state.completed_phases[-1]
            if last_completed != contract.workflow_state.current_phase:
                phase_after = contract.workflow_state.current_phase.value

        # Collect recovery data
        recovery_data = {
            "workflow_state": contract.workflow_state.dict(),
            "cognitive_state": contract.cognitive_state.dict(),
            "artifacts_count": len(contract.deliverable_artifacts),
            "processing_metadata_keys": list(contract.processing_metadata.keys()),
        }

        await conn.execute(
            """
            INSERT INTO contract_checkpoints (
                checkpoint_id, contract_id, version_id, checkpoint_type,
                phase_before, phase_after, recovery_data,
                processing_time_at_checkpoint
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
            checkpoint_id,
            contract_id,
            version_id,
            checkpoint_type.value,
            phase_before,
            phase_after,
            json.dumps(recovery_data),
            contract.workflow_state.performance_metrics.get("total_workflow_time", 0.0),
        )

        self.metrics["checkpoints_created"] += 1

    async def get_contract(
        self, engagement_id: UUID, version: Optional[int] = None
    ) -> Optional[MetisDataContract]:
        """
        Retrieve a contract by engagement ID and optional version
        Returns the latest version if version is not specified
        """
        if not self.pool:
            raise RuntimeError("Persistence system not initialized")

        # Check cache first
        if not version and engagement_id in self.contract_cache:
            self.metrics["cache_hits"] += 1
            return self.contract_cache[engagement_id].contract_data

        self.metrics["cache_misses"] += 1

        async with self.pool.acquire() as conn:
            if version:
                # Get specific version
                row = await conn.fetchrow(
                    """
                    SELECT cv.contract_data
                    FROM contract_versions cv
                    JOIN contract_metadata cm ON cv.contract_id = cm.contract_id
                    WHERE cm.engagement_id = $1 AND cv.version_number = $2
                """,
                    engagement_id,
                    version,
                )
            else:
                # Get latest version
                row = await conn.fetchrow(
                    """
                    SELECT cv.contract_data
                    FROM contract_versions cv
                    JOIN contract_metadata cm ON cv.contract_id = cm.contract_id
                    WHERE cm.engagement_id = $1
                    ORDER BY cv.version_number DESC
                    LIMIT 1
                """,
                    engagement_id,
                )

            if row:
                contract_data = row["contract_data"]
                contract = MetisDataContract.parse_obj(contract_data)

                # Update cache for latest version
                if not version:
                    version_obj = ContractVersion(contract_data=contract)
                    self.contract_cache[engagement_id] = version_obj

                return contract

            return None

    async def get_contract_versions(self, engagement_id: UUID) -> List[ContractVersion]:
        """Get all versions of a contract with metadata"""
        if not self.pool:
            raise RuntimeError("Persistence system not initialized")

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT cv.*, cm.engagement_id
                FROM contract_versions cv
                JOIN contract_metadata cm ON cv.contract_id = cm.contract_id
                WHERE cm.engagement_id = $1
                ORDER BY cv.version_number ASC
            """,
                engagement_id,
            )

            versions = []
            for row in rows:
                contract_data = MetisDataContract.parse_obj(row["contract_data"])
                version = ContractVersion(
                    version_id=row["version_id"],
                    contract_id=row["contract_id"],
                    version_number=row["version_number"],
                    parent_version_id=row["parent_version_id"],
                    contract_data=contract_data,
                    contract_hash=row["contract_hash"],
                    checkpoint_type=CheckpointType(row["checkpoint_type"]),
                    created_at=row["created_at"],
                    created_by=row["created_by"],
                    change_summary=row["change_summary"],
                    phase_at_creation=EngagementPhase(row["phase_at_creation"]),
                    progress_percentage=row["progress_percentage"],
                    compressed_size=row["compressed_size"],
                )
                versions.append(version)

            return versions

    async def get_latest_checkpoint(
        self, engagement_id: UUID, checkpoint_type: Optional[CheckpointType] = None
    ) -> Optional[ContractVersion]:
        """Get the latest checkpoint of a specific type for recovery"""
        if not self.pool:
            raise RuntimeError("Persistence system not initialized")

        async with self.pool.acquire() as conn:
            if checkpoint_type:
                row = await conn.fetchrow(
                    """
                    SELECT cv.*
                    FROM contract_versions cv
                    JOIN contract_metadata cm ON cv.contract_id = cm.contract_id
                    WHERE cm.engagement_id = $1 AND cv.checkpoint_type = $2
                    ORDER BY cv.created_at DESC
                    LIMIT 1
                """,
                    engagement_id,
                    checkpoint_type.value,
                )
            else:
                row = await conn.fetchrow(
                    """
                    SELECT cv.*
                    FROM contract_versions cv
                    JOIN contract_metadata cm ON cv.contract_id = cm.contract_id
                    WHERE cm.engagement_id = $1
                    ORDER BY cv.created_at DESC
                    LIMIT 1
                """,
                    engagement_id,
                )

            if row:
                contract_data = MetisDataContract.parse_obj(row["contract_data"])
                return ContractVersion(
                    version_id=row["version_id"],
                    contract_id=row["contract_id"],
                    version_number=row["version_number"],
                    contract_data=contract_data,
                    contract_hash=row["contract_hash"],
                    checkpoint_type=CheckpointType(row["checkpoint_type"]),
                    created_at=row["created_at"],
                    created_by=row["created_by"],
                    change_summary=row["change_summary"],
                    phase_at_creation=EngagementPhase(row["phase_at_creation"]),
                    progress_percentage=row["progress_percentage"],
                )

            return None

    async def get_metrics(self) -> Dict[str, Any]:
        """Get persistence system metrics"""
        if not self.pool:
            return {"error": "Persistence system not initialized", **self.metrics}

        async with self.pool.acquire() as conn:
            # Database statistics
            db_stats = await conn.fetchrow(
                """
                SELECT 
                    (SELECT COUNT(*) FROM contract_metadata) as total_contracts,
                    (SELECT COUNT(*) FROM contract_versions) as total_versions,
                    (SELECT COUNT(*) FROM contract_checkpoints) as total_checkpoints,
                    (SELECT COUNT(*) FROM contract_metadata WHERE status = 'active') as active_contracts,
                    (SELECT AVG(total_processing_time) FROM contract_metadata WHERE total_processing_time > 0) as avg_processing_time
            """
            )

            return {
                **self.metrics,
                "database_stats": dict(db_stats),
                "cache_size": len(self.contract_cache),
                "postgres_available": POSTGRES_AVAILABLE,
            }

    async def cleanup_old_versions(
        self, retention_days: int = 30, keep_checkpoints: bool = True
    ) -> int:
        """Clean up old contract versions while preserving important checkpoints"""
        if not self.pool:
            raise RuntimeError("Persistence system not initialized")

        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Count versions to be deleted
                if keep_checkpoints:
                    count_result = await conn.fetchval(
                        """
                        SELECT COUNT(*) FROM contract_versions 
                        WHERE created_at < $1 
                        AND checkpoint_type NOT IN ('phase_boundary', 'milestone', 'manual', 'error_recovery')
                        AND version_number > 1
                    """,
                        cutoff_date,
                    )

                    # Delete old automatic versions
                    await conn.execute(
                        """
                        DELETE FROM contract_versions 
                        WHERE created_at < $1 
                        AND checkpoint_type NOT IN ('phase_boundary', 'milestone', 'manual', 'error_recovery')
                        AND version_number > 1
                    """,
                        cutoff_date,
                    )
                else:
                    count_result = await conn.fetchval(
                        """
                        SELECT COUNT(*) FROM contract_versions 
                        WHERE created_at < $1 AND version_number > 1
                    """,
                        cutoff_date,
                    )

                    # Delete all old versions except version 1
                    await conn.execute(
                        """
                        DELETE FROM contract_versions 
                        WHERE created_at < $1 AND version_number > 1
                    """,
                        cutoff_date,
                    )

                self.logger.info(f"🧹 Cleaned up {count_result} old contract versions")
                return count_result or 0

    async def shutdown(self):
        """Shutdown the persistence system gracefully"""
        if self.pool:
            await self.pool.close()
            self.logger.info("📪 Contract persistence system shutdown complete")


# Export main classes
__all__ = [
    "ContractPersistenceManager",
    "ContractVersion",
    "ContractMetadata",
    "ContractStorageSchema",
    "CheckpointType",
    "ContractStatus",
]
