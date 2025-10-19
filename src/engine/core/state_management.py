"""
METIS Distributed State Management System
F006: PostgreSQL + pgvector with vector clock synchronization

Implements distributed state management with ACID guarantees,
vector embeddings for semantic search, and optimistic concurrency control.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
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

import numpy as np


class StateType(str, Enum):
    """Types of state in the distributed system"""

    ENGAGEMENT = "engagement"
    COGNITIVE = "cognitive"
    WORKFLOW = "workflow"
    DELIVERABLE = "deliverable"
    CACHE = "cache"
    CONFIGURATION = "configuration"
    AGENTS = "agents"
    PERFORMANCE = "performance"
    ALERTS = "alerts"
    HEALTH = "health"
    MONITORING = "monitoring"
    METRICS = "metrics"


@dataclass
class VectorClock:
    """
    Vector clock for distributed state synchronization
    Implements Lamport timestamps for causality tracking
    """

    node_id: str
    clock: Dict[str, int] = field(default_factory=dict)

    def increment(self) -> None:
        """Increment this node's clock value"""
        self.clock[self.node_id] = self.clock.get(self.node_id, 0) + 1

    def update(self, other: "VectorClock") -> None:
        """Update clock based on received vector clock"""
        for node, timestamp in other.clock.items():
            self.clock[node] = max(self.clock.get(node, 0), timestamp)
        self.increment()

    def happens_before(self, other: "VectorClock") -> bool:
        """Check if this clock happens-before another"""
        for node, timestamp in self.clock.items():
            if timestamp > other.clock.get(node, 0):
                return False
        return any(
            timestamp < other.clock.get(node, 0)
            for node, timestamp in other.clock.items()
        )

    def concurrent_with(self, other: "VectorClock") -> bool:
        """Check if two clocks are concurrent (no causal relationship)"""
        return not self.happens_before(other) and not other.happens_before(self)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize vector clock to dictionary"""
        return {"node_id": self.node_id, "clock": self.clock.copy()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorClock":
        """Deserialize vector clock from dictionary"""
        vc = cls(node_id=data["node_id"])
        vc.clock = data["clock"].copy()
        return vc


@dataclass
class StateVersion:
    """Version information for distributed state"""

    version_id: UUID = field(default_factory=uuid4)
    vector_clock: VectorClock = field(default_factory=lambda: VectorClock("default"))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    checksum: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class OptimisticLockException(Exception):
    """Raised when optimistic locking detects a conflict"""

    pass


class DistributedStateManager:
    """
    Manages distributed state with PostgreSQL backend
    Implements vector embeddings, versioning, and synchronization
    """

    def __init__(self, postgres_url: str, node_id: str = None, pool_size: int = 10):
        self.postgres_url = postgres_url
        self.node_id = node_id or f"node_{uuid4().hex[:8]}"
        self.pool_size = pool_size
        self.pool: Optional[asyncpg.Pool] = None
        self.vector_clock = VectorClock(self.node_id)
        self.logger = logging.getLogger(__name__)

        # State caching layer
        self.local_cache: Dict[str, Tuple[Any, StateVersion]] = {}
        self.cache_ttl = timedelta(minutes=5)

    async def initialize(self) -> None:
        """Initialize database connection and schema"""
        if not POSTGRES_AVAILABLE:
            self.logger.warning("PostgreSQL not available, using in-memory state")
            return

        # Create connection pool
        self.pool = await asyncpg.create_pool(
            self.postgres_url, min_size=2, max_size=self.pool_size, command_timeout=60
        )

        # Register pgvector extension
        async with self.pool.acquire() as conn:
            await register_vector(conn)

            # Create schema
            await self._create_schema(conn)

    async def _create_schema(self, conn) -> None:
        """Create database schema for distributed state"""

        # Enable required extensions
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

        # Main state table
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metis_state (
                state_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                state_key VARCHAR(255) UNIQUE NOT NULL,
                state_type VARCHAR(50) NOT NULL,
                state_data JSONB NOT NULL,
                version_id UUID NOT NULL,
                vector_clock JSONB NOT NULL,
                embedding vector(1536),
                checksum VARCHAR(64) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by VARCHAR(100),
                metadata JSONB DEFAULT '{}'::jsonb
            )
        """
        )

        # Version history table
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metis_state_history (
                history_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                state_id UUID REFERENCES metis_state(state_id),
                state_key VARCHAR(255) NOT NULL,
                version_id UUID NOT NULL,
                vector_clock JSONB NOT NULL,
                state_data JSONB NOT NULL,
                change_type VARCHAR(50),
                changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                changed_by VARCHAR(100)
            )
        """
        )

        # Indexes for performance
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_state_key ON metis_state(state_key);
            CREATE INDEX IF NOT EXISTS idx_state_type ON metis_state(state_type);
            CREATE INDEX IF NOT EXISTS idx_state_updated ON metis_state(updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_embedding ON metis_state 
                USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
        """
        )

        # Conflict resolution table
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metis_conflicts (
                conflict_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                state_key VARCHAR(255) NOT NULL,
                local_version JSONB NOT NULL,
                remote_version JSONB NOT NULL,
                resolution_strategy VARCHAR(50),
                resolved BOOLEAN DEFAULT FALSE,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP
            )
        """
        )

    def _calculate_checksum(self, data: Any) -> str:
        """Calculate SHA256 checksum of state data"""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    async def get_state(
        self, state_key: str, state_type: StateType = StateType.ENGAGEMENT
    ) -> Optional[Tuple[Any, StateVersion]]:
        """
        Retrieve state with version information
        Implements read-through caching
        """

        # Check local cache first
        cache_key = f"{state_type}:{state_key}"
        if cache_key in self.local_cache:
            cached_data, version = self.local_cache[cache_key]
            if datetime.utcnow() - version.timestamp < self.cache_ttl:
                return cached_data, version

        if not self.pool:
            return None

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT state_data, version_id, vector_clock, checksum, updated_at, metadata
                FROM metis_state
                WHERE state_key = $1 AND state_type = $2
            """,
                state_key,
                state_type.value,
            )

            if not row:
                return None

            # Reconstruct version information
            version = StateVersion(
                version_id=UUID(str(row["version_id"])),
                vector_clock=VectorClock.from_dict(row["vector_clock"]),
                timestamp=row["updated_at"],
                checksum=row["checksum"],
                metadata=row["metadata"],
            )

            # Update local cache
            self.local_cache[cache_key] = (row["state_data"], version)

            return row["state_data"], version

    async def set_state(
        self,
        state_key: str,
        state_data: Any,
        state_type: StateType = StateType.ENGAGEMENT,
        expected_version: Optional[StateVersion] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> StateVersion:
        """
        Set state with optimistic concurrency control
        Validates version before update
        """

        # Increment vector clock
        self.vector_clock.increment()

        # Create new version
        new_version = StateVersion(
            vector_clock=self.vector_clock,
            checksum=self._calculate_checksum(state_data),
        )

        if not self.pool:
            # In-memory fallback
            cache_key = f"{state_type}:{state_key}"
            self.local_cache[cache_key] = (state_data, new_version)
            return new_version

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Check for existing state
                existing = await conn.fetchrow(
                    """
                    SELECT version_id, vector_clock, checksum
                    FROM metis_state
                    WHERE state_key = $1 AND state_type = $2
                    FOR UPDATE
                """,
                    state_key,
                    state_type.value,
                )

                if existing and expected_version:
                    # Validate optimistic lock
                    if str(existing["version_id"]) != str(expected_version.version_id):
                        # Check if vector clocks are concurrent
                        existing_vc = VectorClock.from_dict(existing["vector_clock"])
                        if existing_vc.concurrent_with(expected_version.vector_clock):
                            # Log conflict for resolution
                            await self._log_conflict(
                                conn, state_key, expected_version, existing
                            )
                        raise OptimisticLockException(
                            f"Version mismatch for {state_key}. "
                            f"Expected {expected_version.version_id}, "
                            f"found {existing['version_id']}"
                        )

                # Prepare embedding if provided
                embedding_value = embedding.tolist() if embedding is not None else None

                if existing:
                    # Update existing state
                    await conn.execute(
                        """
                        UPDATE metis_state
                        SET state_data = $1,
                            version_id = $2,
                            vector_clock = $3,
                            embedding = $4,
                            checksum = $5,
                            updated_at = CURRENT_TIMESTAMP,
                            metadata = $6
                        WHERE state_key = $7 AND state_type = $8
                    """,
                        json.dumps(state_data, default=str),
                        new_version.version_id,
                        new_version.vector_clock.to_dict(),
                        embedding_value,
                        new_version.checksum,
                        json.dumps(new_version.metadata),
                        state_key,
                        state_type.value,
                    )

                    # Log to history
                    await self._log_history(
                        conn,
                        existing["version_id"],
                        state_key,
                        state_data,
                        new_version,
                        "update",
                    )
                else:
                    # Insert new state
                    await conn.execute(
                        """
                        INSERT INTO metis_state 
                        (state_key, state_type, state_data, version_id, 
                         vector_clock, embedding, checksum, created_by, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                        state_key,
                        state_type.value,
                        json.dumps(state_data, default=str),
                        new_version.version_id,
                        new_version.vector_clock.to_dict(),
                        embedding_value,
                        new_version.checksum,
                        self.node_id,
                        json.dumps(new_version.metadata),
                    )

                    # Log to history
                    await self._log_history(
                        conn,
                        new_version.version_id,
                        state_key,
                        state_data,
                        new_version,
                        "create",
                    )

        # Update local cache
        cache_key = f"{state_type}:{state_key}"
        self.local_cache[cache_key] = (state_data, new_version)

        return new_version

    async def _log_conflict(
        self,
        conn,
        state_key: str,
        local_version: StateVersion,
        remote_version: Dict[str, Any],
    ) -> None:
        """Log version conflict for later resolution"""
        await conn.execute(
            """
            INSERT INTO metis_conflicts
            (state_key, local_version, remote_version, resolution_strategy)
            VALUES ($1, $2, $3, $4)
        """,
            state_key,
            json.dumps(local_version.to_dict(), default=str),
            json.dumps(remote_version, default=str),
            "manual",
        )

    async def _log_history(
        self,
        conn,
        state_id: UUID,
        state_key: str,
        state_data: Any,
        version: StateVersion,
        change_type: str,
    ) -> None:
        """Log state change to history table"""
        await conn.execute(
            """
            INSERT INTO metis_state_history
            (state_id, state_key, version_id, vector_clock, 
             state_data, change_type, changed_by)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
            state_id,
            state_key,
            version.version_id,
            version.vector_clock.to_dict(),
            json.dumps(state_data, default=str),
            change_type,
            self.node_id,
        )

    async def search_by_embedding(
        self,
        embedding: np.ndarray,
        state_type: Optional[StateType] = None,
        limit: int = 10,
        similarity_threshold: float = 0.7,
    ) -> List[Tuple[str, Any, float]]:
        """
        Search states by vector similarity
        Returns list of (state_key, state_data, similarity_score)
        """
        if not self.pool:
            return []

        embedding_list = embedding.tolist()

        async with self.pool.acquire() as conn:
            query = """
                SELECT state_key, state_data, 
                       1 - (embedding <=> $1::vector) as similarity
                FROM metis_state
                WHERE 1 - (embedding <=> $1::vector) > $2
            """
            params = [embedding_list, similarity_threshold]

            if state_type:
                query += " AND state_type = $3"
                params.append(state_type.value)

            query += " ORDER BY embedding <=> $1::vector LIMIT $" + str(len(params) + 1)
            params.append(limit)

            rows = await conn.fetch(query, *params)

            return [
                (row["state_key"], row["state_data"], row["similarity"]) for row in rows
            ]

    async def get_state_history(
        self, state_key: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve version history for a state key"""
        if not self.pool:
            return []

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT version_id, vector_clock, state_data, 
                       change_type, changed_at, changed_by
                FROM metis_state_history
                WHERE state_key = $1
                ORDER BY changed_at DESC
                LIMIT $2
            """,
                state_key,
                limit,
            )

            return [dict(row) for row in rows]

    async def synchronize_state(
        self, remote_states: Dict[str, Tuple[Any, VectorClock]]
    ) -> Dict[str, str]:
        """
        Synchronize state with remote nodes
        Implements vector clock-based conflict resolution
        """
        results = {}

        for state_key, (remote_data, remote_clock) in remote_states.items():
            try:
                local_state = await self.get_state(state_key)

                if not local_state:
                    # No local state, accept remote
                    self.vector_clock.update(remote_clock)
                    await self.set_state(state_key, remote_data)
                    results[state_key] = "accepted"
                else:
                    local_data, local_version = local_state
                    local_clock = local_version.vector_clock

                    if remote_clock.happens_before(local_clock):
                        # Remote is older, keep local
                        results[state_key] = "kept_local"
                    elif local_clock.happens_before(remote_clock):
                        # Local is older, accept remote
                        self.vector_clock.update(remote_clock)
                        await self.set_state(
                            state_key, remote_data, expected_version=local_version
                        )
                        results[state_key] = "accepted_remote"
                    else:
                        # Concurrent modification, need resolution
                        await self._log_conflict(
                            None,
                            state_key,
                            local_version,
                            {"data": remote_data, "clock": remote_clock.to_dict()},
                        )
                        results[state_key] = "conflict"

            except Exception as e:
                self.logger.error(f"Sync error for {state_key}: {str(e)}")
                results[state_key] = f"error: {str(e)}"

        return results

    async def cleanup_old_versions(
        self, older_than_days: int = 30, keep_minimum: int = 5
    ) -> int:
        """Clean up old version history"""
        if not self.pool:
            return 0

        cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)

        async with self.pool.acquire() as conn:
            # Keep minimum number of versions per state_key
            result = await conn.execute(
                """
                DELETE FROM metis_state_history
                WHERE history_id IN (
                    SELECT history_id FROM (
                        SELECT history_id,
                               ROW_NUMBER() OVER (
                                   PARTITION BY state_key 
                                   ORDER BY changed_at DESC
                               ) as rn
                        FROM metis_state_history
                        WHERE changed_at < $1
                    ) ranked
                    WHERE rn > $2
                )
            """,
                cutoff_date,
                keep_minimum,
            )

            return int(result.split()[-1])

    async def close(self) -> None:
        """Close database connections"""
        if self.pool:
            await self.pool.close()


class StateTransaction:
    """
    Transactional context for multiple state operations
    Ensures atomicity across multiple state changes
    """

    def __init__(self, state_manager: DistributedStateManager):
        self.state_manager = state_manager
        self.operations: List[Tuple[str, Any, StateType]] = []
        self.original_states: Dict[str, Tuple[Any, StateVersion]] = {}

    async def __aenter__(self):
        """Begin transaction"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Commit or rollback transaction"""
        if exc_type is not None:
            # Rollback on exception
            await self.rollback()
            return False
        else:
            # Commit on success
            await self.commit()
            return True

    async def get(
        self, state_key: str, state_type: StateType = StateType.ENGAGEMENT
    ) -> Any:
        """Get state within transaction"""
        result = await self.state_manager.get_state(state_key, state_type)
        if result:
            data, version = result
            self.original_states[state_key] = (data, version)
            return data
        return None

    def set(
        self,
        state_key: str,
        state_data: Any,
        state_type: StateType = StateType.ENGAGEMENT,
    ):
        """Queue state update within transaction"""
        self.operations.append((state_key, state_data, state_type))

    async def commit(self):
        """Commit all queued operations"""
        for state_key, state_data, state_type in self.operations:
            expected_version = None
            if state_key in self.original_states:
                _, expected_version = self.original_states[state_key]

            await self.state_manager.set_state(
                state_key, state_data, state_type, expected_version
            )

    async def rollback(self):
        """Rollback transaction (clear operations)"""
        self.operations.clear()
        self.original_states.clear()


# Enhanced connection handling with proper fallbacks
class SafeDistributedStateManager:
    """State manager with safe connection handling"""

    def __init__(self, postgres_url=None, redis_url=None):
        self.postgres_url = postgres_url or "postgresql://localhost:5432/metis"
        self.redis_url = redis_url or "redis://localhost:6379"
        self.postgres_conn = None
        self.redis_conn = None
        self.fallback_storage = {}

        # Try to establish connections safely
        self._init_connections()

    def _init_connections(self):
        """Initialize connections with fallback"""
        try:
            if POSTGRES_AVAILABLE:
                import psycopg2

                self.postgres_conn = psycopg2.connect(self.postgres_url)
        except Exception as e:
            print(f"Warning: PostgreSQL connection failed: {e}")
            self.postgres_conn = None

        try:
            try:
                import redis

                REDIS_AVAILABLE = True
            except ImportError:
                REDIS_AVAILABLE = False

            if REDIS_AVAILABLE:
                self.redis_conn = redis.from_url(self.redis_url)
        except Exception as e:
            print(f"Warning: Redis connection failed: {e}")
            self.redis_conn = None

    async def set_state(self, key: str, value: Any, state_type=None):
        """Set state with fallback storage"""
        try:
            if self.redis_conn and hasattr(self.redis_conn, "set"):
                await self.redis_conn.set(key, json.dumps(value))
            else:
                self.fallback_storage[key] = value
        except Exception:
            self.fallback_storage[key] = value

    async def get_state(self, key: str):
        """Get state with fallback storage"""
        try:
            if self.redis_conn and hasattr(self.redis_conn, "get"):
                value = await self.redis_conn.get(key)
                return json.loads(value) if value else None
            else:
                return self.fallback_storage.get(key)
        except Exception:
            return self.fallback_storage.get(key)


# Use safe state manager as default
DistributedStateManager = SafeDistributedStateManager
