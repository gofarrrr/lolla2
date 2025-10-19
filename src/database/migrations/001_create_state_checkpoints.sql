-- Migration: Create state_checkpoints table for persistent checkpoint storage
-- Date: 2025-01-XX
-- Description: Replaces in-memory checkpoint storage with database-backed persistence

CREATE TABLE IF NOT EXISTS state_checkpoints (
    -- Primary identifier
    checkpoint_id UUID PRIMARY KEY,

    -- Pipeline tracking
    trace_id UUID NOT NULL,
    stage_name TEXT NOT NULL,

    -- Full checkpoint state (JSONB for flexibility)
    state_data JSONB NOT NULL,

    -- Metadata
    user_id UUID,
    session_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for fast checkpoint retrieval by checkpoint_id
CREATE INDEX IF NOT EXISTS idx_checkpoints_checkpoint_id ON state_checkpoints(checkpoint_id);

-- Index for retrieving all checkpoints for a given trace
CREATE INDEX IF NOT EXISTS idx_checkpoints_trace_id ON state_checkpoints(trace_id);

-- Index for retrieving checkpoints by stage
CREATE INDEX IF NOT EXISTS idx_checkpoints_stage ON state_checkpoints(stage_name);

-- Index for time-based queries
CREATE INDEX IF NOT EXISTS idx_checkpoints_created_at ON state_checkpoints(created_at);

-- Comments for documentation
COMMENT ON TABLE state_checkpoints IS 'Persistent storage for pipeline execution checkpoints, enabling resume after crashes or restarts';
COMMENT ON COLUMN state_checkpoints.state_data IS 'Full serialized StateCheckpoint object as JSONB for maximum flexibility';
COMMENT ON COLUMN state_checkpoints.stage_name IS 'Pipeline stage name (e.g., "Socratic Questions", "Problem Structuring")';
