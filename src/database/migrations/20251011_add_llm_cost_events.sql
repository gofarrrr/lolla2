-- Migration: Add LLM Cost Events Table for Cost Dashboard
-- Created: 2025-10-11
-- Purpose: Track all LLM API calls for cost monitoring and optimization

CREATE TABLE IF NOT EXISTS llm_cost_events (
  id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  engagement_id TEXT,
  phase TEXT,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  tokens_input INT,
  tokens_output INT,
  cost_usd NUMERIC(12,6) NOT NULL,
  latency_ms INT,
  reasoning_enabled BOOLEAN DEFAULT FALSE,
  success BOOLEAN DEFAULT TRUE,
  error_message TEXT,
  request_metadata JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS llm_cost_events_ts_idx ON llm_cost_events(ts DESC);
CREATE INDEX IF NOT EXISTS llm_cost_events_provider_idx ON llm_cost_events(provider);
CREATE INDEX IF NOT EXISTS llm_cost_events_phase_idx ON llm_cost_events(phase);
CREATE INDEX IF NOT EXISTS llm_cost_events_engagement_idx ON llm_cost_events(engagement_id);
CREATE INDEX IF NOT EXISTS llm_cost_events_success_idx ON llm_cost_events(success);

-- Composite index for cost rollups by provider and phase
CREATE INDEX IF NOT EXISTS llm_cost_events_provider_phase_ts_idx
  ON llm_cost_events(provider, phase, ts DESC);

-- Comments for documentation
COMMENT ON TABLE llm_cost_events IS 'Tracks all LLM API calls for cost monitoring and performance analysis';
COMMENT ON COLUMN llm_cost_events.ts IS 'Timestamp of LLM call';
COMMENT ON COLUMN llm_cost_events.engagement_id IS 'Optional engagement ID for multi-turn conversations';
COMMENT ON COLUMN llm_cost_events.phase IS 'Pipeline phase (e.g., hypothesis_generation, analysis_execution)';
COMMENT ON COLUMN llm_cost_events.provider IS 'LLM provider (openrouter, deepseek, anthropic)';
COMMENT ON COLUMN llm_cost_events.model IS 'Model used (grok-4-fast, deepseek-reasoner, claude-3-5-sonnet-20241022)';
COMMENT ON COLUMN llm_cost_events.reasoning_enabled IS 'Whether reasoning mode was enabled for this call';
COMMENT ON COLUMN llm_cost_events.request_metadata IS 'Additional request metadata (task_type, stakeholder_impact, etc.)';
