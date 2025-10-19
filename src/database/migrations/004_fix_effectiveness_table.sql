-- Fix: Ensure mental_model_effectiveness is a TABLE (not a VIEW) and add indexes
-- Safe on local/staging. Drops view if it exists, then (re)creates table and indexes.

DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM pg_catalog.pg_views
    WHERE schemaname = current_schema()
      AND viewname = 'mental_model_effectiveness'
  ) THEN
    DROP VIEW mental_model_effectiveness CASCADE;
  END IF;
END$$;

CREATE TABLE IF NOT EXISTS mental_model_effectiveness (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_domain VARCHAR(100) NOT NULL,
    mental_model_name VARCHAR(255) NOT NULL,
    effectiveness_score DOUBLE PRECISION DEFAULT 0.5,
    total_queries INTEGER DEFAULT 0,
    total_citations INTEGER DEFAULT 0,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT uq_mm_effectiveness UNIQUE (query_domain, mental_model_name)
);

CREATE INDEX IF NOT EXISTS idx_mm_eff_domain ON mental_model_effectiveness(query_domain);
CREATE INDEX IF NOT EXISTS idx_mm_eff_updated ON mental_model_effectiveness(last_updated);

COMMENT ON TABLE mental_model_effectiveness IS 'Domain-specific rolling effectiveness metrics for mental models';

