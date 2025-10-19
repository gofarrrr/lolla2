-- Migration: Depth Enrichment Tables (Q&A precision path)
-- Date: 2025-10-06
-- Description: Adds relational tables for precise mental model Q&A retrieval

-- Note: gen_random_uuid() requires pgcrypto; Supabase projects usually have it enabled by default.
-- Uncomment if needed in your environment:
-- CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- 1) Canonical Q&A chunks per mental model
CREATE TABLE IF NOT EXISTS mental_model_qa_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mental_model_name VARCHAR(255) NOT NULL,     -- canonical snake_case name
    source_file VARCHAR(512) NOT NULL,           -- source md path
    question_num INTEGER,                        -- 1..5 (if known)
    question_type VARCHAR(32),                   -- foundational|evidence|practical|pitfalls|implementation (optional)
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    domain VARCHAR(100),                         -- optional domain tagging (e.g., customer_retention)
    category VARCHAR(100),                       -- e.g., DECISION|PERCEPTION|...
    nway_group VARCHAR(50),                      -- NWAY|NWAY2|MM1|MM2
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Deterministic unique keys (NULLs do not conflict in PG unique indexes)
CREATE UNIQUE INDEX IF NOT EXISTS uq_mmqa_by_num ON mental_model_qa_chunks(mental_model_name, question_num);
CREATE UNIQUE INDEX IF NOT EXISTS uq_mmqa_by_type ON mental_model_qa_chunks(mental_model_name, question_type);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_mmqa_model ON mental_model_qa_chunks(mental_model_name);
CREATE INDEX IF NOT EXISTS idx_mmqa_domain ON mental_model_qa_chunks(domain);
CREATE INDEX IF NOT EXISTS idx_mmqa_category ON mental_model_qa_chunks(category);
CREATE INDEX IF NOT EXISTS idx_mmqa_created_at ON mental_model_qa_chunks(created_at);

COMMENT ON TABLE mental_model_qa_chunks IS 'Canonical, direct-lookup Q&A units for depth enrichment';

-- 2) Effectiveness tracking (EMA over time per domain x model)
CREATE TABLE IF NOT EXISTS mental_model_effectiveness (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_domain VARCHAR(100) NOT NULL,
    mental_model_name VARCHAR(255) NOT NULL,
    effectiveness_score DOUBLE PRECISION DEFAULT 0.5,  -- 0..1
    total_queries INTEGER DEFAULT 0,
    total_citations INTEGER DEFAULT 0,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT uq_mm_effectiveness UNIQUE (query_domain, mental_model_name)
);

CREATE INDEX IF NOT EXISTS idx_mm_eff_domain ON mental_model_effectiveness(query_domain);
CREATE INDEX IF NOT EXISTS idx_mm_eff_updated ON mental_model_effectiveness(last_updated);

COMMENT ON TABLE mental_model_effectiveness IS 'Domain-specific rolling effectiveness metrics for mental models';

-- 3) Usage log for audit and learning
CREATE TABLE IF NOT EXISTS mental_model_usage_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trace_id UUID,
    session_id UUID,
    consultant_id TEXT,
    mental_model_name VARCHAR(255) NOT NULL,
    question_type VARCHAR(32),
    question_num INTEGER,
    was_injected BOOLEAN DEFAULT FALSE,          -- depth enrichment injected content
    usage_stage VARCHAR(64),                     -- e.g., post_consultant, post_critique
    citation_detected BOOLEAN DEFAULT FALSE,     -- whether it was cited downstream
    latency_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_mm_usage_model ON mental_model_usage_log(mental_model_name);
CREATE INDEX IF NOT EXISTS idx_mm_usage_trace ON mental_model_usage_log(trace_id);
CREATE INDEX IF NOT EXISTS idx_mm_usage_created ON mental_model_usage_log(created_at);

COMMENT ON TABLE mental_model_usage_log IS 'Audit log for usage of mental models and injected Q&A content';

