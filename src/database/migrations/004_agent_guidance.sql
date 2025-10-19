-- Migration: Agent Guidance table for NWAY2 role guidance
-- Description: Stores agent-specific performance guidance extracted from NWAY2 files

CREATE TABLE IF NOT EXISTS agent_guidance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_role VARCHAR(50) NOT NULL,
    guidance_type VARCHAR(50),
    question_num INTEGER,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    content_hash VARCHAR(64) UNIQUE NOT NULL,
    source_file VARCHAR(512) NOT NULL,
    word_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_guidance_role ON agent_guidance(agent_role);
CREATE INDEX IF NOT EXISTS idx_agent_guidance_type ON agent_guidance(guidance_type);
CREATE INDEX IF NOT EXISTS idx_agent_guidance_role_type ON agent_guidance(agent_role, guidance_type);
CREATE INDEX IF NOT EXISTS idx_agent_guidance_created_at ON agent_guidance(created_at);

COMMENT ON TABLE agent_guidance IS 'NWAY2-derived agent performance guidance for Devil''s Advocate, Senior Advisor, etc.';
COMMENT ON COLUMN agent_guidance.agent_role IS 'Agent identifier (e.g., devils_advocate, senior_advisor)';
COMMENT ON COLUMN agent_guidance.guidance_type IS 'Guidance category: behaviors | communication | frameworks | tools';
