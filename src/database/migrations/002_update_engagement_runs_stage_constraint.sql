-- Migration: Update engagement_runs stage number constraint for 10-stage pipeline
-- Safe to run multiple times; drops old constraint if present, adds new one.

ALTER TABLE IF EXISTS engagement_runs
    DROP CONSTRAINT IF EXISTS valid_stage_number;

-- Allow stage_number to range from 1 to 10 (including Arbitration/Capture),
-- and ensure progress percentage is within 0..100.
ALTER TABLE engagement_runs
    ADD CONSTRAINT valid_stage_number_v2 CHECK (
        stage_number >= 1 AND stage_number <= 10 AND
        progress_percentage >= 0 AND progress_percentage <= 100
    );

-- Optional: If you want to reflect the UI's total_stages in the row, ensure total_stages matches 10 when full pipeline is enabled.
-- UPDATE engagement_runs SET total_stages = 10 WHERE total_stages IS NULL OR total_stages < 10;

