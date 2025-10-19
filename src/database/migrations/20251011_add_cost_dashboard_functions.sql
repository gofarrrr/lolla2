-- Migration: Add SQL Functions for Cost Dashboard Rollups
-- Created: 2025-10-11
-- Purpose: Efficient aggregation queries for cost analytics

-- Function: Get cost by provider
CREATE OR REPLACE FUNCTION get_cost_by_provider(days_back INT DEFAULT 7)
RETURNS TABLE (
  provider TEXT,
  total_cost NUMERIC,
  call_count BIGINT
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    e.provider,
    SUM(e.cost_usd)::NUMERIC AS total_cost,
    COUNT(*)::BIGINT AS call_count
  FROM llm_cost_events e
  WHERE e.ts >= NOW() - (days_back || ' days')::INTERVAL
  GROUP BY e.provider
  ORDER BY total_cost DESC;
END;
$$ LANGUAGE plpgsql;

-- Function: Get cost by phase
CREATE OR REPLACE FUNCTION get_cost_by_phase(days_back INT DEFAULT 7)
RETURNS TABLE (
  phase TEXT,
  total_cost NUMERIC,
  call_count BIGINT
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    e.phase,
    SUM(e.cost_usd)::NUMERIC AS total_cost,
    COUNT(*)::BIGINT AS call_count
  FROM llm_cost_events e
  WHERE e.ts >= NOW() - (days_back || ' days')::INTERVAL
    AND e.phase IS NOT NULL
  GROUP BY e.phase
  ORDER BY total_cost DESC;
END;
$$ LANGUAGE plpgsql;

-- Function: Get reasoning mode mix
CREATE OR REPLACE FUNCTION get_reasoning_mix(days_back INT DEFAULT 7)
RETURNS TABLE (
  total_calls BIGINT,
  reasoning_enabled_count BIGINT,
  reasoning_enabled_pct NUMERIC
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    COUNT(*)::BIGINT AS total_calls,
    COUNT(*) FILTER (WHERE e.reasoning_enabled = TRUE)::BIGINT AS reasoning_enabled_count,
    ROUND(
      (COUNT(*) FILTER (WHERE e.reasoning_enabled = TRUE)::NUMERIC / COUNT(*)::NUMERIC * 100),
      1
    ) AS reasoning_enabled_pct
  FROM llm_cost_events e
  WHERE e.ts >= NOW() - (days_back || ' days')::INTERVAL;
END;
$$ LANGUAGE plpgsql;

-- Function: Get daily cost series
CREATE OR REPLACE FUNCTION get_daily_cost_series(days_back INT DEFAULT 7)
RETURNS TABLE (
  date DATE,
  total_cost NUMERIC,
  call_count BIGINT
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    e.ts::DATE AS date,
    SUM(e.cost_usd)::NUMERIC AS total_cost,
    COUNT(*)::BIGINT AS call_count
  FROM llm_cost_events e
  WHERE e.ts >= NOW() - (days_back || ' days')::INTERVAL
  GROUP BY e.ts::DATE
  ORDER BY date ASC;
END;
$$ LANGUAGE plpgsql;

-- Function: Get provider health metrics
CREATE OR REPLACE FUNCTION get_provider_health(days_back INT DEFAULT 7)
RETURNS TABLE (
  provider TEXT,
  success_rate NUMERIC,
  avg_latency_ms NUMERIC,
  total_calls BIGINT,
  total_cost NUMERIC
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    e.provider,
    ROUND(
      (COUNT(*) FILTER (WHERE e.success = TRUE)::NUMERIC / COUNT(*)::NUMERIC),
      4
    ) AS success_rate,
    ROUND(AVG(e.latency_ms)::NUMERIC, 2) AS avg_latency_ms,
    COUNT(*)::BIGINT AS total_calls,
    SUM(e.cost_usd)::NUMERIC AS total_cost
  FROM llm_cost_events e
  WHERE e.ts >= NOW() - (days_back || ' days')::INTERVAL
  GROUP BY e.provider
  ORDER BY total_cost DESC;
END;
$$ LANGUAGE plpgsql;

-- Comments
COMMENT ON FUNCTION get_cost_by_provider IS 'Aggregate cost and call count by provider';
COMMENT ON FUNCTION get_cost_by_phase IS 'Aggregate cost and call count by pipeline phase';
COMMENT ON FUNCTION get_reasoning_mix IS 'Calculate reasoning mode utilization percentage';
COMMENT ON FUNCTION get_daily_cost_series IS 'Daily cost time series for charting';
COMMENT ON FUNCTION get_provider_health IS 'Provider health metrics (success rate, latency, cost)';
