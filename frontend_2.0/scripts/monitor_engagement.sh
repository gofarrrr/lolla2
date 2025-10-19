#!/usr/bin/env bash
set -euo pipefail

BACKEND_LOG="/Users/marcin/lolla_v6/backend_uvicorn.log"
FRONT_SCRIPTS_DIR="/Users/marcin/lolla_v6/frontend_2.0/scripts"
OUT_DIR="/Users/marcin/lolla_v6/frontend_2.0"
API_BASE="http://localhost:8000"
FRONT_BASE="http://localhost:3001"

if [ ! -f "$BACKEND_LOG" ]; then
  echo "Backend log not found at $BACKEND_LOG. Ensure backend is running via uvicorn src.main:app." >&2
  exit 1
fi

# Function to poll status until completion
poll_status() {
  local trace_id="$1"
  echo "Monitoring trace $trace_id..."
  while true; do
    local jsn
    jsn=$(curl -s "$API_BASE/api/engagements/$trace_id/status" || true)
    if [ -z "$jsn" ]; then
      echo "No status response yet..."; sleep 3; continue
    fi
    local status
    status=$(echo "$jsn" | jq -r .status)
    local prog
    prog=$(echo "$jsn" | jq -r .progress_percentage)
    echo "[$(date +%H:%M:%S)] status=$status progress=${prog}%"
    if [ "$status" = "COMPLETED" ]; then
      return 0
    elif [ "$status" = "FAILED" ]; then
      echo "Run FAILED:"; echo "$jsn" | jq '.'
      return 2
    fi
    sleep 5
  done
}

# Tail backend log and find the next trace id
echo "Waiting for a new engagement to start (monitoring $BACKEND_LOG)..."
TRACE_ID=""
# Use tail -n0 to start from end, -F to follow even if file is rotated
while read -r line; do
  if echo "$line" | grep -q "Starting new engagement - Trace ID:"; then
    # Extract UUID-like token
    TRACE_ID=$(echo "$line" | grep -Eo '[0-9a-fA-F-]{36}' | tail -n1)
    if [ -n "$TRACE_ID" ]; then
      echo "Detected new trace: $TRACE_ID"
      break
    fi
  fi
done < <(tail -n0 -F "$BACKEND_LOG")

# Poll status until completion
if poll_status "$TRACE_ID"; then
  echo "Run completed: $TRACE_ID"
  # Take screenshots for Evidence and Strategy
  echo "Capturing Evidence and Strategy screenshots..."
  TRACE_ID="$TRACE_ID" OUT_PATH="$OUT_DIR/evidence_${TRACE_ID}.png" node "$FRONT_SCRIPTS_DIR/screenshot_evidence.js" || true
  TRACE_ID="$TRACE_ID" OUT_PATH="$OUT_DIR/strategy_${TRACE_ID}.png" node "$FRONT_SCRIPTS_DIR/screenshot_strategy.js" || true
  echo "Screenshots saved to:"
  echo "  $OUT_DIR/evidence_${TRACE_ID}.png"
  echo "  $OUT_DIR/strategy_${TRACE_ID}.png"
  echo "Dashboard URLs:"
  echo "  $FRONT_BASE/analysis/${TRACE_ID}/dashboard"
  echo "  $FRONT_BASE/analysis/${TRACE_ID}/report_v2"
else
  echo "Run ended without completion for trace $TRACE_ID"
fi
