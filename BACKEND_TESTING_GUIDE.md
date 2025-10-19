# METIS V5.3 Backend Testing Guide

## Overview
This guide provides step-by-step instructions for testing the METIS V5.3 backend platform before frontend integration.

## Prerequisites

### 1. Environment Setup
- Python 3.13.7 installed ✅
- Core dependencies installed (FastAPI, Uvicorn, Anthropic) ✅
- `.env` file exists ✅

### 2. Required API Keys
Configure these in your `.env` file:

**Required (choose at least one)**:
- `ANTHROPIC_API_KEY` - For Claude-based processing
- `DEEPSEEK_API_KEY` - For cost-effective DeepSeek processing

**Optional**:
- `OPENROUTER_API_KEY` - For OpenRouter fallback
- `PERPLEXITY_API_KEY` - For enhanced research (if enabled)
- `SUPABASE_*` - For database features (can test without)

### 3. Verify Dependencies
```bash
# Check installed packages
pip3 list | grep -E "(fastapi|uvicorn|anthropic)"

# If missing, install from requirements
pip3 install -r requirements.txt
```

## Testing Phases

### Phase 1: Backend Startup & Health Checks

#### Step 1: Start the Backend
```bash
# From project root
python3 src/main.py

# Expected output:
# 📝 File logging enabled → ./backend_live.log
# INFO:     Started server process [pid]
# INFO:     Waiting for application startup.
# INFO:     Application startup complete.
# INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

#### Step 2: Test Health Endpoint
```bash
# In a new terminal
curl http://localhost:8000/api/v53/health

# Expected response:
# {"status":"healthy","timestamp":"2025-10-19T...","version":"5.3"}
```

#### Step 3: Test System Status
```bash
curl http://localhost:8000/api/v53/system-status

# Expected response with service cluster info
```

### Phase 2: Core API Endpoint Testing

#### Test 1: Progressive Questions Generation
```bash
# Test the progressive questions API
curl -X POST http://localhost:8000/api/progressive-questions/generate \
  -H "Content-Type: application/json" \
  -d '{
    "business_context": "A SaaS company considering pricing model changes",
    "initial_question": "Should we switch from monthly to annual billing?",
    "depth": 2
  }'

# Expected: JSON response with generated strategic questions
```

#### Test 2: Analysis Execution (V5.3)
```bash
# Test stateful analysis endpoint
curl -X POST http://localhost:8000/api/v53/analysis/execute \
  -H "Content-Type: application/json" \
  -d '{
    "problem": "Evaluate market entry strategy for electric vehicles in Southeast Asia",
    "context": "Mid-sized automotive manufacturer with hybrid vehicle expertise"
  }'

# Expected: JSON response with analysis_id and initial results
```

#### Test 3: Get Analysis Status
```bash
# Replace {analysis_id} with actual ID from previous test
curl http://localhost:8000/api/v53/analysis/{analysis_id}/status

# Expected: Status information with progress indicators
```

### Phase 3: LLM Provider Integration Testing

#### Test 1: Verify Provider Fallback Chain
```bash
# Check logs for provider selection
tail -f backend_live.log | grep -E "(DeepSeek|Anthropic|OpenRouter)"

# Run a test query and observe which provider is used
```

#### Test 2: Test Circuit Breaker
```bash
# Temporarily disable DeepSeek (comment out key in .env)
# Restart backend and run a test - should fallback to Anthropic

# Check logs for circuit breaker messages
grep -i "circuit" backend_live.log
```

#### Test 3: Check Resiliency Metrics
```bash
# Use the observability CLI tool
python3 scripts/summarize_llm_attempts.py backend_live.log

# Expected: Summary of provider attempts, retries, fallbacks
```

### Phase 4: WebSocket Testing (Real-time Updates)

#### Test 1: WebSocket Connection
```bash
# Install wscat if needed: npm install -g wscat
wscat -c ws://localhost:8000/ws

# Expected: Connection established
# Send test message and receive response
```

#### Test 2: Analysis Stream
```python
# Python WebSocket test client
import asyncio
import websockets
import json

async def test_ws():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        # Subscribe to analysis updates
        await websocket.send(json.dumps({
            "action": "subscribe",
            "analysis_id": "test-123"
        }))

        # Receive updates
        while True:
            message = await websocket.recv()
            print(f"Received: {message}")

asyncio.run(test_ws())
```

### Phase 5: API Endpoint Inventory

**V5.3 Core Endpoints**:
- `GET /api/v53/health` - Health check
- `GET /api/v53/system-status` - System status with service clusters
- `POST /api/v53/analysis/execute` - Execute stateful analysis
- `GET /api/v53/analysis/{id}/status` - Get analysis status
- `GET /api/v53/analysis/{id}` - Get full analysis results

**Progressive Questions**:
- `POST /api/progressive-questions/generate` - Generate strategic questions
- `POST /api/progressive-questions/enhance` - Enhance existing questions

**Legacy Endpoints (being migrated)**:
- `POST /api/enhanced_foundation/*` - Foundation services
- `POST /api/analysis_execution/*` - Legacy analysis execution

**WebSocket**:
- `WS /ws` - Real-time analysis updates

## Testing Checklist

- [ ] Backend starts without errors
- [ ] Health endpoint returns 200
- [ ] System status shows service clusters
- [ ] Progressive questions generates results
- [ ] Analysis execution creates analysis_id
- [ ] Analysis status returns progress
- [ ] LLM provider connects successfully
- [ ] Provider fallback works (if primary fails)
- [ ] Circuit breaker activates on failures
- [ ] WebSocket connection establishes
- [ ] Real-time updates stream correctly
- [ ] Logs show structured LLM attempts

## Common Issues & Solutions

### Issue: Backend fails to start
**Solution**: Check `backend_live.log` for detailed error messages

### Issue: API returns 500 errors
**Solution**:
1. Verify API keys in `.env`
2. Check logs for missing dependencies
3. Ensure database connections (if using Supabase)

### Issue: LLM requests timeout
**Solution**:
1. Check `METIS_LLM_TIMEOUT_SECONDS` in `.env` (default: 45s)
2. Increase timeout or check network connectivity
3. Verify API keys are valid

### Issue: WebSocket won't connect
**Solution**:
1. Check CORS settings in `src/main.py`
2. Verify port 8000 is not blocked
3. Test with wscat or simple Python client

## Next Steps

After successful backend testing:
1. Document any issues encountered
2. Verify all critical paths work
3. Proceed to frontend integration testing
4. Set up monitoring/observability for production

## Quick Test Script

```bash
#!/bin/bash
# Save as test_backend.sh

echo "🧪 Testing METIS V5.3 Backend..."

echo "1. Health Check"
curl -s http://localhost:8000/api/v53/health | jq .

echo "\n2. System Status"
curl -s http://localhost:8000/api/v53/system-status | jq .status

echo "\n3. Progressive Questions"
curl -s -X POST http://localhost:8000/api/progressive-questions/generate \
  -H "Content-Type: application/json" \
  -d '{"business_context":"Test","initial_question":"Test?","depth":1}' \
  | jq '.questions | length'

echo "\n✅ Basic tests complete"
```

Make executable: `chmod +x test_backend.sh`
Run: `./test_backend.sh`
