# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Lolla V1.0 is a production-ready cognitive intelligence platform implementing the V5.3 Canonical Standard. It consists of:

- **Backend**: Python-based METIS V5.3 platform with service-oriented architecture
- **Frontend**: Next.js 14 TypeScript application (TreeGlav)
- **Architecture**: 20 specialized services across 4 clusters with resilient manager patterns

## Development Commands

### Backend (Python)
```bash
# Start the V5.3 platform
python3 src/main.py

# Install dependencies
pip install -r requirements-v2.txt

# Run tests using Makefile
make test                    # All tests (unit + integration + e2e)
make test-unit              # Unit tests only
make test-integration       # Integration tests
make test-e2e              # End-to-end tests
make test-performance      # Performance tests (with benchmarking)
make test-security         # Security tests (bandit + safety)
make test-agentic          # Agentic pattern tests
make test-quick            # Quick tests (unit + critical integration)
make test-full             # Full test suite with coverage report

# Quality assurance
python3 evaluation/cqa_v2_integration.py

# Code quality
make lint                  # Lint with ruff, black, mypy
make format               # Format with black and ruff
make check                # Lint + unit + integration tests

# Development workflow (IMPORTANT!)
make clean-cache          # Clear Python bytecode cache
make dev-restart          # Kill backend + clear cache (use when debugging)
```

**⚠️ CRITICAL: Python Bytecode Cache Issue**

Python caches compiled bytecode in `__pycache__/` directories. This can mask code changes during development, causing:
- Debug logs not appearing even though code was modified
- Code changes not taking effect after backend restart
- Confusing "impossible" execution states

**Solution: Always use `make dev-restart` when debugging backend issues!**

This command:
1. Kills any backend process on port 8000
2. Clears all Python bytecode cache (`__pycache__/` and `.pyc` files)
3. Ensures next backend start loads fresh code

**When to use:**
- After adding debug logging and logs don't appear
- After modifying orchestrator/service code
- When code changes seem to have no effect
- During any deep debugging session

### Frontend (Next.js)
```bash
cd frontend

# Development
npm install
npm run dev                # Start dev server
npm run build             # Production build
npm run start             # Start production server

# Quality
npm run lint              # ESLint
npm run type-check        # TypeScript checking

# Testing
npm run test:e2e          # Playwright E2E tests
```

## Architecture Overview

### V5.3 Canonical Standard
The platform implements a service-oriented architecture with:

1. **Single Entry Point**: `src/main.py` with clean dependency injection
2. **20 Specialized Services** across 4 clusters:
   - Reliability Services Cluster (6 services)
   - Selection Services Cluster (6 services) 
   - Application Services Cluster (5 services)
   - Integration Services Cluster (3 services)
3. **Resilient Manager Pattern**: Multi-provider LLM and research managers
4. **Stateful Pipeline Orchestrator**: Checkpoint-based analysis refinement
5. **Glass-Box V4 UnifiedContextStream**: Complete operational transparency
6. **Agentic ULTRATHINK Engine**: Enhanced devil's advocate system

### Key Components
- **Backend API**: FastAPI with CORS enabled, runs on port 8000
- **Frontend**: Next.js with TypeScript, runs on port 3000
- **Database**: Supabase PostgreSQL integration
- **WebSocket**: Real-time communication for analysis updates
- **Testing**: Comprehensive test suite with Playwright E2E, pytest backend

### Service Architecture
Services are organized in `src/services/` with clean boundaries:
- `/reliability/` - Failure detection, validation, feedback orchestration
- `/selection/` - Model selection, provider management  
- `/application/` - Core application logic, iteration management
- `/integration/` - External integrations, research providers

### Frontend Structure
- **App Router**: Next.js 14 app directory structure
- **Components**: Reusable UI components with Tailwind CSS
- **API Integration**: Axios for backend communication
- **State Management**: React Query for server state
- **Real-time**: Socket.io client for WebSocket connections

## Configuration

### Environment Variables
- Backend: Copy `.env.example` to `.env` and configure API keys (OpenRouter, DeepSeek, Anthropic, Perplexity)
- Frontend: Configure `METIS_API_URL` and `METIS_WS_URL` in `frontend/.env.local`

### Provider Policy (2025-10-11)
The system uses a centralized provider policy for LLM routing:
- **ORACLE Flow** (Strategic reasoning): OpenRouter → Anthropic → DeepSeek
- **GENERAL Flow** (Cost-optimized): DeepSeek → Anthropic → OpenRouter

Policy is defined in `src/engine/services/llm/provider_policy.py` and automatically applied based on pipeline phase.

See `ARCHITECT_FIXES_IMPLEMENTATION_SUMMARY.md` for complete implementation details.

### Development Setup
```bash
make dev-setup              # Complete development environment setup
make install-test-deps      # Install testing dependencies only
```

### API Endpoints
- Health Check: `GET /api/v53/health`
- System Status: `GET /api/v53/system-status`  
- Analysis APIs: Various endpoints under `/api/`

## Testing Strategy

The project uses comprehensive testing with clear markers:
- `@pytest.mark.unit` - Fast, isolated tests
- `@pytest.mark.integration` - Service interaction tests
- `@pytest.mark.e2e` - Full system tests
- `@pytest.mark.performance` - Benchmark tests
- `@pytest.mark.security` - Security validation tests

Frontend uses Playwright for E2E testing with comprehensive test coverage.

## Development Notes

- **Code Quality**: Both backend and frontend have strict linting and type checking
- **Service Integration**: All services use dependency injection pattern
- **Error Handling**: Comprehensive error handling with detailed logging
- **Security**: CSP headers configured, PII protection implemented
- **Performance**: DeepSeek-first strategy provides 585% cost savings vs Claude
- **Monitoring**: Glass-box transparency with full operation auditability

## Cognitive Core & Walking Skeleton

The project includes a walking skeleton for the Cognitive Core (V2):

### Key Components
- **CoreOps DSL**: `src/services/coreops_dsl.py` - YAML-based operation definition
- **Evidence Manager**: `src/services/evidence_manager.py` - Content fingerprinting and provenance
- **Cognitive Core Service**: `src/services/cognitive_core_service.py` - Main service orchestrator
- **Example**: `examples/coreops/sample_coreops.yaml` - Sample CoreOps configuration

### Testing the Walking Skeleton
```bash
pytest tests/test_cognitive_core_walking_skeleton.py -v
```

## Method Actor Devils Advocate System

The platform includes an advanced Method Actor Devils Advocate system that combines algorithmic reliability with engaging Method Actor personas for enhanced critical analysis.

### Architecture

**Hybrid Approach**: Combines proven algorithmic engines with Method Actor personas:
- **Algorithmic Foundation**: Systematic bias detection and assumption analysis
- **Method Actor Personas**: Charlie Munger and Russell Ackoff personas for engaging communication
- **Forward Motion Converter**: Transforms challenges into actionable experiments and guardrails
- **Anti-Failure Safeguards**: Prevents gotcha-ism, naysaying, and psychological safety issues

### Key Components

#### 1. Method Actor Devils Advocate (`src/core/method_actor_devils_advocate.py`)
Main implementation combining:
- **Charlie Munger Persona**: Investment wisdom, pattern recognition, inversion thinking
- **Russell Ackoff Persona**: Systems thinking, assumption dissolution, idealized design
- **Forward Motion Converter**: Challenges → experiments/guardrails conversion
- **Tone Safeguards**: Research-validated enabling challenger patterns

#### 2. Configuration (`cognitive_architecture/NWAY_DEVILS_ADVOCATE_001.yaml`)
- **Persona Definitions**: Detailed character descriptions and communication patterns
- **Thin Variables**: Fine-grained control parameters (persona strength, challenge depth)
- **S2 Tier Integration**: Automatic tier classification and escalation triggers
- **Anti-Failure Settings**: Gotcha prevention, psychological safety maintenance

#### 3. Integration Points
- **Enhanced Devils Advocate System**: Optional Method Actor mode with fallback
- **UnifiedContextStream**: Enhanced evidence events for glass-box transparency
- **StatefulPipelineOrchestrator**: Stage-based integration with checkpoint support

### Configuration & Usage

#### Environment Variables
```bash
# Enable Method Actor Devils Advocate
METHOD_ACTOR_DA_ENABLED=true

# Optional: Adjust persona strength (0.0-1.0)
PERSONA_STRENGTH=0.8

# Optional: Set challenge depth (0.0-1.0)
CHALLENGE_DEPTH=0.7
```

#### Key Features

1. **Enabling Challenger Communication**:
   - Always starts with vulnerability ("I've made this mistake myself...")
   - Provides historical analogies and pattern recognition
   - Ends with openness ("What am I missing?")
   - Attacks ideas rigorously while supporting people warmly

2. **Forward Motion Generation**:
   - **Experiments**: Every challenge converts to testable experiments
   - **Guardrails**: Early warning systems and monitoring mechanisms
   - **Reversible Steps**: Low-risk, easily reversible actions preferred
   - **Premortem Scenarios**: "What could go wrong" analysis

3. **Research-Validated Safeguards**:
   - High gotcha prevention (0.9/1.0) to avoid personal attacks
   - Psychological safety maintenance (0.95/1.0)
   - Solution suggestion ratio (0.8/1.0) - always suggest fixes
   - Anti-nihilism protection ensures forward motion

#### Evidence Events

The system generates enhanced evidence events for complete transparency:
- `DEVILS_ADVOCATE_METHOD_ACTOR_COMPLETE`: Full Method Actor analysis complete
- `ENABLING_CHALLENGER_DIALOGUE_GENERATED`: Enabling challenger communication created
- `FORWARD_MOTION_ACTIONS_CREATED`: Experiments and guardrails generated
- `ANTI_FAILURE_SAFEGUARDS_ACTIVATED`: Safety mechanisms engaged

### Integration with Existing System

The Method Actor Devils Advocate integrates seamlessly with the existing enhanced devils advocate system:

```python
# Enhanced Devils Advocate System automatically detects and uses Method Actor mode
from src.core.enhanced_devils_advocate_system import EnhancedDevilsAdvocateSystem

devils_advocate = EnhancedDevilsAdvocateSystem()
result = await devils_advocate.run_enhanced_critique(
    analysis="...",
    context=context,
    method_actor_mode=True  # Optional explicit activation
)
```

### S2 Tier Classification

The system integrates with S2 Kernel tier classification:
- **Tier 1**: Quick algorithmic check with minimal Munger persona
- **Tier 2**: Balanced algorithmic + moderate persona (both Munger and Ackoff)
- **Tier 3**: Full Method Actor dialogue with experiments (high-stakes decisions)

### Quality Metrics

The system tracks research-validated metrics:
- **Enabling Challenger Score**: Measures constructive vs obstructionist behavior
- **Forward Motion Conversion Rate**: Challenge-to-action conversion success
- **Psychological Safety Score**: Maintains stakeholder comfort and openness
- **Anti-Failure Measures**: Prevents gotcha-ism and nihilistic patterns