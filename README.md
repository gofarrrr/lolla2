# Lolla V1.0 - METIS V5.3 Cognitive Intelligence Platform

**Production-ready cognitive intelligence platform implementing the V5.3 Canonical Standard**

[![Architecture Tests](https://img.shields.io/badge/architecture-guardrails%20active-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.13-blue)]()
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()

---

## Overview

Lolla V1.0 is a production-ready cognitive intelligence platform featuring:

- **Backend**: Python-based METIS V5.3 with 25 specialized services across 4 clusters
- **Frontend**: Next.js 14 TypeScript application (TreeGlav)
- **Architecture**: Service-oriented design with resilient manager patterns
- **LLM Integration**: DeepSeek V3.1 (primary) with Claude 3.5 Sonnet fallback
- **Testing**: Comprehensive test suite (unit, integration, E2E, architecture)

---

## Quick Start

### Backend Setup

```bash
# Install dependencies
pip install -r requirements-v2.txt

# Run the platform
python3 src/main.py

# Run tests
make test                    # All tests
make test-architecture       # Architecture guardrails
make test-unit              # Unit tests only
make test-integration       # Integration tests
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Development server
npm run dev

# Production build
npm run build && npm run start
```

---

## Architecture

### Service Clusters (25 services)

1. **Reliability Services** (6 services)
   - Failure detection, validation, feedback orchestration

2. **Selection Services** (6 services)
   - Model selection, provider management

3. **Application Services** (5 services)
   - Core application logic, iteration management

4. **Integration Services** (3 services)
   - External integrations, research providers

### Architecture Guardrails 🛡️

The platform includes automated architecture tests to prevent:
- Cross-layer import violations (baseline: 146 files)
- main.py complexity growth (budget: 850 LOC, 70 complexity)
- Interface package availability

Run: `make test-architecture`

See: [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md) for details

---

## Development Workflow

### Critical: Bytecode Cache Issue ⚠️

Python caches compiled bytecode which can mask code changes. **Always run**:

```bash
make dev-restart  # Kills backend + clears cache
```

**When to use**:
- After modifying orchestrator/service code
- When debug logs don't appear
- Before investigating "weird" behavior
- After pulling latest changes

### Running Tests

```bash
make test                  # All tests (unit + integration + e2e)
make test-architecture     # Architecture guardrails
make test-unit            # Unit tests only
make test-integration     # Integration tests
make test-e2e            # End-to-end tests
make test-quick          # Quick tests (unit + critical integration)
```

### Code Quality

```bash
make lint      # Lint with ruff, black, mypy
make format    # Format with black and ruff
make check     # Lint + unit + integration tests
```

- Legacy API sunset in flight → consult [DEPRECATION_PLAN.md](DEPRECATION_PLAN.md) before touching files under `src/engine/api/`.
- Observability helpers: `scripts/summarize_llm_attempts.py --log backend_live.log` summarises LLM fallback metrics from structured logs.

---

## Recent Achievements (Operation Lean)

### Completed Refactoring (4/5 targets) ✅

1. **unified_client.py** - 19% LOC reduction, CC=81 → CC=5
2. **main.py** - 42% LOC reduction (1384 → 804 LOC)
3. **method_actor_devils_advocate.py** - 42% LOC reduction (1160 → 678 LOC)
4. **data_contracts.py** - 44% LOC reduction (1523 → 850 LOC across 6 modules)

See: [LEAN_ROADMAP.md](LEAN_ROADMAP.md) for details

### Architecture Guardrails (2025-10-20) ✅

- 3 automated architecture tests preventing regression
- 15 backup/pilot files removed (clean codebase)
- 6 stage executors standardized
- CI integration ready (`make test-architecture`)

See: [ARCHITECTURE_WORK_COMPLETE.md](ARCHITECTURE_WORK_COMPLETE.md)

---

## API Endpoints

### Health & Status
- `GET /api/v53/health` - Health check
- `GET /api/v53/system-status` - System status

### Analysis
- `POST /api/v53/analyze` - Cognitive analysis
- `GET /api/v53/confidence/*` - Confidence metrics

### Legacy API (Deprecation Planned)
See: [DEPRECATION_PLAN.md](DEPRECATION_PLAN.md) for migration timeline

---

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# LLM Providers
OPENROUTER_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
PERPLEXITY_API_KEY=your_key_here

# Backend
METIS_API_URL=http://localhost:8000
METIS_WS_URL=ws://localhost:8000

# Database
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

### Provider Policy

- **ORACLE Flow** (Strategic reasoning): OpenRouter → Anthropic → DeepSeek
- **GENERAL Flow** (Cost-optimized): DeepSeek → Anthropic → OpenRouter

See: `src/engine/services/llm/provider_policy.py`

---

## Testing Strategy

Tests are organized with pytest markers:

- `@pytest.mark.unit` - Fast, isolated tests
- `@pytest.mark.integration` - Service interaction tests
- `@pytest.mark.e2e` - Full system tests
- `@pytest.mark.architecture` - Architecture guardrails
- `@pytest.mark.performance` - Benchmark tests
- `@pytest.mark.security` - Security validation tests

---

## Documentation

- **[ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md)** - Dev workflow & architecture patterns
- **[LEAN_ROADMAP.md](LEAN_ROADMAP.md)** - Refactoring roadmap & progress
- **[DEPRECATION_PLAN.md](DEPRECATION_PLAN.md)** - API deprecation timeline
- **[CLAUDE.md](CLAUDE.md)** - Project-specific AI assistant instructions
- **[docs/ARCHITECTURE_AUDIT_REPORT_2025-10-20.md](docs/ARCHITECTURE_AUDIT_REPORT_2025-10-20.md)** - Architecture audit

---

## Contributing

### Pre-Commit Workflow

1. Install pre-commit hook (optional):
   ```bash
   cp scripts/pre-commit.example .git/hooks/pre-commit
   chmod +x .git/hooks/pre-commit
   ```

2. Before committing:
   ```bash
   make dev-restart        # Clear bytecode cache
   make test-architecture  # Run guardrails
   make check             # Lint + unit + integration
   ```

3. Commit:
   ```bash
   git commit -m "feat: your changes"
   ```

---

## License

[Your License Here]

---

## Contact

- **Repository**: https://github.com/gofarrrr/lolla2
- **Issues**: https://github.com/gofarrrr/lolla2/issues

---

**Status**: ✅ Production-Ready  
**Version**: 1.0  
**Last Updated**: 2025-10-20
