# METIS V5 Testing and Development Makefile
.PHONY: help test test-unit test-integration test-e2e test-performance test-security test-architecture test-all install-test-deps clean-test coverage lint format check

# Default target
help:
	@echo "Lolla V6 Testing Commands:"
	@echo "  make install-test-deps  Install all testing dependencies"
	@echo "  make test              Run all tests"
	@echo "  make test-unit         Run unit tests only"
	@echo "  make test-integration  Run integration tests"
	@echo "  make test-e2e          Run end-to-end tests"
	@echo "  make run-e2e-local     Run local E2E pipeline"
	@echo "  make test-performance  Run performance tests"
	@echo "  make test-security     Run security tests"
	@echo "  make test-architecture Run architecture guardrails (dependency direction, budgets)"
	@echo "  make test-agentic      Run agentic pattern tests"
	@echo "  make coverage          Generate test coverage report"
	@echo "  make lint              Run code linting"
	@echo "  make format            Format code"
	@echo "  make check             Run all checks (lint + test)"
	@echo "  make clean-test        Clean test artifacts"
	@echo ""
	@echo "Development Commands:"
	@echo "  make clean-cache       Clear Python bytecode cache (use if code changes not taking effect)"
	@echo "  make dev-restart       Kill backend + clear cache (fast debug workflow)"
	@echo "  make dev-setup         Setup development environment"
	@echo ""
	@echo "Refactoring Commands (Operation Scalpel):"
	@echo "  make generate-baseline Generate complexity baseline report"
	@echo "  make validate-goldens  Validate golden fixtures"
	@echo "  make check-complexity  Check complexity budgets (CI enforcement)"
	@echo "  make refactor-preflight Run all pre-flight checks before refactoring"

# Install testing dependencies
install-test-deps:
	@echo "Installing testing dependencies..."
	pip install -r requirements-v2.txt
	@if [ -f requirements-test.txt ]; then pip install -r requirements-test.txt; fi
	@if [ -f requirements.semantic.txt ]; then pip install -r requirements.semantic.txt; fi

# Run full pytest suite
test:
	@echo "🧪 Running full pytest suite..."
	pytest -q
	@echo "✅ Pytest completed"

# Unit tests (fast, isolated)
test-unit:
	@echo "🧪 Running unit tests..."
	@if [ -d tests/unit ]; then \
		pytest tests/unit -m "unit" --tb=short -v; \
	else \
		echo "ℹ️ Skipping unit tests: tests/unit not found"; \
	fi

# Integration tests (services working together)
test-integration:
	@echo "🔗 Running integration tests..."
	@if [ -d tests/integration ]; then \
		pytest tests/integration -m "integration" --tb=short -v; \
	else \
		echo "ℹ️ Skipping integration tests: tests/integration not found"; \
	fi

# End-to-end tests (full system)
test-e2e:
	@echo "🚀 Running end-to-end tests..."
	pytest tests/e2e -m "e2e" --tb=short -v

# Local end-to-end run with sample data
run-e2e-local:
	@echo "🚀 Running local E2E pipeline..."
	@if [ -f scripts/run_v6_pipeline_with_executors.py ]; then \
		python3 scripts/run_v6_pipeline_with_executors.py; \
	else \
		echo "⚠️  E2E script not found, running basic tests instead"; \
		pytest tests/ -m "not slow" --tb=short; \
	fi

# Live, network-dependent E2E (staging/CI nightly)
test-e2e-live:
	@echo "🌐 Running LIVE E2E against staging (requires env secrets)..."
	ENABLE_LIVE_API_CALLS=true \
	FORGE_GOLDEN_THREAD=true \
	python3 scripts/run_v6_pipeline_with_executors.py

# Performance tests
test-performance:
	@echo "⚡ Running performance tests..."
	pytest tests/performance -m "performance" --tb=short -v --benchmark-only

# Security tests
test-security:
	@echo "🔒 Running security tests..."
	bandit -r src/ -f json -o security-report.json || true
	safety check --json --output safety-report.json || true
	pytest tests/ -m "safety" --tb=short -v

# Architecture guardrails
test-architecture:
	@echo "🏛️ Running architecture guardrails..."
	pytest tests/architecture -m "architecture" --tb=short -v

# Agentic pattern tests (for enhancement validation)
test-agentic:
	@echo "🤖 Running agentic pattern tests..."
	pytest tests/ -m "agentic" --tb=short -v

# API tests only
test-api:
	@echo "🌐 Running API tests..."
	pytest tests/ -m "api" --tb=short -v

# Service tests only
test-services:
	@echo "⚙️ Running service tests..."
	pytest tests/ -m "service" --tb=short -v

# Slow tests (with timeout)
test-slow:
	@echo "🐌 Running slow tests..."
	pytest tests/ -m "slow" --tb=short -v --timeout=300

# Run your existing sovereign test
test-sovereign:
	@echo "👑 Running Evergreen Sovereign Test..."
	python tests/validation/run_evergreen_sovereign_test.py

# Coverage report
coverage:
	@echo "📊 Generating coverage report..."
	pytest --cov=src --cov-report=html --cov-report=term-missing tests/
	@echo "Coverage report generated in htmlcov/"

# Lint code
RUFF_TARGETS= \
	src/core/unified_context_stream.py \
	src/core/parallel_forges_breadth_mode.py

lint:
	@echo "🔍 Linting code..."
	@if [ -x ".venv/bin/ruff" ]; then \
		.venv/bin/ruff check $(RUFF_TARGETS); \
	else \
		ruff check $(RUFF_TARGETS); \
	fi
	@command -v black >/dev/null 2>&1 && black --check src/ tests/ || true
	mypy src/ || true

# Format code
format:
	@echo "✨ Formatting code..."
	black src/ tests/
	ruff --fix src/ tests/

# Run all checks
check: lint test-unit test-integration
	@echo "✅ All checks passed"

# Guardrails (pattern checks)
guardrails:
	@echo "🛡️  Running pattern guard..."
	python3 scripts/guardrails/pattern_guard.py

# Include guardrails in 'check'
check: guardrails

# Clean test artifacts
clean-test:
	@echo "🧹 Cleaning test artifacts..."
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	rm -f security-report.json
	rm -f safety-report.json
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Clean Python bytecode cache (FAST - use when code changes aren't taking effect)
clean-cache:
	@echo "🧹 Clearing Python bytecode cache..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cache cleared - restart backend to load fresh code"

# Development: Kill backend + clear cache + restart (USE THIS when debugging)
dev-restart:
	@echo "🔄 Development restart (cache-cleared)..."
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || echo "No backend running on port 8000"
	@sleep 1
	@$(MAKE) clean-cache
	@echo "✅ Ready to start backend with: python3 src/main.py"
	@echo "   (or run in background: nohup python3 src/main.py > /tmp/backend.log 2>&1 & )"

# Load testing (requires locust)
load-test:
	@echo "🚀 Running load tests..."
	locust -f tests/load/locustfile.py --host=http://localhost:8000 --users=10 --spawn-rate=2 --run-time=1m --headless

# Install development environment
dev-setup: install-test-deps
	@echo "🔧 Setting up development environment..."
	pre-commit install || echo "pre-commit not available"
	@echo "✅ Development environment ready"

# Quick test (unit + critical integration)
test-quick:
	@echo "⚡ Running quick tests..."
	pytest tests/unit tests/integration/test_critical.py -v --tb=short

# Test with coverage and HTML report
test-full:
	@echo "🎯 Running full test suite with coverage..."
	pytest --cov=src --cov-report=html --cov-report=term --tb=short -v
	@echo "📊 Coverage report: htmlcov/index.html"

# Run tests in parallel (requires pytest-xdist)
test-parallel:
	@echo "⚡ Running tests in parallel..."
	pytest -n auto --tb=short -v

# ============================================================
# REFACTORING ROADMAP TARGETS (Operation Scalpel)
# ============================================================

# Generate complexity baseline (radon report)
generate-baseline:
	@echo "📊 Generating complexity baseline..."
	@radon cc -j src > complexity_baseline/radon_report.json
	@echo "✅ Baseline saved to complexity_baseline/radon_report.json"
	@echo "📈 Summary:"
	@python3 -c "import json; r=json.load(open('complexity_baseline/radon_report.json')); print(f'  Total files: {len(r)}'); mx=max((max([f['complexity'] for f in fs], default=0) for fs in r.values()), default=0); print(f'  Max CC: {mx}')"

# Validate golden fixtures
validate-goldens:
	@echo "🧪 Validating golden fixtures..."
	@if [ -d tests/golden ]; then \
		pytest tests/golden -v --tb=short; \
	else \
		echo "⚠️  tests/golden/ not found - creating directory"; \
		mkdir -p tests/golden; \
		echo "ℹ️  Add golden fixture tests to tests/golden/"; \
	fi

# Check complexity budgets (CI enforcement)
check-complexity:
	@echo "🔍 Checking complexity budgets..."
	@radon cc -j src > complexity_baseline/radon_report.json
	@python3 scripts/ci/check_complexity_budget.py \
		--current complexity_baseline/radon_report.json \
		--budgets complexity_baseline/complexity_budgets.json
	@echo "✅ Complexity budgets OK"

# Refactoring pre-flight check (before starting refactoring)
refactor-preflight:
	@echo "🚀 Running refactoring pre-flight checks..."
	@$(MAKE) generate-baseline
	@$(MAKE) check-complexity
	@$(MAKE) validate-goldens
	@$(MAKE) test-unit
	@echo "✅ Pre-flight complete - ready to refactor"

# Add refactoring commands to help
.PHONY: generate-baseline validate-goldens check-complexity refactor-preflight
