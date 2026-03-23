.PHONY: help up down logs test test-e2e seed bench lint format clean

help:  ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ── Docker ───────────────────────────────────────────────────

up:  ## Start all services
	@cp -n .env.example .env 2>/dev/null || true
	docker compose up -d --build
	@echo "\n🚀 Services starting..."
	@echo "   App:        http://localhost:8080/docs"
	@echo "   Grafana:    http://localhost:3000  (admin/admin)"
	@echo "   Prometheus: http://localhost:9090"
	@echo "   ChromaDB:   http://localhost:8000"

down:  ## Stop all services
	docker compose down

down-clean:  ## Stop and remove all data
	docker compose down -v

logs:  ## Tail app logs
	docker compose logs -f app

logs-all:  ## Tail all service logs
	docker compose logs -f

# ── Development ──────────────────────────────────────────────

seed:  ## Seed sample documents
	python scripts/seed.py

bench:  ## Run benchmark
	python scripts/benchmark.py

test:  ## Run unit tests
	python -m pytest tests/ -v --ignore=tests/test_e2e.py -x

test-e2e:  ## Run end-to-end tests (services must be running)
	python -m pytest tests/test_e2e.py -v -m e2e

test-all:  ## Run all tests
	python -m pytest tests/ -v

# ── Code Quality ─────────────────────────────────────────────

lint:  ## Lint with ruff
	ruff check .

format:  ## Format with ruff
	ruff format .

clean:  ## Remove Python caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
