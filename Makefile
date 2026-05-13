.PHONY: help install dev test lint format type-check build push clean seed migrate docker-up docker-down docker-logs

PYTHON := python3
PIP := pip
IMAGE_NAME := rag-platform
IMAGE_TAG ?= latest
REGISTRY ?= ghcr.io/nguyentuansi

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"
	$(PIP) install pre-commit
	pre-commit install

dev: ## Start development server with hot reload
	uvicorn src.rag.api.app:app --host 0.0.0.0 --port 8000 --reload --reload-dir src/

test: ## Run test suite
	pytest tests/ -v --tb=short --cov=src/rag --cov-report=term-missing --cov-report=html

test-fast: ## Run tests without coverage
	pytest tests/ -v --tb=short -x

lint: ## Run ruff linter
	ruff check src/ --fix
	ruff format src/

format: ## Format code
	ruff format src/
	ruff check src/ --fix --select I

type-check: ## Run mypy type checker
	mypy src/rag --ignore-missing-imports

security: ## Run security checks
	bandit -r src/ -ll -x "*/test*"
	safety check --file requirements.txt

build: ## Build Docker production image
	docker build -f docker/Dockerfile --target runtime -t $(IMAGE_NAME):$(IMAGE_TAG) .

build-dev: ## Build Docker development image
	docker build -f docker/Dockerfile.dev -t $(IMAGE_NAME)-dev:$(IMAGE_TAG) .

push: build ## Push image to registry
	docker tag $(IMAGE_NAME):$(IMAGE_TAG) $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
	docker push $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

clean: ## Remove build artifacts, cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true

seed: ## Seed sample documents into the vector store
	$(PYTHON) scripts/seed_data.py

migrate: ## Run any pending migrations
	$(PYTHON) scripts/migrate.py

docker-up: ## Start all services via docker-compose
	docker-compose up -d

docker-up-prod: ## Start production stack
	docker-compose -f docker-compose.prod.yml up -d

docker-down: ## Stop all docker-compose services
	docker-compose down

docker-logs: ## Tail logs from all services
	docker-compose logs -f

health: ## Check API health
	$(PYTHON) scripts/health_check.py

shell: ## Open Python shell with app context loaded
	PYTHONPATH=src $(PYTHON) -c "from src.rag.core.config import get_settings; s = get_settings(); print('Settings loaded:', s.environment)"
