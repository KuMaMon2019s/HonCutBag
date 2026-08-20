UV ?= uv
UV_SYNC = $(UV) sync --locked --managed-python
UV_RUN = $(UV) run --locked --managed-python

.PHONY: help install doctor test lint run clean docker-up docker-down

help:
	@echo "HonCut Pipeline Commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make doctor       - Verify the locked project interpreter"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run critical Ruff checks"
	@echo "  make run          - Run pipeline (requires INPUT_FILE)"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make docker-up    - Start Docker services"
	@echo "  make docker-down  - Stop Docker services"

install:
	@command -v $(UV) >/dev/null 2>&1 || { echo "Error: uv is required (https://docs.astral.sh/uv/)"; exit 1; }
	@echo "Installing locked dependencies with uv..."
	$(UV_SYNC)

doctor:
	@command -v $(UV) >/dev/null 2>&1 || { echo "Error: uv is required (https://docs.astral.sh/uv/)"; exit 1; }
	$(UV_RUN) python scripts/check_python_environment.py

test: doctor
	@echo "Running tests..."
	$(UV_RUN) python -m pytest pipeline/tests/ pipeline/test_story_order_multimodal.py -v

lint: doctor
	@echo "Running critical Ruff checks..."
	$(UV_RUN) python -m ruff check pipeline/src pipeline/tests --select E9,F63,F7,F82

run: doctor
	@echo "Running pipeline..."
	@if [ -z "$(INPUT_FILE)" ]; then \
		echo "Error: INPUT_FILE is required. Usage: make run INPUT_FILE=path/to/file.txt"; \
		exit 1; \
	fi
	@# 计算今天的日期和序号
	@TODAY=$$(date +%Y-%m-%d); \
	SEQ=01; \
	while [ -d "workspaces/$${TODAY}_$${SEQ}" ]; do \
		SEQ=$$(printf "%02d" $$(($$(echo $$SEQ | sed 's/^0//') + 1))); \
	done; \
	WORKSPACE="workspaces/$${TODAY}_$${SEQ}"; \
	echo "Creating workspace: $$WORKSPACE"; \
	mkdir -p "$$WORKSPACE/input" "$$WORKSPACE/output" "$$WORKSPACE/shots"; \
	cp "$(INPUT_FILE)" "$$WORKSPACE/input/"; \
	echo "Running pipeline in $$WORKSPACE..."; \
	$(UV_RUN) python pipeline/src/pipeline_runner.py \
		--input "$$WORKSPACE/input/$$(basename $(INPUT_FILE))" \
		--output-dir "$$WORKSPACE/output" \
		--auto-approve

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-up:
	@echo "Starting Docker services..."
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	@echo "Stopping Docker services..."
	docker-compose -f docker/docker-compose.yml down
