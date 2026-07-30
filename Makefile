.PHONY: help install test run clean docker-up docker-down

help:
	@echo "HonCut Pipeline Commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make test         - Run tests"
	@echo "  make run          - Run pipeline (requires INPUT_FILE)"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make docker-up    - Start Docker services"
	@echo "  make docker-down  - Stop Docker services"

install:
	@echo "Installing dependencies..."
	pip install -r pipeline/requirements.txt
	pip install -e .

test:
	@echo "Running tests..."
	pytest pipeline/tests/ -v

run:
	@echo "Running pipeline..."
	@if [ -z "$(INPUT_FILE)" ]; then \
		echo "Error: INPUT_FILE is required. Usage: make run INPUT_FILE=path/to/file.txt"; \
		exit 1; \
	fi
	python pipeline/src/pipeline_runner.py --input $(INPUT_FILE) --output-dir data/output

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
