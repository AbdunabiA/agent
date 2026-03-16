.PHONY: install dev test lint format type-check run chat doctor clean

install:
	pip install -e .
	python scripts/fix_editable_install.py

dev:
	pip install -e ".[dev]"
	python scripts/fix_editable_install.py
	pre-commit install

test:
	pytest -v --cov=agent --cov-report=term-missing

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

type-check:
	mypy src/

run:
	agent start

chat:
	agent chat

doctor:
	agent doctor

clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
