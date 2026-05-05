.PHONY: venv test lint format check

test:
	pytest -q

lint:
	ruff check .

format:
	ruff format .

check: lint test
