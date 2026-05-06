#!/usr/bin/env bash
set -euo pipefail

echo "Removing Python, pytest, and ruff caches..."
rm -rf .pytest_cache/
rm -rf .ruff_cache/
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

echo "Done."
