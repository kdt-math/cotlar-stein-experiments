#!/usr/bin/env bash
set -euo pipefail

bash scripts/clean_outputs.sh
bash scripts/clean_cache.sh

echo "All generated files cleaned."
