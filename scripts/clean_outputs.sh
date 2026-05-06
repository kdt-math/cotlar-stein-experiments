#!/usr/bin/env bash
set -euo pipefail

echo "Removing generated experiment outputs..."
rm -rf results/
rm -rf figures/

echo "Done."
