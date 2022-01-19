#!/bin/bash
set -uo pipefail
set +e

FAILURE=false

echo "safety (failure is tolerated)"
FILE=requirements/prod.txt
if [ -f "$FILE" ]; then
    # We're in the main path
    safety check -r requirements/prod.txt -r requirements/dev.txt
else
    # We're in the cellsignal path
    safety check -r ../requirements/prod.txt -r ../requirements/dev.txt
fi


if [ "$FAILURE" = true ]; then
  echo "Linting failed"
  exit 1
fi
echo "Linting passed"
exit 0