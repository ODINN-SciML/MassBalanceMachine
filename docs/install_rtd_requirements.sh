#!/usr/bin/env bash
# Install the dependencies of the Read the Docs build from docs/requirements-docs.txt,
# with the CPU build of torch: the default Linux wheel pulls several GB of CUDA libraries
# that the documentation does not need. How to regenerate docs/requirements-docs.txt is
# described in docs/contributing.rst ("Changing the dependencies").
# Kept in a script because Read the Docs splits job commands on spaces and `;` even
# inside quotes.
set -euo pipefail

requirements=docs/requirements-docs.txt
filtered=$(mktemp)

torch_pin=$(grep -oE '^torch==[^;[:space:]]+' "$requirements")
uv pip install --python python "$torch_pin" --index-url https://download.pytorch.org/whl/cpu

grep -vE '^(torch|triton|nvidia-[^=]+)==' "$requirements" > "$filtered"
uv pip install --python python -r "$filtered"
uv pip install --python python --no-deps .
