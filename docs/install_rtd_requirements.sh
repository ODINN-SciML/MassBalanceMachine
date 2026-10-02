#!/usr/bin/env bash
# Install the dependencies of the Read the Docs build with the CPU build of torch: the
# default Linux wheel pulls several GB of CUDA libraries that the documentation does not
# need. `poetry install` would replace it with the CUDA build, so the dependencies are
# exported and installed with pip instead.
# Kept in a script because Read the Docs splits job commands on spaces and `;` even
# inside quotes.
set -euo pipefail

poetry lock
poetry export --with docs --without-hashes -o requirements-docs.txt

torch_pin=$(grep -oE '^torch==[^;[:space:]]+' requirements-docs.txt)
python -m pip install "$torch_pin" --index-url https://download.pytorch.org/whl/cpu

grep -vE '^(torch|triton|nvidia-[^=]+)==' requirements-docs.txt > requirements-docs-cpu.txt
python -m pip install -r requirements-docs-cpu.txt
python -m pip install --no-deps .
