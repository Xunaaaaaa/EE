#!/usr/bin/env bash
# Run the full experiment for one or more seeds, stopping at the first failure.
# Usage: ./run_pipeline.sh [seed ...]      (default: 42)
# Scripts are only ever read as text by these Python steps; nothing is executed.
set -euo pipefail
cd "$(dirname "$0")"
PY=./venv/Scripts/python.exe
[ -x "$PY" ] || PY=./venv/bin/python

"$PY" 01_download_data.py

for seed in "${@:-42}"; do
    export EE_SEED="$seed"
    mkdir -p "logs/seed_$seed"
    for step in 02_preprocess 03_train_svm codebert_finetune 05_obfuscate 06_evaluate_obfuscation 04_evaluate; do
        echo "=== seed $seed: $step ==="
        "$PY" -u "$step.py" 2>&1 | tee "logs/seed_$seed/$step.log"
    done
done
