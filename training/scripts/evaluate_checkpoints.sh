#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "$0")" && pwd)
TRAINING_ROOT=$(cd "$THIS_DIR/.." && pwd)
REPO_ROOT=$(cd "$TRAINING_ROOT/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}

: "${CHECKPOINT_ROOT:?Set CHECKPOINT_ROOT to the GRPO run directory containing checkpoint-* folders.}"
: "${VALIDATION_DATASET:?Set VALIDATION_DATASET to the held-out validation parquet, never the test parquet.}"

BASE_CONFIG=${BASE_CONFIG:-$REPO_ROOT/mmbench/configs/pair_eval.yaml}
METRICS_ROOT=${METRICS_ROOT:-$CHECKPOINT_ROOT/validation_metrics}
VALIDATION_SCOPE=${VALIDATION_SCOPE:-par}
VALIDATION_CONFIG="$METRICS_ROOT/validation_config.yaml"

"$PYTHON_BIN" "$THIS_DIR/make_validation_config.py" \
  --base-config "$BASE_CONFIG" \
  --validation-dataset "$VALIDATION_DATASET" \
  --output "$VALIDATION_CONFIG"

shopt -s nullglob
checkpoints=("$CHECKPOINT_ROOT"/checkpoint-*)
if [[ ${#checkpoints[@]} -eq 0 ]]; then
  echo "No checkpoint-* directories found under $CHECKPOINT_ROOT" >&2
  exit 1
fi

for checkpoint in "${checkpoints[@]}"; do
  name=$(basename "$checkpoint")
  CONFIG="$VALIDATION_CONFIG" \
  ADAPTER_PATH="$checkpoint" \
  QUESTION_SCOPE="$VALIDATION_SCOPE" \
  WORK_DIR="$METRICS_ROOT/$name" \
    bash "$REPO_ROOT/mmbench/scripts/run_tool_eval.sh"
done

"$PYTHON_BIN" "$THIS_DIR/select_checkpoint.py" \
  --metrics-root "$METRICS_ROOT" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --metric acc_plain \
  --output "$METRICS_ROOT/checkpoint_selection.json"
