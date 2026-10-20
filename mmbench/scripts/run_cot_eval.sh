#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$THIS_DIR/.." && pwd)

CONFIG=${CONFIG:-$PROJECT_ROOT/configs/cot_eval.yaml}
ADAPTER_PATH=${ADAPTER_PATH:-}
WORK_DIR=${WORK_DIR:-}
PYTHON_BIN=${PYTHON_BIN:-python3}
SWIFT_SRC_ROOT=${SWIFT_SRC_ROOT:-$PROJECT_ROOT/../ms-swift}

CMD=(
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/eval_cot.py"
  --config "$CONFIG"
)
if [[ -n "$ADAPTER_PATH" ]]; then
  CMD+=(--adapter "$ADAPTER_PATH")
fi
if [[ -n "$WORK_DIR" ]]; then
  CMD+=(--work-dir "$WORK_DIR")
fi

cd "$PROJECT_ROOT"
LOG_LEVEL=${LOG_LEVEL:-WARNING} \
PYTHONPATH="$SWIFT_SRC_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
"${CMD[@]}"
