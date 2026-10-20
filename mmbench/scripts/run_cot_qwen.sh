#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$THIS_DIR/.." && pwd)

export CONFIG=${CONFIG:-$PROJECT_ROOT/configs/cot_eval.yaml}

bash "$PROJECT_ROOT/scripts/run_cot_eval.sh"
