#!/usr/bin/env bash
set -euo pipefail

# Evaluate the downloaded DexJoCo multi-task Pi0.5 checkpoint once per task.
# The caller supplies installation and artifact locations so the same command
# works on AMD395, Radeon Cloud, or another ROCm host.
ROOT="${DEXJOCO_ROOT:?Set DEXJOCO_ROOT to the DexJoCo AMD checkout}"
OPENPI_PYTHON="${OPENPI_PYTHON:?Set OPENPI_PYTHON to the ROCm JAX Python executable}"
DEXJOCO_EVAL="${DEXJOCO_EVAL:?Set DEXJOCO_EVAL to the dexjoco-openpi-eval executable}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to pi05_dexjoco_multi_task}"
OUT="${OUT:-$ROOT/runs/amd_openpi_jax010_multitask}"
PORT="${PORT:-8020}"
MARKER="$OUT/multitask_eval_complete"

mkdir -p "$OUT"
if [[ -f "$MARKER" ]]; then
  echo "already_complete: $MARKER"
  exit 0
fi

cd "$ROOT"
export MUJOCO_GL=egl
export JAX_PLATFORMS=rocm
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40
export DEXJOCO_PYTORCH_COMPILE_MODE=none
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/openpi/src:$ROOT/openpi/packages/openpi-client/src"

SERVER_LOG="$OUT/policy_server.log"
SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

(
  cd "$ROOT/openpi"
  "$OPENPI_PYTHON" scripts/serve_policy.py \
    --port "$PORT" \
    policy:checkpoint \
    --policy.config multi_task \
    --policy.dir "$CHECKPOINT"
) >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

"$OPENPI_PYTHON" - "$PORT" <<'PY'
import socket
import sys
import time

port = int(sys.argv[1])
for _ in range(300):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit(f"policy server did not open port {port}")
PY

tasks=(
  bimanual_assembly
  bimanual_hanoi
  bimanual_microwave_cook
  bimanual_photograph
  bimanual_unlock_ipad
  click_mouse
  fold_glasses
  hammer_nail
  pick_bucket
  pinch_tongs
  water_plant
)

for task in "${tasks[@]}"; do
  task_out="$OUT/$task"
  if [[ -f "$task_out/eval_info.json" ]] || find "$task_out" -maxdepth 1 -type f -name 'success_rate_*_1.txt' -print -quit | grep -q .; then
    echo "skip_existing: $task"
    continue
  fi
  mkdir -p "$task_out"
  echo "start: $task $(date -Is)" | tee -a "$OUT/eval.log"
  "$DEXJOCO_EVAL" \
    --config="$ROOT/configs/multi_task/$task.yaml" \
    --seed=0 \
    --port="$PORT" \
    --output="$task_out" \
    --render-mode=rgb_array \
    --pad-state-dim46 \
    --episodes=1 \
    2>&1 | tee -a "$OUT/eval.log"
  echo "done: $task $(date -Is)" | tee -a "$OUT/eval.log"
done

date -Is > "$MARKER"
echo "complete: $MARKER" | tee -a "$OUT/eval.log"
