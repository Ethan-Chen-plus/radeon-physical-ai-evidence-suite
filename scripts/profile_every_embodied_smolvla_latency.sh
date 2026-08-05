#!/usr/bin/env bash
set -euo pipefail

# Run model-only SmolVLA latency profiling on an AMD ROCm host.

PROJECT_DIR="${PROJECT_DIR:?Set PROJECT_DIR to the Every Embodied MuJoCo PnP checkout}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LEGACY_LEROBOT="${LEGACY_LEROBOT:?Set LEGACY_LEROBOT to the compatible LeRobot source checkout}"
POLICY_PATH="${POLICY_PATH:?Set POLICY_PATH to the released SmolVLA checkpoint}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_SCRIPT="${PROFILE_SCRIPT:-$SCRIPT_DIR/profile_every_embodied_smolvla_latency.py}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/runs/benchmarks}"
RUN_ID="${RUN_ID:-smolvla_weighted500_model_latency_$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-0}"
WARMUP="${WARMUP:-8}"
ITERS="${ITERS:-60}"

RUN_DIR="$RUN_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"
cd "$PROJECT_DIR"

export PYTHONPATH="$LEGACY_LEROBOT:$PROJECT_DIR:/tmp:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-}"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"

{
  echo "run_dir=$RUN_DIR"
  echo "project_dir=$PROJECT_DIR"
  echo "python=$PYTHON_BIN"
  echo "legacy_lerobot=$LEGACY_LEROBOT"
  echo "policy=$(readlink -f "$POLICY_PATH")"
  echo "hostname=$(hostname)"
  date -Is
  /opt/rocm/bin/rocm-smi --showproductname --showuse --showmemuse --showtemp 2>/dev/null || true
} | tee "$RUN_DIR/meta.log"

"$PYTHON_BIN" "$PROFILE_SCRIPT" \
  --policy-path "$POLICY_PATH" \
  --device cuda \
  --seed "$SEED" \
  --warmup "$WARMUP" \
  --iters "$ITERS" \
  --output-json "$RUN_DIR/model_latency.json" 2>&1 | tee "$RUN_DIR/profile.log"

echo "Model latency evidence root: $RUN_DIR"
