#!/usr/bin/env bash
set -euo pipefail

# Run a small end-to-end Every Embodied SmolVLA benchmark on an AMD ROCm host.
# This script is intended to be launched on the target machine, not through SSH.
# It records strict physical success, per-episode elapsed time, action steps, and
# coarse ROCm SMI samples without modifying the protected checkpoint.

PROJECT_DIR="${PROJECT_DIR:?Set PROJECT_DIR to the Every Embodied MuJoCo PnP checkout}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LEGACY_LEROBOT="${LEGACY_LEROBOT:?Set LEGACY_LEROBOT to the compatible LeRobot source checkout}"
AUDIT_SCRIPT="${AUDIT_SCRIPT:?Set AUDIT_SCRIPT to audit_smolvla_physical.py}"
POLICY_PATH="${POLICY_PATH:?Set POLICY_PATH to the released SmolVLA checkpoint}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/runs/benchmarks}"
RUN_ID="${RUN_ID:-smolvla_weighted500_amd_panel6_$(date +%Y%m%d_%H%M%S)}"
SEEDS=(${SEEDS:-0 1 2})
MAX_ACTION_STEPS="${MAX_ACTION_STEPS:-600}"
ROCM_SMI="${ROCM_SMI:-/opt/rocm/bin/rocm-smi}"

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
  "$ROCM_SMI" --showproductname --showuse --showmemuse --showtemp 2>/dev/null || true
  "$PYTHON_BIN" - <<'PY'
import platform
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

print("python_platform=", platform.platform())
print("torch=", torch.__version__)
print("hip=", torch.version.hip)
print("cuda_available=", torch.cuda.is_available())
print("legacy_lerobot_common=ok")
if torch.cuda.is_available():
    print("device=", torch.cuda.get_device_name(0))
PY
} | tee "$RUN_DIR/meta.log"

(
  while true; do
    date +%s.%N
    "$ROCM_SMI" --showuse --showmemuse --showtemp 2>/dev/null \
      | grep -E 'GPU\[0\]|Temperature|GPU use|GPU Memory' || true
    sleep 1
  done
) > "$RUN_DIR/rocm_smi_sample.log" 2>&1 &
SAMPLER=$!

cleanup() {
  kill "$SAMPLER" 2>/dev/null || true
  wait "$SAMPLER" 2>/dev/null || true
}
trap cleanup EXIT

for color in red blue; do
  instr="Place the ${color} mug on the plate."
  /usr/bin/time -f "wall_seconds=%e\nmax_rss_kb=%M" -o "$RUN_DIR/time_${color}.txt" \
    "$PYTHON_BIN" "$AUDIT_SCRIPT" \
      --policy-path "$POLICY_PATH" \
      --device cuda \
      --seeds "${SEEDS[@]}" \
      --instruction "$instr" \
      --max-action-steps "$MAX_ACTION_STEPS" \
      --output-jsonl "$RUN_DIR/eval_${color}_seeds.jsonl" \
      --summary-json "$RUN_DIR/summary_${color}_seeds.json" \
      --log-steps-jsonl "$RUN_DIR/trace_${color}_seeds.jsonl" \
      --log-every 100 > "$RUN_DIR/eval_${color}_seeds.log" 2>&1
  echo "${color}_status=0" | tee -a "$RUN_DIR/meta.log"
  sed "s/^/${color}_/" "$RUN_DIR/time_${color}.txt" | tee -a "$RUN_DIR/meta.log"
  cat "$RUN_DIR/summary_${color}_seeds.json" | tee -a "$RUN_DIR/meta.log"
done

"$PYTHON_BIN" - "$RUN_DIR" <<'PY'
from pathlib import Path
import json
import re
import statistics
import sys

run = Path(sys.argv[1])
episodes = []
for p in sorted(run.glob("eval_*_seeds.jsonl")):
    color = p.name.split("_")[1]
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["color"] = color
        episodes.append(row)

text = (run / "rocm_smi_sample.log").read_text(errors="replace")
gpu = [int(x) for x in re.findall(r"GPU use \(%\):\s*(\d+)", text)]
mem = [int(x) for x in re.findall(r"GPU Memory Allocated \(VRAM%\):\s*(\d+)", text)]
temp = [float(x) for x in re.findall(r"Temperature \(Sensor edge\) \(C\):\s*([0-9.]+)", text)]
elapsed = [float(r.get("elapsed_s", 0)) for r in episodes]
action_steps = [int(r.get("action_steps", 0)) for r in episodes]

def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    idx = (len(values) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] * (1 - frac) + values[hi] * frac

out = {
    "run_dir": str(run),
    "episodes": len(episodes),
    "physical_success_count": sum(1 for r in episodes if r.get("physical_success")),
    "success_count": sum(1 for r in episodes if r.get("success")),
    "episode_elapsed_mean_s": round(statistics.mean(elapsed), 3) if elapsed else None,
    "episode_elapsed_p95_s": round(percentile(elapsed, 0.95), 3) if elapsed else None,
    "episode_elapsed_values_s": elapsed,
    "action_steps_mean": round(statistics.mean(action_steps), 3) if action_steps else None,
    "action_steps_p95": round(percentile(action_steps, 0.95), 3) if action_steps else None,
    "gpu_use_peak_percent": max(gpu) if gpu else None,
    "vram_peak_percent": max(mem) if mem else None,
    "temp_peak_c": max(temp) if temp else None,
    "gpu_use_samples": len(gpu),
}
(run / "panel_benchmark_summary.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False) + "\n"
)
print(json.dumps(out, indent=2, ensure_ascii=False))
PY

echo "Benchmark evidence root: $RUN_DIR"
