# Reproducibility Guide

This guide maps each public result to an executable workflow on AMD Radeon and
ROCm. Run commands from the repository root unless a section says otherwise.

## 1. Hardware and Operating Systems

| Host | GPU | Primary workloads |
|---|---|---|
| Radeon Developer Cloud | AMD Radeon PRO W7900, 48 GiB | VLA and ACT training |
| Local AMD workstation | AMD Ryzen AI MAX+ 395, Radeon 8060S, gfx1151 | simulation, JAX/PyTorch inference, evaluation, rendering |

The reference hosts use Ubuntu 24.04. Store datasets, checkpoints, and caches on
a persistent large-volume path. Set these variables before setup:

```bash
export WORK_ROOT=/workspace/physical-ai
export HF_HOME=$WORK_ROOT/cache/huggingface
export DATA_ROOT=$WORK_ROOT/datasets
export CHECKPOINT_ROOT=$WORK_ROOT/checkpoints
export RUN_ROOT=$WORK_ROOT/runs
mkdir -p "$HF_HOME" "$DATA_ROOT" "$CHECKPOINT_ROOT" "$RUN_ROOT"
```

## 2. Base ROCm Check

The PyTorch workflows expect an AMD ROCm build of PyTorch. Verify it before
installing simulator packages:

```bash
python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("hip", torch.version.hip)
print("available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
assert torch.cuda.is_available()
PY
```

ROCm follows PyTorch's `cuda` device API, so command-line arguments use
`--device cuda` on AMD hosts.

## 3. Evidence-Only Validation

This check requires only Python 3.10+ and validates the public JSON schemas,
headline totals, linked media, and SHA-256 records:

```bash
python3 scripts/validate_public_bundle.py
```

Expected final line:

```text
PUBLIC_BUNDLE_OK
```

## 4. Every Embodied SmolVLA

The public course notebooks contain the full native training and evaluation
workflow. Download the protected model from Hugging Face and follow the AMD
workflow notebook:

- Notebooks: <https://github.com/datawhalechina/every-embodied/tree/main/16-%E4%B8%93%E9%A2%98%E7%BB%84%E9%98%9F%E5%AD%A6%E4%B9%A0/04-AMD-ROCm%E7%AD%96%E7%95%A5%E5%A4%8D%E5%88%BB%E4%B8%93%E9%A2%98/notebooks/workflows>
- Model: <https://huggingface.co/Datawhale/every-embodied-smolvla-mujoco-pnp>

For the fixed AMD profiling panel:

```bash
PROJECT_DIR=/path/to/every-embodied-pnp \
POLICY_PATH=/path/to/weighted_000500 \
RUN_ROOT="$RUN_ROOT/smolvla" \
bash scripts/benchmark_every_embodied_smolvla.sh

PROJECT_DIR=/path/to/every-embodied-pnp \
POLICY_PATH=/path/to/weighted_000500 \
RUN_ROOT="$RUN_ROOT/smolvla-latency" \
bash scripts/profile_every_embodied_smolvla_latency.sh
```

The strict 60-episode protocol evaluates red and blue task variants with 30
fixed seeds each. The released checkpoint records 57/60 physical successes.

## 5. RoboCasa365 on AMD

Install the official RoboCasa, RoboSuite, and MuJoCo repositories in editable
mode, then configure off-screen rendering:

```bash
export MUJOCO_GL=egl
python3 -m pip install imageio imageio-ffmpeg numpy
python3 -m pip install -e /path/to/robosuite
python3 -m pip install -e /path/to/robocasa
```

Run the official fixed-arm environment gate:

```bash
python3 scripts/robocasa_amd_smoke.py \
  --task PickPlaceCounterToCabinet \
  --episodes 3 \
  --steps 40 \
  --out-dir "$RUN_ROOT/robocasa-smoke"
```

Run the PandaOmron mobile action-contract gate:

```bash
python3 scripts/robocasa_mobile_mvp.py \
  --env-name PickPlaceCounterToMicrowave \
  --dataset-base-path "$DATA_ROOT/robocasa" \
  --output-dir "$RUN_ROOT/robocasa-mobile-gate" \
  --episodes 1 \
  --steps 40
```

The policy evaluator accepts a trained SmolVLA path and uses RoboCasa's task
success predicate:

```bash
python3 scripts/evaluate_robocasa_mobile_smolvla.py \
  --policy-path "$CHECKPOINT_ROOT/robocasa-mobile-smolvla" \
  --dataset-root "$DATA_ROOT/robocasa-mobile" \
  --dataset-repo-id datawhale/robocasa-mobile \
  --dataset-base-path "$DATA_ROOT/robocasa" \
  --output-dir "$RUN_ROOT/robocasa-mobile-eval" \
  --episodes 50 \
  --device cuda
```

The public benchmark summary in `evidence/robocasa-official-match.json` contains
the shared 16-task x 50-episode GR00T and Pi0.5 protocol.

## 6. DexJoCo Pi0.5 with ROCm JAX

The AMD395 JAX environment was created without Docker using the AMD ROCm pip
route summarized in the [environment matrix](ENVIRONMENT_MATRIX.md):

```text
Python 3.12
ROCm 7.14.0
JAX 0.10.0
JAXlib 0.10.0
device: rocm:0 (gfx1151)
```

Clone the upstream project and download the multi-task checkpoint:

```bash
git clone https://github.com/brave-eai/dexjoco.git
```

Set `DEXJOCO_ROOT`, `CHECKPOINT`, and the Python executable for the ROCm JAX
environment, then run the official 11-task protocol:

```bash
DEXJOCO_ROOT=/path/to/dexjoco \
CHECKPOINT=/path/to/pi05_dexjoco_multi_task \
OPENPI_PYTHON=/path/to/openpi-amd-jax010/bin/python \
DEXJOCO_EVAL=/path/to/dexjoco-openpi-eval \
bash scripts/run_dexjoco_pi05_multitask_eval.sh
```

The fixed official seed is 0 and produces 5/11 successes. The optional recovery
runner searches seeds 1-10 only for failed tasks and stops at the first success:

```bash
python3 scripts/run_dexjoco_pi05_multitask_recovery.py --help
```

## 7. DISCOVERSE

Clone the official repository and install the AMD-compatible MuJoCo/PyTorch
environment described in `docs/AMD_MIGRATION_PLAYBOOK.md`:

```bash
git clone https://github.com/discoverse-dev/DISCOVERSE.git
```

The public release records runtime, AIRBOT, MMK2, expert-data, 3DGS, and video
gates. Redistribution-restricted assets stay in their upstream installation and
are referenced at runtime.

## 8. Predictive CBF for Unitree G1

The portable replay validates the predictive CBF tensor path on ROCm and renders
a MuJoCo G1 scene:

```bash
python3 code/perceptive_cbf_rl_amd/g1_amd_dodge_replay.py \
  --episodes 8 \
  --output-dir "$RUN_ROOT/pacman-g1"
```

The frozen evidence reports 8/8 safe replays with a 0.422 m minimum clearance.

## 9. Outputs and Acceptance

Every scored run should archive:

```text
run_manifest.json
eval_info.json or summary.json
per-episode videos
training metrics when training is run
ROCm device and software versions
sha256sum.txt
```

Use `python3 scripts/validate_public_bundle.py` before publishing a new release.
