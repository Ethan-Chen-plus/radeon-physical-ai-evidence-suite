# Track 3 - Datawhale-EAI - Radeon Physical AI Evidence Suite

Radeon Physical AI Evidence Suite is a cross-framework robot-learning and
simulation stack for AMD Radeon and ROCm. It connects policy training,
closed-loop household and dexterous-task evaluation, simulator migration,
multi-view rendering, performance telemetry, and SHA-pinned evidence.

## Official Deliverables

| Track 3 requirement | Submission material |
|---|---|
| Technical report | [English PDF](../../output/pdf/datawhale-eai-radeon-physical-ai-technical-report.pdf) and [source](../../docs/TECHNICAL_REPORT.md) |
| Dedicated source repository | [Ethan-Chen-plus/radeon-physical-ai-evidence-suite](https://github.com/Ethan-Chen-plus/radeon-physical-ai-evidence-suite) |
| Frozen source revision | [v1.0.1-amd-hackathon-english](https://github.com/Ethan-Chen-plus/radeon-physical-ai-evidence-suite/releases/tag/v1.0.1-amd-hackathon-english) |
| Reproducibility README | [Step-by-step AMD guide](../../docs/REPRODUCIBILITY.md) |
| Demonstration video | [4:59 English 1080p video](https://ethan-chen-plus.github.io/amd-physical-ai-showcase/assets/videos/amd-physical-ai-demo-en.mp4) |
| Supplementary showcase | [English interactive site](https://ethan-chen-plus.github.io/amd-physical-ai-showcase/) |
| Models | [SmolVLA](https://huggingface.co/Datawhale/every-embodied-smolvla-mujoco-pnp), [Pi0](https://huggingface.co/Datawhale/every-embodied-pi0-mujoco-pnp), [ACT](https://huggingface.co/Datawhale/every-embodied-act-mujoco-pnp), [RoboWits ACT](https://huggingface.co/Datawhale/robowits-act-amd-rocm) |
| Evidence and integrity | [`evidence/`](../../evidence), [`SHA256SUMS`](../../SHA256SUMS), and the [validator](../../scripts/validate_public_bundle.py) |

## Selected Results

| Workstream | Frozen result |
|---|---:|
| Every Embodied SmolVLA | **57/60** strict physical successes |
| RoboCasa365 GR00T | **230/800** over 16 tasks x 50 episodes |
| RoboCasa365 Pi0.5 | **142/800** over the same protocol |
| DexJoCo Pi0.5 | **5/11** at official seed 0; success examples archived for 10/11 tasks |
| DISCOVERSE | AIRBOT **12/12**, MMK2 **8/8**, four three-view HD task replays |
| Unitree G1 predictive CBF | **8/8** fixed-seed replays, 0.422 m minimum clearance |

## AMD Radeon and ROCm

- AMD Radeon PRO W7900: SmolVLA and ACT training.
- AMD Ryzen AI MAX+ 395: PyTorch/JAX inference, closed-loop evaluation,
  RoboCasa, DexJoCo, DISCOVERSE, MuJoCo/EGL, 3DGS, and performance profiling.
- Frameworks: ROCm PyTorch and native ROCm JAX 0.10.
- Measured SmolVLA full-forward latency on AMD395: 438.332 ms mean and
  440.883 ms p95 over 60 timed calls.
- Measured simulator throughput: 518.736 simulation steps/s.

## Quick Verification

```bash
git clone https://github.com/Ethan-Chen-plus/radeon-physical-ai-evidence-suite.git
cd radeon-physical-ai-evidence-suite
git checkout v1.0.1-amd-hackathon-english
python3 scripts/validate_public_bundle.py
```

Expected result: `PUBLIC_BUNDLE_OK`.

## Team

**Kewei Chen - project lead and primary maintainer.** System architecture, AMD
ROCm and JAX migration, simulator integration, model training and evaluation,
evidence validation, technical report, demo film, and public showcase.
