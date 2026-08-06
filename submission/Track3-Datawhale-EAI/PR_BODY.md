## Track 3 Submission - Radeon Physical AI Evidence Suite

Radeon Physical AI Evidence Suite is Datawhale-EAI's reproducible Physical AI
stack for AMD Radeon and ROCm. It connects VLA training, closed-loop household
and dexterous evaluation, simulator migration, native ROCm JAX, multi-view
rendering, performance telemetry, and SHA-pinned evidence.

### Official deliverables

| Requirement | Material |
|---|---|
| Technical report | English PDF in this PR and source repository |
| Source code | https://github.com/Ethan-Chen-plus/radeon-physical-ai-evidence-suite |
| Frozen release | https://github.com/Ethan-Chen-plus/radeon-physical-ai-evidence-suite/releases/tag/v1.0.2-amd-hackathon-final |
| Reproducibility | `docs/REPRODUCIBILITY.md` plus `scripts/validate_public_bundle.py` |
| Demo video | https://ethan-chen-plus.github.io/amd-physical-ai-showcase/assets/videos/amd-physical-ai-demo-en.mp4 |
| Interactive showcase | https://ethan-chen-plus.github.io/amd-physical-ai-showcase/ |
| Model artifacts | Datawhale Hugging Face links in the submission README |

### Selected results

- SmolVLA: **57/60** strict physical successes.
- RoboCasa365: GR00T **230/800** and Pi0.5 **142/800** over the same
  16-task x 50-episode protocol.
- DexJoCo Pi0.5: **5/11** at official seed 0; reproducible success examples for
  10/11 tasks within the fixed seed archive.
- DISCOVERSE: AIRBOT **12/12**, MMK2 **8/8**, four three-view HD task replays,
  and validated 3DGS rendering.
- Unitree G1 predictive CBF: **8/8** fixed-seed replays with 0.422 m minimum
  clearance.

### AMD Radeon and ROCm execution

The project uses an AMD Radeon PRO W7900 for model training and an AMD Ryzen AI
MAX+ 395 for ROCm PyTorch/JAX inference, closed-loop evaluation, simulation,
rendering, and profiling. The measured SmolVLA full-forward latency on AMD395 is
438.332 ms mean / 440.883 ms p95; aggregate simulator throughput is 518.736
steps/s.

### Team

**Kewei Chen - project lead and primary maintainer:** architecture, AMD ROCm and
JAX migration, simulator integration, model training and evaluation, evidence
validation, report, demo film, and website.

**Yayu Long - evaluation and learning experience contributor:** learner-facing
AMD environment and notebook validation, evaluation record and representative
video review, reproduction documentation, and final submission quality
assurance.
