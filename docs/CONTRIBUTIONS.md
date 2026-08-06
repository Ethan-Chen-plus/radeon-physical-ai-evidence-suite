# Project Contributions

Radeon Physical AI Evidence Suite combines established open-source robotics
projects with new AMD-focused integration, evaluation, and publication work.

## Datawhale-EAI Work

1. ROCm PyTorch and native ROCm JAX execution paths across the selected policy
   and simulator stacks.
2. A matched RoboCasa365 GR00T/Pi0.5 evaluation workflow plus a PandaOmron
   12-dimensional action and three-camera runtime gate.
3. Shared evidence records that connect task protocol, checkpoint identity,
   result JSON, video, telemetry, and SHA-256.
4. AMD profiling tools for policy latency, simulator throughput, GPU use, and
   memory use.
5. DISCOVERSE multi-view, expert-data, and 3DGS validation workflows on AMD.
6. A portable Unitree G1 predictive-CBF replay implemented with ROCm PyTorch
   and MuJoCo.
7. An English technical report, a 4:59 demonstration film, an English
   interactive showcase, and a deterministic release validator.

## Upstream Foundations

The model architectures, benchmark tasks, simulators, and public checkpoints
remain the work of their respective upstream authors. They are listed in
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md), and each component keeps
its upstream license and model or dataset terms.

## Team

**Kewei Chen, project lead and primary maintainer:** system architecture,
ROCm/JAX migration, simulator integration, training and evaluation, evidence
validation, report, film, and website.

**Yayu Long, evaluation and learning experience contributor:** learner-facing
AMD environment and notebook validation, evaluation and representative-video
review, reproduction documentation, and final submission quality assurance.
