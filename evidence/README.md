# Public Evidence Manifest

This directory contains compact, reviewable result records copied from the
published showcase at the submission freeze.

| File | Scope |
|---|---|
| `robocasa-official-match.json` | Shared 16-task x 50-episode GR00T and Pi0.5 evaluation |
| `dexjoco-showcase.json` | Official seed-0 and deterministic first-success search for 11 tasks |
| `discoverse-hd-showcase.json` | Four MMK2 task replays with native multi-view video metadata |
| `discoverse-3dgs-showcase.json` | 3D Gaussian Splatting renderer and video validation |
| `perceptive-cbf-rl-g1-eval.json` | Eight fixed-seed Unitree G1 predictive CBF replays |
| `demo-release.json` | Demo-film metadata, languages, runtime, and media hashes |

Every file is included in the repository-level `SHA256SUMS`. Run
`python3 scripts/validate_public_bundle.py` to recompute totals and verify the
release contract.

