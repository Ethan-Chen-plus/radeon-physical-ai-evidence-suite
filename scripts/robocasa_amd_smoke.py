#!/usr/bin/env python3
"""Headless RoboCasa AMD smoke: reset, random rollout, MP4, and JSON metadata.

This deliberately tests the base RoboCasa/RoboSuite/MuJoCo path without
requiring LeRobot, a policy checkpoint, or a CUDA-specific renderer.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

# The source checkout layout is ``repo/robosuite/robosuite`` and
# ``repo/robocasa/robocasa``. Leaving the script directory on sys.path makes
# Python treat the outer folders as namespace packages and hides the editable
# installs. Remove it before importing the packages.
_script_dir = str(Path(__file__).resolve().parent)
if _script_dir in sys.path:
    sys.path.remove(_script_dir)

import imageio.v2 as imageio
import mujoco
import numpy as np
import robosuite
import robocasa  # noqa: F401 - registers RoboCasa environments
from robocasa.utils.env_utils import create_env

# Some editable-install import paths leave robosuite.__file__ unset after
# RoboCasa's registry imports. The upstream controller loader uses it to find
# JSON configs, so restore the equivalent source path deterministically.
if getattr(robosuite, "__file__", None) is None:
    candidate_roots = [Path(path) for path in robosuite.__path__]
    candidate_roots += [path / "robosuite" for path in candidate_roots]
    package_root = next(path for path in candidate_roots if (path / "controllers").is_dir())
    robosuite.__file__ = str(package_root / "__init__.py")


def run_episode(env, seed: int, steps: int, writer) -> dict:
    if hasattr(env, "rng"):
        env.rng = np.random.default_rng(seed)
    env.reset()
    success = False
    for step in range(steps):
        action = np.random.uniform(low=env.action_spec[0], high=env.action_spec[1])
        env.step(action)
        frame = env.sim.render(height=512, width=768, camera_name="robot0_agentview_center")[::-1]
        writer.append_data(frame)
        success = bool(env._check_success())
        if success:
            break
    return {"seed": seed, "steps": step + 1, "success": success}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="PickPlaceCounterToCabinet")
    parser.add_argument("--split", default="pretrain", choices=["pretrain", "target", "all"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("results/robocasa_amd_smoke"))
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.out_dir / f"{args.task}_{args.split}_random.mp4"
    started = time.time()
    env = create_env(env_name=args.task, seed=args.seed, split=args.split)
    episodes = []
    try:
        with imageio.get_writer(video_path, fps=20) as writer:
            for episode in range(args.episodes):
                episodes.append(run_episode(env, args.seed + episode, args.steps, writer))
    finally:
        env.close()

    result = {
        "kind": "amd_base_smoke",
        "task": args.task,
        "split": args.split,
        "episodes": episodes,
        "success_count": sum(item["success"] for item in episodes),
        "video": str(video_path.resolve()),
        "elapsed_sec": round(time.time() - started, 3),
        "versions": {
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "robosuite": robosuite.__version__,
        },
        "note": "Random rollout smoke only; not a learned-policy score.",
    }
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
