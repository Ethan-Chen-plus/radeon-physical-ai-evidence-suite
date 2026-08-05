#!/usr/bin/env python3
"""Profile Every Embodied SmolVLA model-only action latency.

Run this from the Every Embodied MuJoCo PnP project directory with the legacy
LeRobot source and project root already on PYTHONPATH. The script loads one
static MuJoCo observation, constructs the normal SmolVLA batch, then measures
`policy.select_action(batch)` without advancing physics.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from eval_policy_success import make_smolvla_policy, to_tensor_image


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    idx = (len(values) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def synchronize(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--instruction", default="Place the red mug on the plate.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from mujoco_env.y_env2 import SimpleEnv2

    env = SimpleEnv2("./asset/example_scene_y2.xml", action_type="joint_angle")
    try:
        env.reset(seed=args.seed)
        env.set_instruction(args.instruction)
        policy = make_smolvla_policy(args.device, args.policy_path)
        policy.reset()

        state = np.asarray(env.get_ee_pose(), dtype=np.float32)
        image, wrist_image = env.grab_image()
        batch = {
            "observation.state": torch.tensor(np.asarray([state]), dtype=torch.float32, device=args.device),
            "observation.image": to_tensor_image(image).unsqueeze(0).to(args.device),
            "observation.wrist_image": to_tensor_image(wrist_image).unsqueeze(0).to(args.device),
            "task": [env.instruction],
        }

        reset_each_times: list[float] = []
        queued_times: list[float] = []

        with torch.no_grad():
            for _ in range(max(args.warmup, 0)):
                policy.reset()
                _ = policy.select_action(batch)
            synchronize(args.device)

            for _ in range(args.iters):
                policy.reset()
                synchronize(args.device)
                start = time.perf_counter()
                action = policy.select_action(batch)
                synchronize(args.device)
                reset_each_times.append((time.perf_counter() - start) * 1000.0)
                _ = action.detach().cpu().numpy()

            policy.reset()
            for _ in range(max(args.warmup, 0)):
                _ = policy.select_action(batch)
            synchronize(args.device)
            for _ in range(args.iters):
                synchronize(args.device)
                start = time.perf_counter()
                action = policy.select_action(batch)
                synchronize(args.device)
                queued_times.append((time.perf_counter() - start) * 1000.0)
                _ = action.detach().cpu().numpy()

        def summarize(values: list[float]) -> dict:
            return {
                "count": len(values),
                "mean_ms": round(statistics.mean(values), 3) if values else None,
                "median_ms": round(statistics.median(values), 3) if values else None,
                "p95_ms": round(percentile(values, 0.95), 3) if values else None,
                "min_ms": round(min(values), 3) if values else None,
                "max_ms": round(max(values), 3) if values else None,
            }

        result = {
            "policy": "smolvla",
            "policy_path": str(args.policy_path),
            "device": args.device,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "seed": args.seed,
            "instruction": args.instruction,
            "warmup": args.warmup,
            "iters": args.iters,
            "full_forward_reset_each": summarize(reset_each_times),
            "steady_select_without_reset": summarize(queued_times),
            "note": (
                "full_forward_reset_each resets the action queue before each call; "
                "steady_select_without_reset may include cached chunk retrieval and is lower-bound timing."
            ),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    finally:
        try:
            env.env.close_viewer()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
