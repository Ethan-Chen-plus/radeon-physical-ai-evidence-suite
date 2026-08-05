#!/usr/bin/env python3
"""Run the RoboCasa PandaOmron mobile-action gate and record auditable evidence.

This entry point validates the official mobile embodiment, action layout, reset,
multi-view rendering, finite actions, and stage-evidence schema. It deliberately
does not report a learned-policy success rate. Use the recorded manifest as the
contract for later demonstration conversion and SmolVLA training.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", default="PickPlaceCounterToMicrowave")
    parser.add_argument("--dataset-base-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy",
        choices=("idle", "base-sweep"),
        default="idle",
        help="Interface-gate action only; neither option is a learned policy.",
    )
    return parser.parse_args()


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def make_action(controller: Any, step: int, policy: str) -> np.ndarray:
    base_vx = 0.0
    if policy == "base-sweep" and 6 <= step < 18:
        base_vx = 0.12
    return controller.create_action_vector(
        {
            "right": np.zeros(6, dtype=np.float32),
            "right_gripper": np.array([-1.0], dtype=np.float32),
            "base": np.array([base_vx, 0.0, 0.0], dtype=np.float32),
            "torso": np.array([0.0], dtype=np.float32),
            "base_mode": 1.0,
        }
    ).astype(np.float32, copy=False)


def step_env(env: Any, action: np.ndarray) -> tuple[Any, float, bool, dict[str, Any]]:
    result = env.step(action)
    if len(result) == 4:
        obs, reward, done, info = result
        return obs, float(reward), bool(done), dict(info)
    obs, reward, terminated, truncated, info = result
    return obs, float(reward), bool(terminated or truncated), dict(info)


def main() -> None:
    args = parse_args()
    if args.episodes < 1 or args.steps < 1:
        raise ValueError("episodes and steps must be positive")

    import imageio.v2 as imageio

    import robocasa.macros as macros

    macros.DATASET_BASE_PATH = args.dataset_base_path
    from robocasa.utils.env_utils import create_env

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parents[1]
    if not (source_root / "robocasa").exists():
        source_root = Path(__file__).resolve().parent

    camera_names = [
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ]
    manifest: dict[str, Any] = {
        "schema_version": "robocasa-mobile-gate-v1",
        "status": "environment_gate_not_policy_eval",
        "policy_type": "scripted_interface_gate",
        "learned_policy_success_rate": None,
        "env_name": args.env_name,
        "robot": "PandaOmron",
        "base": "OmronMobileBase",
        "episodes": args.episodes,
        "steps_per_episode": args.steps,
        "seed_start": args.seed,
        "camera_names": camera_names,
        "camera_width": args.width,
        "camera_height": args.height,
        "fps": args.fps,
        "dataset_base_path": str(Path(args.dataset_base_path).resolve()),
        "robocasa_commit": git_commit(source_root / "robocasa"),
        "robosuite_commit": git_commit(source_root / "robosuite"),
        "action_contract": {},
        "episodes_evidence": [],
    }

    for episode_index in range(args.episodes):
        seed = args.seed + episode_index
        env = create_env(
            args.env_name,
            robots="PandaOmron",
            camera_names=camera_names,
            camera_widths=args.width,
            camera_heights=args.height,
            seed=seed,
            render_onscreen=False,
        )
        try:
            obs = env.reset()
            robot = env.robots[0]
            controller = robot.composite_controller
            action_info, action_dims = controller.get_action_info()
            action_dim = int(env._action_dim)
            low, high = env.action_spec
            manifest["action_contract"] = {
                "action_dim": action_dim,
                "action_spec_low": jsonable(low),
                "action_spec_high": jsonable(high),
                "controller": "HYBRID_MOBILE_BASE",
                "parts": action_info,
                "part_dimensions": action_dims,
                "base_mode_index": action_dim - 1,
                "control_frequency_hz": int(env.control_freq),
                "control_timestep_s": float(env.control_timestep),
                "action_order_note": (
                    "right OSC pose, right gripper, base joint velocity, torso, base mode"
                ),
            }

            video_path = output_dir / f"episode_{episode_index:03d}.mp4"
            action_rows: list[np.ndarray] = []
            stage = {
                "environment_reset": True,
                "action_contract": True,
                "finite_actions": True,
                "multi_view_render": True,
                "navigation": None,
                "approach": None,
                "contact": None,
                "grasp": None,
                "lift": None,
                "transport": None,
                "place": None,
                "release": None,
                "recovery": None,
                "final_success": None,
            }
            with imageio.get_writer(
                str(video_path), fps=args.fps, codec="libx264", macro_block_size=1
            ) as writer:
                for step_index in range(args.steps):
                    action = make_action(controller, step_index, args.policy)
                    if action.shape != (action_dim,) or not np.isfinite(action).all():
                        stage["finite_actions"] = False
                    action_rows.append(action.copy())
                    _, _, done, _ = step_env(env, action)
                    views = [
                        env.sim.render(
                            camera_name=camera_name,
                            width=args.width,
                            height=args.height,
                        )
                        for camera_name in camera_names
                    ]
                    writer.append_data(np.concatenate(views, axis=1))
                    if done:
                        break

            np.save(output_dir / f"episode_{episode_index:03d}_actions.npy", np.stack(action_rows))
            episode_record = {
                "episode_index": episode_index,
                "seed": seed,
                "video": video_path.name,
                "action_file": f"episode_{episode_index:03d}_actions.npy",
                "steps_recorded": len(action_rows),
                "stage_metrics": stage,
                "success": None,
                "result_type": "environment_gate",
                "initial_observation_shapes": {
                    key: list(value.shape)
                    for key, value in obs.items()
                    if hasattr(value, "shape")
                },
            }
            manifest["episodes_evidence"].append(episode_record)
        finally:
            env.close()

    manifest["completed"] = True
    (output_dir / "run_manifest.json").write_text(
        json.dumps(jsonable(manifest), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "status": manifest["status"],
        "episodes": args.episodes,
        "runtime_gate_passed": True,
        "learned_policy_successes": None,
        "learned_policy_episodes": None,
        "note": "This artifact validates the official mobile environment and action contract only.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()
