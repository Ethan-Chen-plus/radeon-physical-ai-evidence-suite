#!/usr/bin/env python3
"""Evaluate a trained SmolVLA policy on official RoboCasa mobile tasks.

The evaluator uses the PandaOmron mobile-manipulation environments, the
12-dimensional HYBRID_MOBILE_BASE action contract, and RoboCasa's own task
success predicate. It records reproducible rollout videos, per-episode JSON,
policy/data/environment metadata, and a SHA256 manifest. It never converts
render success or a finite rollout into a learned-policy success.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np


CAMERA_NAMES = [
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]
DEFAULT_TASKS = [
    ("NavigateKitchen", "NavigateKitchen"),
    ("PickPlaceCounterToCabinet", "PickPlaceCounterToCabinet"),
    ("PickPlaceDrawerToCounter", "PickPlaceDrawerToCounter"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-base-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--tasks",
        default=",".join(name for name, _ in DEFAULT_TASKS),
        help="Comma-separated official RoboCasa task names.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def state_from_observation(obs: dict[str, Any]) -> np.ndarray:
    """Build the 16-D RoboCasa LeRobot state: joints, velocities, gripper."""

    return np.concatenate(
        [
            np.asarray(obs["robot0_joint_pos"], dtype=np.float32),
            np.asarray(obs["robot0_joint_vel"], dtype=np.float32),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ]
    )


def action_from_policy(action: Any, action_dim: int, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    array = action.detach().to("cpu").numpy() if hasattr(action, "detach") else np.asarray(action)
    array = np.asarray(array)
    if array.ndim == 3:
        array = array[:, 0]
    if array.ndim == 2:
        array = array[0]
    if array.shape != (action_dim,):
        raise ValueError(f"Policy produced action shape {array.shape}, expected {(action_dim,)}")
    if not np.isfinite(array).all():
        raise ValueError("Policy produced non-finite action")
    return np.clip(array.astype(np.float32, copy=False), low, high)


def render_panel(env: Any, width: int, height: int) -> np.ndarray:
    views = [
        env.sim.render(camera_name=name, width=width, height=height)
        for name in CAMERA_NAMES
    ]
    return np.concatenate(views, axis=1)


def step_env(env: Any, action: np.ndarray) -> tuple[Any, float, bool, dict[str, Any]]:
    """Normalize Gym and Gymnasium step return conventions."""

    result = env.step(action)
    if len(result) == 4:
        obs, reward, done, info = result
        return obs, float(reward), bool(done), dict(info)
    obs, reward, terminated, truncated, info = result
    return obs, float(reward), bool(terminated or truncated), dict(info)


def task_pairs(selected: str) -> list[tuple[str, str]]:
    requested = [item.strip() for item in selected.split(",") if item.strip()]
    known = {name: instruction for name, instruction in DEFAULT_TASKS}
    unknown = sorted(set(requested) - set(known))
    if unknown:
        raise ValueError(f"Unsupported task(s): {unknown}; choose from {sorted(known)}")
    return [(name, known[name]) for name in requested]


def load_dataset_stats(dataset_root: Path) -> dict[str, dict[str, np.ndarray]]:
    """Load LeRobot stats without importing the optional datasets package."""

    stats_path = dataset_root / "meta" / "stats.json"
    raw = json.loads(stats_path.read_text(encoding="utf-8"))
    return {
        feature: {stat_name: np.asarray(stat_value, dtype=np.float32) for stat_name, stat_value in values.items()}
        for feature, values in raw.items()
    }


def ensure_optional_datasets_stub() -> None:
    """Keep policy imports usable when inference does not need HF datasets."""

    try:
        import datasets  # noqa: F401

        return
    except Exception:
        for module_name in list(sys.modules):
            if module_name == "datasets" or module_name.startswith("datasets."):
                del sys.modules[module_name]

    datasets = types.ModuleType("datasets")
    datasets.__path__ = []
    datasets.__spec__ = importlib.machinery.ModuleSpec("datasets", loader=None, is_package=True)
    datasets.Dataset = type("Dataset", (), {})
    datasets.DatasetDict = type("DatasetDict", (), {})
    datasets.IterableDataset = type("IterableDataset", (), {})
    datasets.Features = type("Features", (), {})
    datasets.load_dataset = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("HF dataset loading is unavailable in this local policy-evaluation process")
    )
    table = types.ModuleType("datasets.table")
    table.embed_table_storage = lambda value: value
    utils = types.ModuleType("datasets.utils")
    logging_module = types.ModuleType("datasets.utils.logging")
    logging_module.disable_progress_bar = lambda: None
    logging_module.enable_progress_bar = lambda: None
    utils.__path__ = []
    utils.logging = logging_module
    features = types.ModuleType("datasets.features")
    features.__path__ = []
    feature_impl = types.ModuleType("datasets.features.features")
    feature_impl.register_feature = lambda *args, **kwargs: args[0] if args else None
    features.features = feature_impl
    datasets.table = table
    datasets.utils = utils
    datasets.features = features
    sys.modules["datasets"] = datasets
    sys.modules["datasets.table"] = table
    sys.modules["datasets.utils"] = utils
    sys.modules["datasets.utils.logging"] = logging_module
    sys.modules["datasets.features"] = features
    sys.modules["datasets.features.features"] = feature_impl


def main() -> None:
    args = parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        raise ValueError("episodes and max_steps must be positive")

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # The remote AMD image has an incompatible optional coverage package.
    sys.modules.setdefault("coverage", None)

    import imageio.v2 as imageio
    import torch

    import robocasa.macros as macros

    macros.DATASET_BASE_PATH = str(args.dataset_base_path)
    from robocasa.utils.env_utils import create_env

    ensure_optional_datasets_stub()
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.control_utils import prepare_observation_for_inference

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = args.policy_path.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    task_specs = task_pairs(args.tasks)
    device = torch.device(args.device)

    dataset_stats = load_dataset_stats(dataset_root)
    policy = SmolVLAPolicy.from_pretrained(policy_path, local_files_only=True)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        dataset_stats=dataset_stats,
    )

    source_root = Path(os.environ.get("ROBOCASA_AMD_ROOT", Path(__file__).resolve().parent))
    manifest: dict[str, Any] = {
        "schema_version": "robocasa-mobile-smolvla-eval-v1",
        "status": "formal_policy_eval",
        "policy": "SmolVLA",
        "policy_path": str(policy_path),
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(dataset_root),
        "robot": "PandaOmron",
        "base": "OmronMobileBase",
        "action_dim": None,
        "state_dim": 16,
        "camera_names": CAMERA_NAMES,
        "camera_width": args.width,
        "camera_height": args.height,
        "fps": args.fps,
        "episodes_per_task": args.episodes,
        "seed_start": args.seed_start,
        "max_steps": args.max_steps,
        "tasks": [name for name, _ in task_specs],
        "episodes": [],
        "success_rate": None,
        "environment_success_predicate": "env._check_success()",
        "robocasa_commit": git_commit(source_root / "robocasa"),
        "robosuite_commit": git_commit(source_root / "robosuite"),
        "started_at_unix": time.time(),
    }

    total = 0
    successes = 0
    try:
        for task_name, instruction in task_specs:
            task_dir = output_dir / task_name
            task_dir.mkdir(parents=True, exist_ok=True)
            for episode_index in range(args.episodes):
                seed = args.seed_start + episode_index
                env = create_env(
                    task_name,
                    robots="PandaOmron",
                    camera_names=CAMERA_NAMES,
                    camera_widths=args.width,
                    camera_heights=args.height,
                    seed=seed,
                    render_onscreen=False,
                )
                policy.reset()
                episode_success = False
                video_path = task_dir / f"episode_{episode_index:03d}_seed_{seed}.mp4"
                steps = 0
                terminal = False
                info_keys: set[str] = set()
                try:
                    obs = env.reset()
                    runtime_instruction = str(env.get_ep_meta().get("lang", instruction))
                    action_dim = int(env._action_dim)
                    low, high = env.action_spec
                    manifest["action_dim"] = action_dim
                    with imageio.get_writer(
                        str(video_path), fps=args.fps, codec="libx264", macro_block_size=1
                    ) as writer:
                        for step_index in range(args.max_steps):
                            model_obs = {
                                "observation.state": state_from_observation(obs),
                                "observation.image": np.asarray(
                                    env.sim.render(
                                        camera_name=CAMERA_NAMES[0],
                                        width=args.width,
                                        height=args.height,
                                    ), dtype=np.uint8
                                ),
                                "observation.image2": np.asarray(
                                    env.sim.render(
                                        camera_name=CAMERA_NAMES[1],
                                        width=args.width,
                                        height=args.height,
                                    ), dtype=np.uint8
                                ),
                                "observation.image3": np.asarray(
                                    env.sim.render(
                                        camera_name=CAMERA_NAMES[2],
                                        width=args.width,
                                        height=args.height,
                                    ), dtype=np.uint8
                                ),
                            }
                            model_obs = prepare_observation_for_inference(
                                model_obs,
                                device=device,
                                task=runtime_instruction,
                                robot_type="PandaOmron",
                            )
                            model_obs = preprocessor(model_obs)
                            with torch.inference_mode():
                                action = policy.select_action(model_obs)
                            action = postprocessor(action)
                            action_np = action_from_policy(action, action_dim, low, high)
                            obs, reward, terminal, info = step_env(env, action_np)
                            info_keys.update(str(key) for key in info.keys())
                            writer.append_data(render_panel(env, args.width, args.height))
                            steps = step_index + 1
                            episode_success = bool(env._check_success())
                            if terminal:
                                break
                finally:
                    env.close()

                total += 1
                successes += int(episode_success)
                episode_record = {
                    "task": task_name,
                    "instruction": runtime_instruction,
                    "episode_index": episode_index,
                    "seed": seed,
                    "steps": steps,
                    "terminal": terminal,
                    "success": episode_success,
                    "video": str(video_path.relative_to(output_dir)),
                    "info_keys": sorted(info_keys),
                    "failure_stage": None if episode_success else "task_success_predicate_false",
                    "evaluation_type": "official_environment_success_predicate",
                }
                manifest["episodes"].append(episode_record)
                print(json.dumps(episode_record, ensure_ascii=True), flush=True)
    finally:
        manifest["finished_at_unix"] = time.time()
        manifest["successes"] = successes
        manifest["episodes_total"] = total
        manifest["success_rate"] = successes / total if total else None
        manifest_path = output_dir / "eval_info.json"
        manifest_path.write_text(json.dumps(jsonable(manifest), indent=2) + "\n", encoding="utf-8")
        sha_lines = [
            f"{sha256_file(path)}  {path.relative_to(output_dir)}"
            for path in sorted(output_dir.rglob("*"))
            if path.is_file() and path.name != "SHA256SUMS"
        ]
        (output_dir / "SHA256SUMS").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
        print(json.dumps({"eval_info": str(manifest_path), "successes": successes, "episodes": total}, indent=2))


if __name__ == "__main__":
    main()
