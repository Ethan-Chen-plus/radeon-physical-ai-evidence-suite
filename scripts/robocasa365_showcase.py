"""High-resolution, multi-view RoboCasa365 showcase rollouts.

This is deliberately separate from the official evaluator.  The policy still
receives the official 256x256 camera observations; only the diagnostic video
renderer is upgraded for presentation and visual inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np

import gr00t.eval.simulation as simulation
from gr00t.eval.robot import RobotInferenceServer
from gr00t.eval.simulation import (
    MultiStepConfig,
    SimulationConfig,
    SimulationInferenceClient,
)
from gr00t.eval.wrappers.video_recording_wrapper import VideoRecorder
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from robocasa.utils.dataset_registry_utils import get_task_horizon


DEFAULT_VIEWS = (
    "robot0_agentview_center",
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)


def _force_eager_vision_attention():
    """Avoid a broken optional FlashAttention symbol in the showcase process."""
    from transformers import AutoModel

    original_from_config = AutoModel.from_config

    def from_config(config, *args, **kwargs):
        model = original_from_config(config, *args, **kwargs)
        vision_model = getattr(model, "vision_model", None)
        if vision_model is not None:
            for module in vision_model.modules():
                if module.__class__.__name__ == "SiglipAttention":
                    module.config._attn_implementation = "eager"
        return model

    AutoModel.from_config = from_config


def _base_sim(env):
    """Find the MuJoCo sim behind Gym and RoboCasa wrappers."""
    current = env
    for _ in range(8):
        if hasattr(current, "sim"):
            return current.sim
        next_env = getattr(current, "env", None)
        if next_env is None or next_env is current:
            break
        current = next_env
    raise RuntimeError("Could not find MuJoCo sim behind RoboCasa environment")


class MultiViewShowcaseWrapper(gym.Wrapper):
    """Record a labeled camera grid without changing policy observations."""

    def __init__(
        self,
        env,
        video_recorder: VideoRecorder,
        video_dir: Path,
        view_names: tuple[str, ...],
        width: int,
        height: int,
        columns: int = 2,
        steps_per_render: int = 1,
        contrast: float = 1.05,
        brightness: int = 3,
    ):
        super().__init__(env)
        self.video_dir = video_dir
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.video_recorder = video_recorder
        self.view_names = view_names
        self.width = width
        self.height = height
        self.columns = max(1, columns)
        self.rows = (len(view_names) + self.columns - 1) // self.columns
        self.steps_per_render = max(1, steps_per_render)
        self.contrast = contrast
        self.brightness = brightness
        self.file_path: Path | None = None
        self.step_count = 0
        self.is_success = False

    def _finalize(self):
        if self.file_path is None:
            return
        self.video_recorder.stop()
        if self.file_path.exists() and self.file_path.stat().st_size > 0:
            status = "success" if self.is_success else "failure"
            target = self.file_path.with_name(f"{self.file_path.stem}_{status}.mp4")
            os.replace(self.file_path, target)
        self.file_path = None

    def reset(self, **kwargs):
        self._finalize()
        result = super().reset(**kwargs)
        self.step_count = 1
        self.is_success = False
        self.file_path = self.video_dir / f"episode_{uuid.uuid4().hex}.mp4"
        return result

    def _render_view(self, camera_name: str) -> np.ndarray:
        frame = _base_sim(self.env).render(
            camera_name=camera_name,
            width=self.width,
            height=self.height,
        )
        # MuJoCo's direct offscreen buffer is vertically inverted relative to
        # the observation returned by the RoboCasa wrapper.
        frame = frame[::-1].copy()
        if self.contrast != 1.0 or self.brightness:
            frame = cv2.convertScaleAbs(
                frame, alpha=float(self.contrast), beta=int(self.brightness)
            )
        return frame

    def _compose(self) -> np.ndarray:
        tile_w = self.width
        tile_h = self.height
        canvas = np.zeros((self.rows * tile_h, self.columns * tile_w, 3), dtype=np.uint8)
        for idx, camera_name in enumerate(self.view_names):
            frame = self._render_view(camera_name)
            row, col = divmod(idx, self.columns)
            x0, y0 = col * tile_w, row * tile_h
            canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = frame
            cv2.rectangle(canvas, (x0 + 10, y0 + 10), (x0 + 300, y0 + 45), (0, 0, 0), -1)
            cv2.putText(
                canvas,
                camera_name.replace("robot0_", ""),
                (x0 + 18, y0 + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return canvas

    def step(self, action):
        result = super().step(action)
        self.step_count += 1
        if self.file_path is not None and self.step_count % self.steps_per_render == 0:
            if not self.video_recorder.is_ready():
                self.video_recorder.start(self.file_path)
            self.video_recorder.write_frame(self._compose())
            self.is_success = bool(result[-1]["success"])
        return result

    def render(self, mode="rgb_array", **kwargs):
        if self.video_recorder.is_ready():
            self.video_recorder.stop()
        return self.file_path

    def close(self):
        self._finalize()
        return super().close()


def _create_showcase_env(config: SimulationConfig, idx: int) -> gym.Env:
    env = gym.make(config.env_name, split=config.split, enable_render=True)
    video = config.video
    recorder = VideoRecorder.create_h264(
        fps=video.fps,
        codec=video.codec,
        input_pix_fmt=video.input_pix_fmt,
        crf=video.crf,
        thread_type=video.thread_type,
        thread_count=video.thread_count,
    )
    env = MultiViewShowcaseWrapper(
        env,
        recorder,
        video_dir=Path(video.video_dir),
        view_names=tuple(video.showcase_views or DEFAULT_VIEWS),
        width=video.showcase_width,
        height=video.showcase_height,
        columns=video.showcase_columns,
        steps_per_render=video.steps_per_render,
        contrast=video.showcase_contrast,
        brightness=video.showcase_brightness,
    )
    return simulation.MultiStepWrapper(
        env,
        video_delta_indices=config.multistep.video_delta_indices,
        state_delta_indices=config.multistep.state_delta_indices,
        n_action_steps=config.multistep.n_action_steps,
        max_episode_steps=config.multistep.max_episode_steps,
    )


def run_server(data_config: str, model_path: str, embodiment_tag: str, port: int):
    data_config = DATA_CONFIG_MAP[data_config]
    policy = Gr00tPolicy(
        model_path=model_path,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=embodiment_tag,
        denoising_steps=4,
    )
    RobotInferenceServer(policy, port=port).run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--task", required=True, help="One RoboCasa task, e.g. CloseFridge")
    parser.add_argument("--split", choices=["pretrain", "target"], default="pretrain")
    parser.add_argument("--data_config", default="panda_omron")
    parser.add_argument("--embodiment_tag", default="new_embodiment")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_episodes", type=int, default=2)
    parser.add_argument("--n_action_steps", type=int, default=16)
    parser.add_argument("--port", type=int, default=5569)
    parser.add_argument("--showcase_views", nargs="+", default=list(DEFAULT_VIEWS))
    parser.add_argument(
        "--showcase_columns",
        type=int,
        default=2,
        help="Number of tiles per row in the showcase grid.",
    )
    parser.add_argument("--showcase_width", type=int, default=960)
    parser.add_argument("--showcase_height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--contrast", type=float, default=1.05)
    parser.add_argument("--brightness", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Install the showcase-only environment factory. The official evaluator is
    # untouched and continues to use its 256x256 single-view recorder.
    simulation._create_single_env = _create_showcase_env
    _force_eager_vision_attention()

    server_thread = threading.Thread(
        target=run_server,
        args=(args.data_config, args.model_path, args.embodiment_tag, args.port),
        daemon=True,
    )
    server_thread.start()
    time.sleep(2)

    client = SimulationInferenceClient(host="localhost", port=args.port)
    horizon = get_task_horizon(args.task)
    video = simulation.VideoConfig(
        video_dir=str(output_dir),
        steps_per_render=1,
        fps=args.fps,
        crf=args.crf,
    )
    # Keep the upstream VideoConfig unchanged; these attributes are consumed
    # only by the showcase factory installed above.
    video.showcase_views = tuple(args.showcase_views)
    video.showcase_width = args.showcase_width
    video.showcase_height = args.showcase_height
    video.showcase_columns = max(1, args.showcase_columns)
    video.showcase_contrast = args.contrast
    video.showcase_brightness = args.brightness

    config = SimulationConfig(
        env_name=f"robocasa/{args.task}",
        split=args.split,
        n_episodes=args.n_episodes,
        n_envs=1,
        video=video,
        multistep=MultiStepConfig(
            n_action_steps=args.n_action_steps,
            max_episode_steps=horizon,
        ),
    )
    _, successes = client.run_simulation(config)
    columns = max(1, args.showcase_columns)
    rows = (len(args.showcase_views) + columns - 1) // columns
    result = {
        "task": args.task,
        "split": args.split,
        "episodes": len(successes),
        "successes": int(sum(bool(x) for x in successes)),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "views": list(args.showcase_views),
        "grid": {"columns": columns, "rows": rows},
        "per_view_resolution": [args.showcase_width, args.showcase_height],
        "output_resolution": [args.showcase_width * columns, args.showcase_height * rows],
        "fps": args.fps,
        "policy_observation_resolution": [256, 256],
        "formal_eval_unchanged": True,
        "videos": sorted(str(p) for p in output_dir.glob("*.mp4")),
    }
    (output_dir / "showcase_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
