#!/usr/bin/env python3
"""Find one successful rollout per DexJoCo multi-task task on new seeds."""

from __future__ import annotations

import json
import os
import pathlib
import socket
import subprocess
import sys
import time


ROOT = pathlib.Path(os.environ.get("DEXJOCO_ROOT", ".")).expanduser().resolve()
PYTHON = os.environ.get("OPENPI_PYTHON", sys.executable)
EVALUATOR = os.environ.get("DEXJOCO_EVAL", "dexjoco-openpi-eval")
CHECKPOINT = pathlib.Path(
    os.environ.get("CHECKPOINT", ROOT / "checkpoints/pi05_dexjoco_multi_task")
).expanduser().resolve()
BASE_OUT = pathlib.Path(
    os.environ.get("BASE_OUT", ROOT / "runs/amd_openpi_jax010_multitask")
).expanduser().resolve()
OUT = pathlib.Path(
    os.environ.get("OUT", ROOT / "runs/amd_openpi_jax010_multitask_recovery")
).expanduser().resolve()
PORT = int(os.environ.get("PORT", "8021"))
MAX_SEED = int(os.environ.get("MAX_SEED", "10"))
TASKS = [
    "bimanual_assembly",
    "bimanual_hanoi",
    "bimanual_microwave_cook",
    "bimanual_photograph",
    "bimanual_unlock_ipad",
    "click_mouse",
    "fold_glasses",
    "hammer_nail",
    "pick_bucket",
    "pinch_tongs",
    "water_plant",
]


def print_help() -> None:
    print(
        "DexJoCo Pi0.5 recovery evaluation\n\n"
        "Configuration is supplied through environment variables:\n"
        "  DEXJOCO_ROOT   DexJoCo checkout\n"
        "  OPENPI_PYTHON  ROCm JAX Python executable\n"
        "  DEXJOCO_EVAL   dexjoco-openpi-eval executable\n"
        "  CHECKPOINT     multi-task Pi0.5 checkpoint\n"
        "  BASE_OUT       completed official seed-0 result directory\n"
        "  OUT            recovery result directory\n"
        "  PORT           policy server port (default: 8021)\n"
        "  MAX_SEED       last recovery seed (default: 10)"
    )


def wait_for_port(port: int) -> None:
    for _ in range(300):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(1)
    raise RuntimeError(f"policy server did not open port {port}")


def marker(path: pathlib.Path, name: str) -> bool:
    return (path / name).exists()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    complete = OUT / "recovery_eval_complete"
    if complete.exists():
        print(f"already_complete: {complete}")
        return 0

    env = os.environ.copy()
    env.update(
        {
            "MUJOCO_GL": "egl",
            "JAX_PLATFORMS": "rocm",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.40",
            "DEXJOCO_PYTORCH_COMPILE_MODE": "none",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": f"{ROOT}/openpi/src:{ROOT}/openpi/packages/openpi-client/src",
        }
    )
    server_log = (OUT / "policy_server.log").open("a", encoding="utf-8")
    server = subprocess.Popen(
        [
            PYTHON,
            "scripts/serve_policy.py",
            "--port",
            str(PORT),
            "policy:checkpoint",
            "--policy.config",
            "multi_task",
            "--policy.dir",
            str(CHECKPOINT),
        ],
        cwd=ROOT / "openpi",
        env=env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
    )
    try:
        wait_for_port(PORT)
        base = json.loads((BASE_OUT / "multitask_summary.json").read_text())
        base_success = {x["task"]: x["successes"] > 0 for x in base["tasks"]}
        log_path = OUT / "recovery_eval.log"
        with log_path.open("a", encoding="utf-8") as log:
            for task in TASKS:
                task_out = OUT / task
                task_out.mkdir(parents=True, exist_ok=True)
                first = task_out / "first_success.json"
                exhausted = task_out / f"no_success_after_{MAX_SEED}_seeds"
                if first.exists() or exhausted.exists():
                    print(f"skip_existing: {task}", flush=True)
                    continue
                if base_success.get(task, False):
                    first.write_text(
                        json.dumps({"task": task, "source": "official_seed0", "seed": 0, "success": True})
                        + "\n"
                    )
                    print(f"already_successful_official_seed0: {task}", flush=True)
                    continue

                found = False
                for seed in range(1, MAX_SEED + 1):
                    attempt = task_out / f"seed_{seed}"
                    if marker(attempt, "success_rate_1_1.txt"):
                        found = True
                        break
                    if marker(attempt, "success_rate_0_1.txt"):
                        continue
                    attempt.mkdir(parents=True, exist_ok=True)
                    command = [
                        EVALUATOR,
                        f"--config={ROOT}/configs/multi_task/{task}.yaml",
                        f"--seed={seed}",
                        f"--port={PORT}",
                        f"--output={attempt}",
                        "--render-mode=rgb_array",
                        "--pad-state-dim46",
                        "--episodes=1",
                    ]
                    line = f"start: {task} seed={seed} {time.strftime('%Y-%m-%dT%H:%M:%S%z')}"
                    print(line, flush=True)
                    log.write(line + "\n")
                    log.flush()
                    with log_path.open("a", encoding="utf-8") as episode_log:
                        result = subprocess.run(
                            command,
                            cwd=ROOT,
                            env=env,
                            stdout=episode_log,
                            stderr=subprocess.STDOUT,
                            check=False,
                        )
                    done = f"done: {task} seed={seed} exit={result.returncode}"
                    print(done, flush=True)
                    log.write(done + "\n")
                    log.flush()
                    if marker(attempt, "success_rate_1_1.txt"):
                        first.write_text(
                            json.dumps({"task": task, "source": "recovery", "seed": seed, "success": True})
                            + "\n"
                        )
                        found = True
                        break
                if not found:
                    exhausted.touch()
                    print(f"no_success_after_{MAX_SEED}_seeds: {task}", flush=True)

        rows = []
        for item in base["tasks"]:
            task = item["task"]
            row = {
                "task": task,
                "official_seed0_success": bool(item["successes"]),
                "first_success_seed": 0 if item["successes"] else None,
                "recovery_attempt_limit": MAX_SEED,
                "recovery_video_root": str(OUT / task),
            }
            first = OUT / task / "first_success.json"
            if first.exists():
                data = json.loads(first.read_text())
                if data.get("source") == "recovery":
                    row["first_success_seed"] = data["seed"]
            row["has_success_video"] = row["first_success_seed"] is not None
            row["no_success_within_attempt_limit"] = (OUT / task / f"no_success_after_{MAX_SEED}_seeds").exists()
            rows.append(row)
        summary = {
            "protocol": {
                "base_result": str(BASE_OUT / "multitask_summary.json"),
                "official_seed": 0,
                "recovery_seeds": list(range(1, MAX_SEED + 1)),
                "episodes_per_recovery_seed": 1,
                "checkpoint": str(CHECKPOINT),
            },
            "tasks": rows,
        }
        (OUT / "recovery_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        complete.write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
        return 0
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
        server_log.close()


if __name__ == "__main__":
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print_help()
        raise SystemExit(0)
    raise SystemExit(main())
