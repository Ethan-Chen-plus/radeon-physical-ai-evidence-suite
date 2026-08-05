#!/usr/bin/env python3
"""Audit RoboCasa365 PandaOmron LeRobot datasets before policy training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED_CAMERAS = {
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fetch-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def audit_task(task: str, task_root: Path, fetch_record: dict) -> dict:
    info_path = task_root / "meta" / "info.json"
    stats_path = task_root / "meta" / "stats.json"
    if not info_path.exists():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    cameras = sorted(REQUIRED_CAMERAS.intersection(features))
    action = features.get("action", {})
    checks = {
        "robot_type_pandaomron": info.get("robot_type") == "PandaOmron",
        "fps_20": info.get("fps") == 20,
        "action_12d": action.get("shape") == [12],
        "three_rgb_cameras": len(cameras) == len(REQUIRED_CAMERAS),
        "info_present": info_path.exists(),
        "stats_present": stats_path.exists(),
    }
    missing_files = [
        record["path"]
        for record in fetch_record["files"]
        if not (task_root / record["path"]).exists()
    ]
    checks["all_manifest_files_present"] = not missing_files
    return {
        "task": task,
        "repo_id": fetch_record["repo_id"],
        "revision": fetch_record["revision"],
        "root": str(task_root),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "fps": info.get("fps"),
        "robot_type": info.get("robot_type"),
        "action_shape": action.get("shape"),
        "cameras": cameras,
        "checks": checks,
        "missing_files": missing_files,
        "status": "pass" if all(checks.values()) else "blocked",
        "file_sha256": {
            record["path"]: sha256_file(task_root / record["path"])
            for record in fetch_record["files"]
            if (task_root / record["path"]).exists()
        },
    }


def main() -> None:
    args = parse_args()
    fetch_manifest = json.loads(args.fetch_manifest.read_text(encoding="utf-8"))
    tasks = []
    for task, record in fetch_manifest["tasks"].items():
        tasks.append(audit_task(task, args.root / task, record))
    output = {
        "schema_version": "robocasa-mobile-dataset-audit-v1",
        "source_manifest": str(args.fetch_manifest),
        "policy_status": "audited_data_not_trained",
        "formal_success_rate": None,
        "tasks": tasks,
        "all_checks_pass": all(task["status"] == "pass" for task in tasks),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    if not output["all_checks_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
