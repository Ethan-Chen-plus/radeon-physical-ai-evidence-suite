#!/usr/bin/env python3
"""Prepare a SmolVLA base directory for RoboCasa PandaOmron mobile data.

The source checkpoint is reused with hard links. Only the policy feature
metadata is copied and rewritten so the training processor is built for the
RoboCasa 16-D state and 12-D action contract instead of the source 6-D setup.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    return parser.parse_args()


def hardlink_tree(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"target already exists: {target}")
    shutil.copytree(source, target, copy_function=os.link, symlinks=True)


def set_shape(feature: dict, shape: list[int]) -> None:
    feature["shape"] = shape


def prepare(source: Path, target: Path) -> None:
    hardlink_tree(source, target)

    config_path = target / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.setdefault("input_features", {}).setdefault("observation.state", {})["shape"] = [16]
    config.setdefault("output_features", {}).setdefault("action", {})["shape"] = [12]
    config_path.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")

    pre_path = target / "policy_preprocessor.json"
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    for step in pre.get("steps", []):
        if step.get("registry_name") == "normalizer_processor":
            features = step.setdefault("config", {}).setdefault("features", {})
            if "observation.state" in features:
                set_shape(features["observation.state"], [16])
            if "action" in features:
                set_shape(features["action"], [12])
            step.pop("state_file", None)
    pre_path.write_text(json.dumps(pre, indent=2) + "\n", encoding="utf-8")

    post_path = target / "policy_postprocessor.json"
    post = json.loads(post_path.read_text(encoding="utf-8"))
    for step in post.get("steps", []):
        if step.get("registry_name") == "unnormalizer_processor":
            features = step.setdefault("config", {}).setdefault("features", {})
            if "action" in features:
                set_shape(features["action"], [12])
            step.pop("state_file", None)
    post_path.write_text(json.dumps(post, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "source": str(source),
        "target": str(target),
        "reuse": "hardlink_model_weights",
        "state_shape": [16],
        "action_shape": [12],
        "removed_stale_processor_state": True,
    }
    (target / "mobile_base_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    args = parse_args()
    prepare(args.source, args.target)
    print(args.target)
