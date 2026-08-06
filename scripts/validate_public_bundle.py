#!/usr/bin/env python3
"""Validate the public Track 3 evidence and release contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_robocasa() -> None:
    data = load("evidence/robocasa-official-match.json")
    assert len(data["protocol"]["tasks"]) == 16
    assert data["protocol"]["episodes_per_task"] == 50
    expected = {"gr00t": (800, 230), "pi05": (800, 142)}
    for model, (episodes, successes) in expected.items():
        record = data["models"][model]
        task_episodes = sum(item["episodes"] for item in record["tasks"])
        task_successes = sum(item["successes"] for item in record["tasks"])
        assert (record["episodes"], record["successes"]) == (episodes, successes)
        assert (task_episodes, task_successes) == (episodes, successes)


def check_dexjoco() -> None:
    data = load("evidence/dexjoco-showcase.json")
    protocol = data["protocol"]
    assert protocol["tasks"] == 11
    assert protocol["official_seed"] == 0
    assert protocol["official_successes"] == 5
    assert protocol["success_seed_tasks"] == 10
    assert len(data["tasks"]) == 10
    assert all(item["website_video"] for item in data["tasks"])


def check_discoverse() -> None:
    hd = load("evidence/discoverse-hd-showcase.json")
    assert len(hd["tasks"]) == 4
    assert all(item["successes"] == 1 for item in hd["tasks"])
    assert all(item["native_camera_videos"] == 3 for item in hd["tasks"])
    gs = load("evidence/discoverse-3dgs-showcase.json")
    assert len(gs["targets"]) == 2
    assert len(gs["videos"]) == 2
    assert all(item["gaussian_enabled"] for item in gs["targets"])


def check_g1() -> None:
    data = load("evidence/perceptive-cbf-rl-g1-eval.json")
    summary = data["summary"]
    assert summary["episodes"] == 8
    assert summary["clearance_preserved"] == 8
    assert abs(summary["min_clearance_m"] - 0.42238) < 1e-6


def check_demo() -> None:
    data = load("evidence/demo-release.json")
    assert data["duration_seconds"] == 299
    assert data["resolution"] == "1920x1080"
    assert data["fps"] == 30
    languages = {item["language"] for item in data["deliverables"]}
    assert languages == {
        "English",
        "English subtitles",
        "Unitree G1 free-base whole-body dodge film",
        "Unitree G1 poster",
        "Unitree G1 evaluation JSON",
        "Unitree G1 run manifest",
        "Unitree G1 free-base artifact checksums",
    }


def check_release_files() -> None:
    required = [
        "README.md",
        "docs/ENVIRONMENT_MATRIX.md",
        "docs/REPRODUCIBILITY.md",
        "docs/TECHNICAL_REPORT.md",
        "output/pdf/datawhale-eai-radeon-physical-ai-technical-report.pdf",
        "THIRD_PARTY_NOTICES.md",
        "LICENSE",
        "SHA256SUMS",
    ]
    for relative in required:
        path = ROOT / relative
        assert path.is_file() and path.stat().st_size > 0, relative

    entries = {}
    for line in (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        entries[relative.lstrip("* ")] = digest
    for relative in required[:-1]:
        assert relative in entries, f"missing SHA entry: {relative}"
        assert sha256(ROOT / relative) == entries[relative], relative


def main() -> None:
    check_robocasa()
    check_dexjoco()
    check_discoverse()
    check_g1()
    check_demo()
    check_release_files()
    print("PUBLIC_BUNDLE_OK")


if __name__ == "__main__":
    main()
