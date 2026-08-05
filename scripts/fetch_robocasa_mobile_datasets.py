#!/usr/bin/env python3
"""Download a small, auditable RoboCasa365 mobile-policy dataset subset.

The script uses public Hugging Face task mirrors because the upstream Box
links are not reachable from the AMD395 host. Each repository is downloaded
into its own directory, supports byte-range resume, and records the source
revision, file sizes, and SHA256 digests in a manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


DEFAULT_TASKS = {
    "NavigateKitchen": "jellyho/robocasa365-NavigateKitchen",
    "PickPlaceCounterToCabinet": "jellyho/robocasa365-PickPlaceCounterToCabinet",
    "PickPlaceDrawerToCounter": "jellyho/robocasa365-PickPlaceDrawerToCounter",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--task", action="append", choices=sorted(DEFAULT_TASKS))
    return parser.parse_args()


def fetch_json(url: str) -> object:
    request = Request(url, headers={"User-Agent": "datawhale-eai-robocasa-fetch/1.0"})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_resumable(url: str, destination: Path, expected_size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    current_size = partial.stat().st_size if partial.exists() else 0
    if current_size == expected_size:
        partial.replace(destination)
        return
    if current_size > expected_size:
        partial.unlink()
        current_size = 0

    for attempt in range(4):
        headers = {"User-Agent": "datawhale-eai-robocasa-fetch/1.0"}
        if current_size:
            headers["Range"] = f"bytes={current_size}-"
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=120) as response:
                status = getattr(response, "status", 200)
                mode = "ab" if current_size and status == 206 else "wb"
                if mode == "wb":
                    current_size = 0
                with partial.open(mode) as handle:
                    while True:
                        block = response.read(4 * 1024 * 1024)
                        if not block:
                            break
                        handle.write(block)
            current_size = partial.stat().st_size
            if current_size == expected_size:
                partial.replace(destination)
                return
        except (HTTPError, OSError) as exc:
            current_size = partial.stat().st_size if partial.exists() else 0
            if current_size > expected_size:
                partial.unlink()
                current_size = 0
            if attempt == 3:
                raise RuntimeError(f"download failed: {url}: {exc}") from exc
            time.sleep(2**attempt)
    raise RuntimeError(f"incomplete download: {destination} ({current_size}/{expected_size})")


def main() -> None:
    args = parse_args()
    tasks = args.task or list(DEFAULT_TASKS)
    args.root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "robocasa-mobile-fetch-v1",
        "source_type": "huggingface_task_mirror",
        "revision": args.revision,
        "tasks": {},
        "policy_status": "data_download_only_not_trained",
    }

    for task in tasks:
        repo_id = DEFAULT_TASKS[task]
        tree_url = (
            f"https://huggingface.co/api/datasets/{repo_id}/tree/{args.revision}"
            "?recursive=true&expand=true"
        )
        tree = fetch_json(tree_url)
        files = [entry for entry in tree if entry.get("type") == "file"]
        task_root = args.root / task
        task_records = []
        for entry in files:
            relative_path = entry["path"]
            expected_size = int(entry.get("size", 0))
            url = (
                f"https://huggingface.co/datasets/{repo_id}/resolve/{args.revision}/"
                f"{relative_path}?download=true"
            )
            destination = task_root / relative_path
            print(f"[{task}] {relative_path} ({expected_size} bytes)", flush=True)
            download_resumable(url, destination, expected_size)
            task_records.append(
                {
                    "path": relative_path,
                    "size": expected_size,
                    "sha256": sha256_file(destination),
                    "url": url,
                }
            )
        manifest["tasks"][task] = {
            "repo_id": repo_id,
            "revision": args.revision,
            "root": str(task_root),
            "files": task_records,
        }

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.manifest}")


if __name__ == "__main__":
    main()
