#!/usr/bin/env python3
"""Audit RoboNana training prompts against RoboTwin seen/unseen templates."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


PLACEHOLDER = re.compile(r"\{[^{}]+\}")


def template_pattern(template: str) -> re.Pattern[str]:
    """Convert one RoboTwin placeholder template into a full-string regex."""

    parts = []
    cursor = 0
    for match in PLACEHOLDER.finditer(template):
        parts.append(re.escape(template[cursor : match.start()]))
        parts.append(".+?")
        cursor = match.end()
    parts.append(re.escape(template[cursor:]))
    return re.compile("^" + "".join(parts) + "$")


def prompt_class(prompt: str, seen: list[re.Pattern], unseen: list[re.Pattern]) -> str:
    seen_match = any(pattern.fullmatch(prompt) for pattern in seen)
    unseen_match = any(pattern.fullmatch(prompt) for pattern in unseen)
    if seen_match and unseen_match:
        return "ambiguous"
    if seen_match:
        return "seen_only"
    if unseen_match:
        return "unseen_only"
    return "unmatched"


def episode_prompts(task_dir: Path) -> list[tuple[int, str]]:
    rows = []
    with (task_dir / "meta" / "episodes.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            tasks = row.get("tasks", [])
            if isinstance(tasks, str):
                prompt = tasks.strip()
            else:
                prompt = next(
                    (value.strip() for value in tasks if isinstance(value, str) and value.strip()),
                    "",
                )
            rows.append((int(row["episode_index"]), prompt))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--require-seen", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    template_root = (
        args.robotwin_root.expanduser().resolve() / "description" / "task_instruction"
    )
    task_dirs = sorted(
        path
        for split in ("Clean", "Randomized")
        for path in (dataset_root / split).iterdir()
        if (path / "meta" / "episodes.jsonl").is_file()
    )
    by_task: dict[str, dict] = defaultdict(
        lambda: {
            "episodes": 0,
            "unique_prompts": set(),
            "seen_only": 0,
            "unseen_only": 0,
            "ambiguous": 0,
            "unmatched": 0,
            "cache_mismatches": 0,
            "splits": {},
        }
    )
    for task_dir in task_dirs:
        template_path = template_root / f"{task_dir.name}.json"
        if not template_path.is_file():
            raise FileNotFoundError(f"missing RoboTwin instruction template: {template_path}")
        templates = json.loads(template_path.read_text(encoding="utf-8"))
        seen = [template_pattern(value) for value in templates.get("seen", [])]
        unseen = [template_pattern(value) for value in templates.get("unseen", [])]
        summary = by_task[task_dir.name]
        split = task_dir.parent.name
        split_count = 0
        for episode_index, prompt in episode_prompts(task_dir):
            category = prompt_class(prompt, seen, unseen)
            summary[category] += 1
            summary["episodes"] += 1
            summary["unique_prompts"].add(prompt)
            split_count += 1
            cache_sidecar = (
                task_dir / "flux_cache" / "language" / f"episode_{episode_index:06d}.json"
            )
            if not cache_sidecar.is_file():
                summary["cache_mismatches"] += 1
                continue
            cached_prompt = json.loads(cache_sidecar.read_text(encoding="utf-8")).get(
                "prompt"
            )
            if cached_prompt != prompt:
                summary["cache_mismatches"] += 1
        summary["splits"][split] = split_count

    serializable = {}
    for task, summary in sorted(by_task.items()):
        serializable[task] = {
            **summary,
            "unique_prompts": len(summary["unique_prompts"]),
        }
    totals = {
        key: sum(int(row[key]) for row in serializable.values())
        for key in (
            "episodes",
            "unique_prompts",
            "seen_only",
            "unseen_only",
            "ambiguous",
            "unmatched",
            "cache_mismatches",
        )
    }
    report = {
        "dataset_root": str(dataset_root),
        "template_root": str(template_root),
        "task_count": len(serializable),
        "totals": totals,
        "tasks": serializable,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")

    failed = totals["cache_mismatches"] > 0
    if args.require_seen:
        failed = failed or totals["unseen_only"] > 0 or totals["unmatched"] > 0
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
