"""Validate a labeled video manifest and create group-safe data splits."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ("real", "deepfake", "ai_generated")
SPLITS = ("train", "validation", "test")
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
REQUIRED_COLUMNS = ("video_path", "label", "group_id", "source", "method")
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm"}


class ManifestError(ValueError):
    """A manifest is invalid or cannot produce a useful group-safe split."""


def load_manifest(manifest_path: Path, dataset_root: Path) -> list[dict[str, str]]:
    """Read and validate manifest rows and ensure videos are inside dataset_root."""
    try:
        root = dataset_root.resolve(strict=True)
    except OSError as error:
        raise ManifestError(f"Dataset root does not exist: {error}") from error
    if not root.is_dir():
        raise ManifestError(f"Dataset root is not a directory: {root}")

    rows: list[dict[str, str]] = []
    seen_paths: set[Path] = set()
    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as manifest_file:
            reader = csv.DictReader(manifest_file)
            if reader.fieldnames is None:
                raise ManifestError("Manifest is empty or missing a CSV header.")
            missing_columns = [name for name in REQUIRED_COLUMNS if name not in reader.fieldnames]
            if missing_columns:
                raise ManifestError(f"Manifest is missing columns: {', '.join(missing_columns)}")

            for line_number, row in enumerate(reader, start=2):
                values = {column: (row.get(column) or "").strip() for column in REQUIRED_COLUMNS}
                if not any(values.values()):
                    continue
                if any(not value for value in values.values()):
                    raise ManifestError(f"Line {line_number}: all required columns must be filled in.")
                if values["label"] not in LABELS:
                    raise ManifestError(
                        f"Line {line_number}: label must be one of {', '.join(LABELS)}."
                    )

                relative_path = Path(values["video_path"])
                if relative_path.is_absolute():
                    raise ManifestError(
                        f"Line {line_number}: video_path must be relative to the dataset root."
                    )
                try:
                    video_path = (root / relative_path).resolve(strict=True)
                    video_path.relative_to(root)
                except (OSError, ValueError):
                    raise ManifestError(
                        f"Line {line_number}: video path is missing or escapes the dataset root."
                    ) from None
                if not video_path.is_file():
                    raise ManifestError(f"Line {line_number}: video_path is not a file.")
                if video_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                    raise ManifestError(
                        f"Line {line_number}: use an MP4, MOV or WebM video."
                    )
                if video_path in seen_paths:
                    raise ManifestError(f"Line {line_number}: duplicate video_path.")
                seen_paths.add(video_path)
                rows.append(values)
    except OSError as error:
        raise ManifestError(f"Could not read manifest: {error}") from error

    if not rows:
        raise ManifestError("Manifest contains no video rows.")
    missing_labels = [label for label in LABELS if not any(row["label"] == label for row in rows)]
    if missing_labels:
        raise ManifestError(f"Manifest has no examples for: {', '.join(missing_labels)}.")
    return rows


def _assignment_cost(
    split_counts: dict[str, Counter[str]],
    split_sizes: Counter[str],
    label_totals: Counter[str],
    total_rows: int,
) -> float:
    cost = 0.0
    for split in SPLITS:
        for label in LABELS:
            target = label_totals[label] * SPLIT_RATIOS[split]
            scale = max(target, 1.0)
            cost += ((split_counts[split][label] - target) / scale) ** 2
        target_size = total_rows * SPLIT_RATIOS[split]
        cost += 0.5 * ((split_sizes[split] - target_size) / max(target_size, 1.0)) ** 2
    return cost


def assign_group_splits(
    rows: list[dict[str, str]], seed: int = 42
) -> dict[str, str]:
    """Assign each group wholly to one split while approximating class balance."""
    group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    label_groups: Counter[str] = Counter()
    label_totals: Counter[str] = Counter()
    for row in rows:
        group_counts[row["group_id"]][row["label"]] += 1
        label_totals[row["label"]] += 1
    for counts in group_counts.values():
        label_groups.update(counts.keys())

    too_few_groups = [label for label in LABELS if label_groups[label] < len(SPLITS)]
    if too_few_groups:
        raise ManifestError(
            "Need at least three distinct group_id values containing each class to create "
            f"train/validation/test splits. Not enough independent groups for: "
            f"{', '.join(too_few_groups)}."
        )

    rng = random.Random(seed)
    groups = list(group_counts)
    rarity = {
        group_id: max(
            group_counts[group_id][label] / label_totals[label]
            for label in group_counts[group_id]
        )
        for group_id in groups
    }
    best_assignment: dict[str, str] | None = None
    best_cost = float("inf")

    for _ in range(256):
        rng.shuffle(groups)
        ordered_groups = sorted(
            groups,
            key=lambda group_id: (rarity[group_id], len(group_counts[group_id])),
            reverse=True,
        )
        split_counts = {split: Counter() for split in SPLITS}
        split_sizes: Counter[str] = Counter()
        split_group_counts: Counter[str] = Counter()
        assignment: dict[str, str] = {}

        for group_id in ordered_groups:
            candidates: list[tuple[float, str]] = []
            for split in SPLITS:
                split_counts[split].update(group_counts[group_id])
                split_sizes[split] += sum(group_counts[group_id].values())
                split_group_counts[split] += 1
                cost = _assignment_cost(split_counts, split_sizes, label_totals, len(rows))
                cost += 0.02 * split_group_counts[split] ** 2
                candidates.append((cost, split))
                split_counts[split].subtract(group_counts[group_id])
                split_sizes[split] -= sum(group_counts[group_id].values())
                split_group_counts[split] -= 1

            minimum = min(cost for cost, _ in candidates)
            chosen = rng.choice([split for cost, split in candidates if abs(cost - minimum) < 1e-12])
            assignment[group_id] = chosen
            split_counts[chosen].update(group_counts[group_id])
            split_sizes[chosen] += sum(group_counts[group_id].values())
            split_group_counts[chosen] += 1

        if any(split_counts[split][label] == 0 for split in SPLITS for label in LABELS):
            continue
        cost = _assignment_cost(split_counts, split_sizes, label_totals, len(rows))
        if cost < best_cost:
            best_cost = cost
            best_assignment = assignment.copy()

    if best_assignment is None:
        raise ManifestError(
            "Could not make all three classes appear in every split without separating "
            "shared groups. Collect more independent groups or review group_id assignments."
        )
    return best_assignment


def write_split_manifest(
    rows: list[dict[str, str]], assignment: dict[str, str], output_path: Path
) -> None:
    """Write a derived manifest with a split column."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [*REQUIRED_COLUMNS, "split"]
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "split": assignment[row["group_id"]]})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Path to the labeled CSV.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root folder containing the local videos referenced by video_path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/split_manifest.csv"),
        help="Output CSV path (default: dataset/split_manifest.csv).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible splits.")
    arguments = parser.parse_args()

    try:
        rows = load_manifest(arguments.manifest, arguments.dataset_root)
        assignment = assign_group_splits(rows, arguments.seed)
        write_split_manifest(rows, assignment, arguments.output)
    except ManifestError as error:
        print(f"Dataset preparation failed: {error}", file=sys.stderr)
        return 1

    print(f"Validated {len(rows)} videos across {len(assignment)} independent groups.")
    for split in SPLITS:
        split_rows = [row for row in rows if assignment[row["group_id"]] == split]
        counts = Counter(row["label"] for row in split_rows)
        print(
            f"{split}: {len(split_rows)} videos; "
            + ", ".join(f"{label}={counts[label]}" for label in LABELS)
        )
    print(f"Wrote split manifest to {arguments.output.resolve()}")
    print("Group-safe splits reduce leakage; results still need evaluation on new people and sources.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
