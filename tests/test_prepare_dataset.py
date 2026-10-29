import csv
import tempfile
import unittest
from pathlib import Path

from prepare_dataset import (
    LABELS,
    ManifestError,
    assign_group_splits,
    load_manifest,
    write_split_manifest,
)


class DatasetManifestTests(unittest.TestCase):
    def _make_dataset(self, root: Path) -> Path:
        manifest_path = root / "manifest.csv"
        rows = []
        for group_index in range(6):
            for label_index, label in enumerate(LABELS):
                relative_path = Path(label) / f"clip_{group_index}.mp4"
                video_path = root / relative_path
                video_path.parent.mkdir(parents=True, exist_ok=True)
                video_path.write_bytes(b"test fixture")
                group_id = (
                    f"source_{group_index}"
                    if label in {"real", "deepfake"}
                    else f"generated_{group_index}"
                )
                rows.append(
                    {
                        "video_path": relative_path.as_posix(),
                        "label": label,
                        "group_id": group_id,
                        "source": "synthetic_test_fixture",
                        "method": "test_only",
                    }
                )
        with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
            writer = csv.DictWriter(manifest_file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return manifest_path

    def test_validates_manifest_and_creates_reproducible_group_splits(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = self._make_dataset(root)

            rows = load_manifest(manifest_path, root)
            first_assignment = assign_group_splits(rows, seed=7)
            second_assignment = assign_group_splits(rows, seed=7)

            self.assertEqual(first_assignment, second_assignment)
            self.assertEqual(set(first_assignment.values()), {"train", "validation", "test"})
            self.assertEqual(len(first_assignment), len({row["group_id"] for row in rows}))
            for label in LABELS:
                for split in ("train", "validation", "test"):
                    self.assertTrue(
                        any(
                            row["label"] == label and first_assignment[row["group_id"]] == split
                            for row in rows
                        )
                    )

            output_path = root / "splits" / "split_manifest.csv"
            write_split_manifest(rows, first_assignment, output_path)
            with output_path.open(encoding="utf-8", newline="") as output_file:
                output_rows = list(csv.DictReader(output_file))
            self.assertEqual(len(output_rows), len(rows))
            self.assertTrue(all(row["split"] == first_assignment[row["group_id"]] for row in output_rows))
            for group_id in first_assignment:
                group_splits = {row["split"] for row in output_rows if row["group_id"] == group_id}
                self.assertEqual(len(group_splits), 1)

    def test_rejects_paths_that_escape_dataset_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "dataset"
            root.mkdir()
            outside_video = Path(temporary_directory) / "outside.mp4"
            outside_video.write_bytes(b"test fixture")
            manifest_path = root / "manifest.csv"
            manifest_path.write_text(
                "video_path,label,group_id,source,method\n"
                "../outside.mp4,real,person01,recording,camera\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ManifestError, "missing or escapes"):
                load_manifest(manifest_path, root)

    def test_rejects_too_few_independent_groups_per_class(self):
        rows = [
            {
                "video_path": f"{label}_{index}.mp4",
                "label": label,
                "group_id": f"{label}_group_{index}",
                "source": "test",
                "method": "test",
            }
            for label in LABELS
            for index in range(2)
        ]

        with self.assertRaisesRegex(ManifestError, "at least three distinct"):
            assign_group_splits(rows)


if __name__ == "__main__":
    unittest.main()
