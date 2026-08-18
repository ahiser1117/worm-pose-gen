import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.build_balanced_real_crop_benchmark import (
    select_balanced_cases,
    selection_digest,
    validate_balanced_hdf5,
    validate_output_targets,
    write_balanced_hdf5,
)
from worm_pose_gen.real_crop import atomic_publish


def valid_entry(recording: str, frame: int, end: str, fraction: float) -> dict:
    return {
        "source_group": recording,
        "source_frame_index": frame,
        "hidden_end": end,
        "hidden_fraction": fraction,
        "rejection_reason": None,
    }


class BalancedRealCropTests(unittest.TestCase):
    def test_sha_selection_is_balanced_deterministic_and_unique(self) -> None:
        entries = [valid_entry("r1", frame, end, fraction)
                   for end in ("head", "tail") for fraction in (.1, .2)
                   for frame in range(8)]
        kwargs = dict(seed=7, recordings=("r1",), ends=("head", "tail"),
                      fractions=(.1, .2), per_cell=3)
        first, pools = select_balanced_cases(entries, **kwargs)
        second, _ = select_balanced_cases(reversed(entries), **kwargs)
        self.assertEqual([selection_digest(7, entry) for entry in first],
                         [selection_digest(7, entry) for entry in second])
        self.assertEqual(len(first), 12)
        self.assertEqual(set(pools.values()), {8})
        identities = {(entry["source_group"], entry["source_frame_index"],
                       entry["hidden_end"], entry["hidden_fraction"]) for entry in first}
        self.assertEqual(len(identities), len(first))

    def test_selection_rejects_duplicates_and_short_cells(self) -> None:
        duplicate = valid_entry("r1", 1, "head", .1)
        with self.assertRaises(ValueError):
            select_balanced_cases([duplicate, dict(duplicate)], seed=1, recordings=("r1",),
                                  ends=("head",), fractions=(.1,), per_cell=1)
        with self.assertRaises(RuntimeError):
            select_balanced_cases([duplicate], seed=1, recordings=("r1",),
                                  ends=("head",), fractions=(.1,), per_cell=2)

    def test_collision_symlink_overwrite_and_partial_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "proxy.h5"
            metrics = root / "metrics.json"
            source.write_bytes(b"proxy")
            metrics.write_bytes(b"metrics")
            with self.assertRaises(ValueError):
                validate_output_targets(source, (source, metrics))
            alias = root / "alias.h5"
            alias.symlink_to(source)
            with self.assertRaises(ValueError):
                validate_output_targets(alias, (source, metrics))
            output = root / "benchmark.h5"
            partial = root / "benchmark.h5.partial"
            partial.symlink_to(source)
            with self.assertRaises(ValueError):
                validate_output_targets(output, (source, metrics))
            partial.unlink()
            with self.assertRaises(RuntimeError):
                atomic_publish(source, output, lambda path: (path.write_bytes(b"partial"),
                                                              (_ for _ in ()).throw(RuntimeError())))
            self.assertFalse(partial.exists())
            atomic_publish(source, output, lambda path: path.write_bytes(b"complete"))
            with self.assertRaises(FileExistsError):
                validate_output_targets(output, (source, metrics))

    def test_hdf5_schema_completion_and_exhaustive_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.h5"
            image = np.arange(192 * 256, dtype=np.float32).reshape(192, 256)
            line = np.stack((np.linspace(0, 255, 5), np.linspace(0, 191, 5)), 1)
            record = {
                "resized_image": image,
                "source_centerline_xy": line + 2,
                "transformed_centerline_xy": line,
                "support": np.asarray([True, True, False, False, False]),
                "source_group": "r1", "source_frame_index": 3,
                "accepted_image_index": 0, "sample_position": 2,
                "hidden_end": "tail", "hidden_fraction": .4,
                "source_window_k": 96,
                "source_window_bounds_xyxy_half_open": np.asarray([1, 2, 385, 290]),
                "source_to_window_transform": np.eye(3),
                "window_to_source_transform": np.eye(3),
                "source_to_resized_transform": np.eye(3),
                "resized_to_source_transform": np.eye(3),
                "selection_sha256": "a" * 64, "accepted_image_sha256": "b" * 64,
                "source_window_sha256": "c" * 64,
                "resized_image_sha256": "", "support_bitmask": "11000",
            }
            from scripts.build_balanced_real_crop_benchmark import array_sha256
            record["resized_image_sha256"] = array_sha256(image)
            write_balanced_hdf5(path, [record], {"test": "yes"})
            result = validate_balanced_hdf5(path, [record], maximum_bytes=2_000_000)
            self.assertGreater(result["size_bytes"], 0)
            self.assertEqual(len(result["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
