from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import h5py
import numpy as np

from worm_pose_gen.io import (
    GROUP_PATH, IncompleteOutputError, OutputProvenance, PoseHDF5Writer,
    SourceIdentity, open_completed_output, validate_output,
)


class PoseHDF5WriterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.source = root / "source.h5"
        with h5py.File(self.source, "w") as handle:
            handle.create_dataset("frames", data=np.zeros((3, 10, 12), dtype=np.uint8))
        self.output = root / "pose.h5"
        self.provenance = OutputProvenance(
            source=SourceIdentity.from_path(self.source, "/frames"),
            checkpoint_sha256="a" * 64, config_sha256="b" * 64,
            git_commit="deadbeef", package_versions={"h5py": h5py.__version__},
            geometry_convention="pixel centers; x right, y down; body index 0 probable head",
            image_height=10, image_width=12,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def batch(self, indices: list[int]) -> dict[str, np.ndarray]:
        count, body = len(indices), 3
        centerline = np.tile(np.array([[1, 1], [5, 5], [11, 9]], dtype=np.float32), (count, 1, 1))
        return {
            "centerline_xy": centerline,
            "tangent_angle": np.zeros((count, body), np.float32),
            "curvature": np.zeros((count, body), np.float32),
            "in_fov_mask": np.ones((count, body), bool),
            "image_support_probability": np.full((count, body), 0.8, np.float32),
            "angle_uncertainty": np.full((count, body), 0.1, np.float32),
            "head_tail_probability": np.full(count, 0.75, np.float32),
            "quality_score": np.full(count, 0.9, np.float32),
            "frame_index": np.asarray(indices, np.int64),
            "timestamp": np.asarray(indices, np.float64) / 20,
        }

    def test_streamed_publish_has_contract_and_provenance(self) -> None:
        writer = PoseHDF5Writer(self.output, body_points=3, provenance=self.provenance, chunk_frames=2)
        with writer:
            writer.append(**self.batch([0, 1]))
            self.assertTrue(writer.partial_path.exists())
            self.assertFalse(self.output.exists())
            writer.append(**self.batch([2]))
        self.assertFalse(writer.partial_path.exists())
        self.assertEqual(validate_output(self.output), 3)
        with open_completed_output(self.output) as handle:
            group = handle[GROUP_PATH]
            self.assertTrue(group.attrs["complete"])
            self.assertEqual(group["centerline_xy"].shape, (3, 3, 2))
            self.assertEqual(group["centerline_xy"].compression, "gzip")
            self.assertIsNotNone(group["centerline_xy"].chunks)
            provenance = group["provenance"].attrs
            self.assertEqual(provenance["source_dataset_path"], "/frames")
            self.assertEqual(provenance["source_size_bytes"], self.source.stat().st_size)

    def test_exception_preserves_and_rejects_incomplete_partial(self) -> None:
        writer = PoseHDF5Writer(self.output, body_points=3, provenance=self.provenance)
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with writer:
                writer.append(**self.batch([0]))
                raise RuntimeError("stop")
        self.assertTrue(writer.partial_path.exists())
        self.assertFalse(self.output.exists())
        with self.assertRaises(IncompleteOutputError):
            validate_output(writer.partial_path)
        with self.assertRaises(IncompleteOutputError):
            PoseHDF5Writer(self.output, body_points=3, provenance=self.provenance).open()

    def test_does_not_overwrite_completed_output(self) -> None:
        with PoseHDF5Writer(self.output, body_points=3, provenance=self.provenance) as writer:
            writer.append(**self.batch([0]))
        with self.assertRaises(FileExistsError):
            PoseHDF5Writer(self.output, body_points=3, provenance=self.provenance).open()

    def test_validation_rejects_bad_mapping_values_and_bounds(self) -> None:
        writer = PoseHDF5Writer(self.output, body_points=3, provenance=self.provenance)
        writer.open()
        bad = self.batch([1, 0])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            writer.append(**bad)
        bad = self.batch([0])
        bad["image_support_probability"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            writer.append(**bad)
        bad = self.batch([0])
        bad["centerline_xy"][0, 0] = [-1, 2]
        with self.assertRaisesRegex(ValueError, "half-open"):
            writer.append(**bad)
        writer.abort()

    def test_uniformly_missing_timestamps_are_allowed(self) -> None:
        batch = self.batch([0, 2])
        batch["timestamp"][:] = np.nan
        with PoseHDF5Writer(self.output, body_points=3, provenance=self.provenance) as writer:
            writer.append(**batch)
        self.assertEqual(validate_output(self.output), 2)


if __name__ == "__main__":
    unittest.main()
