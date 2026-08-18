import hashlib
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from worm_pose_gen.training_data import EXPECTED_RECORDS, ProxyDataset, SyntheticTierCDataset, normalize_image


def _proxy(path: Path) -> str:
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = 1
        handle.attrs["complete"] = True
        for record_number, record in enumerate(EXPECTED_RECORDS):
            group = handle.create_group(record)
            group.create_dataset("accepted", data=np.array([True, False, True]))
            group.create_dataset("accepted_sample_position", data=np.array([0, 2]))
            group.create_dataset("accepted_frame_index", data=np.array([10, 30]))
            group.create_dataset("accepted_image", data=np.full((2, 20, 30), 20 + record_number, np.uint8))
            centerline = np.full((3, 100, 2), np.nan, np.float32)
            centerline[0, :, 0] = np.linspace(0, 29, 100); centerline[0, :, 1] = 10
            centerline[2] = centerline[0]
            group.create_dataset("centerline_xy", data=centerline)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TrainingDataTests(unittest.TestCase):
    def test_normalization_is_deterministic(self) -> None:
        image = np.arange(35, dtype=np.uint8).reshape(5, 7)
        first = normalize_image(image); second = normalize_image(image)
        self.assertEqual(first.shape, (1, 192, 256))
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_proxy_recording_fold_and_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proxy.h5"; digest = _proxy(path)
            train = ProxyDataset(path, expected_sha256=digest, fold=1, split="train")
            validation = ProxyDataset(path, expected_sha256=digest, fold=1, split="validation")
            self.assertEqual(train.records, (EXPECTED_RECORDS[0], EXPECTED_RECORDS[2]))
            self.assertEqual(validation.records, (EXPECTED_RECORDS[1],))
            self.assertEqual((len(train), len(validation)), (4, 2))
            self.assertIs(train._file, validation._file)
            sample = validation[0]
            self.assertEqual(sample["image"].shape, (1, 192, 256))
            self.assertEqual(sample["centerline_xy"].shape, (100, 2))
            self.assertIsNotNone(validation._file)
            with self.assertRaises(RuntimeError):
                ProxyDataset(path, expected_sha256="0" * 64, fold=0, split="train")
            train.close(); validation.close()

    def test_tier_c_is_deterministic(self) -> None:
        dataset = SyntheticTierCDataset(2, seed=1234, profile="development")
        a, b = dataset[1], dataset[1]
        torch.testing.assert_close(a["image"], b["image"], rtol=0, atol=0)
        torch.testing.assert_close(a["centerline_xy"], b["centerline_xy"], rtol=0, atol=0)


if __name__ == "__main__": unittest.main()
