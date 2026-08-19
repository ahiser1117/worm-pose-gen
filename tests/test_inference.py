from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from worm_pose_gen.inference import (
    VALIDATION_STATUS,
    ExploratoryInferenceRequired,
    ExploratoryPoseInference,
    InferenceContractError,
    TemporalInferenceUnsupported,
    infer_hdf5,
)
from worm_pose_gen.io import GROUP_PATH, open_completed_output, sha256_file, validate_output
from worm_pose_gen.model import WormProposalModule


def _write_synthetic_source(path: Path, *, frames: int = 5, height: int = 73, width: int = 97) -> None:
    values = np.arange(frames * height * width, dtype=np.uint32)
    images = np.remainder(values, 256).astype(np.uint8).reshape(frames, height, width)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("/frames", data=images, chunks=(1, height, width))


def _write_checkpoint(path: Path, *, degenerate: bool = False) -> None:
    """Write a loadable Lightning checkpoint with input-independent geometry."""

    model = WormProposalModule(variant="coordinate")
    final = model.head[-1]
    if not isinstance(final, torch.nn.Linear):
        raise AssertionError("expected the proposal head to end in a Linear layer")
    with torch.no_grad():
        final.weight.zero_()
        final.bias.zero_()
        if not degenerate:
            x = torch.linspace(0.1, 0.9, model.hparams.num_points)
            y = torch.full_like(x, 0.5)
            normalized = torch.stack((x, y), dim=-1)
            final.bias[: 2 * model.hparams.num_points].copy_((normalized - 0.5).reshape(-1))
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": "2.5.5",
            "epoch": 0,
            "global_step": 1,
        },
        path,
    )


def _write_final_config(
    path: Path,
    checkpoint: Path,
    **overrides: object,
) -> None:
    declaration: dict[str, object] = {
        "enabled_only_with_explicit_opt_in": True,
        "validation_status": VALIDATION_STATUS,
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_variant": "coordinate",
        "encoder_pool_output": [2, 2],
        "body_points": 100,
        "input_height": 192,
        "input_width": 256,
    }
    declaration.update(overrides)
    path.write_text(
        json.dumps(
            {"deployment_authorized": False, "exploratory_inference": declaration},
            sort_keys=True,
        ),
        encoding="utf-8",
    )


class InferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.directory = Path(self._temporary_directory.name)

    def test_opt_in_is_required_before_checkpoint_access(self) -> None:
        with self.assertRaisesRegex(ExploratoryInferenceRequired, "allow-exploratory"):
            ExploratoryPoseInference.from_checkpoint(
                self.directory / "does-not-exist.ckpt",
                original_height=73,
                original_width=97,
                config_path=self.directory / "does-not-exist.yaml",
                device="cpu",
            )

        with self.assertRaises(ExploratoryInferenceRequired):
            infer_hdf5(
                source_path=self.directory / "does-not-exist.h5",
                dataset_path="/frames",
                checkpoint_path=self.directory / "does-not-exist.ckpt",
                config_path=self.directory / "does-not-exist.yaml",
                output_path=self.directory / "output.h5",
                device="cpu",
            )

    def test_checkpoint_one_frame_and_batch_exact_original_mapping(self) -> None:
        checkpoint = self.directory / "model.ckpt"
        config = self.directory / "final.yaml"
        _write_checkpoint(checkpoint)
        _write_final_config(config, checkpoint)
        engine = ExploratoryPoseInference.from_checkpoint(
            checkpoint,
            original_height=73,
            original_width=97,
            config_path=config,
            device="cpu",
            allow_exploratory=True,
        )
        frame = np.zeros((73, 97), dtype=np.uint8)
        one = engine.predict_frame(frame)
        batch = engine.predict_batch(np.stack((frame, frame), axis=0))

        self.assertEqual(len(one), 1)
        self.assertEqual(batch.centerline_xy.shape, (2, 100, 2))
        self.assertAlmostEqual(batch.centerline_xy[0, 0, 0].item(), 0.1 * 97, places=5)
        self.assertAlmostEqual(batch.centerline_xy[0, -1, 0].item(), 0.9 * 97, places=5)
        torch.testing.assert_close(
            batch.centerline_xy[..., 1], torch.full((2, 100), 0.5 * 73)
        )
        torch.testing.assert_close(
            batch.tangent_angle,
            torch.zeros_like(batch.tangent_angle),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            batch.curvature, torch.zeros_like(batch.curvature), atol=1e-6, rtol=0
        )
        self.assertTrue(bool(batch.in_fov_mask.all()))
        self.assertTrue(bool(torch.all(batch.image_support_probability == 0.5)))
        self.assertTrue(bool(torch.all(batch.angle_uncertainty == torch.pi)))
        self.assertTrue(bool(torch.all(batch.head_tail_probability == 0.5)))
        self.assertTrue(bool(torch.all(batch.quality_score == 0)))
        self.assertEqual(batch.validation_status, VALIDATION_STATUS)

    def test_temporal_window_api_is_explicitly_unsupported(self) -> None:
        checkpoint = self.directory / "model.ckpt"
        config = self.directory / "final.yaml"
        _write_checkpoint(checkpoint)
        _write_final_config(config, checkpoint)
        engine = ExploratoryPoseInference.from_checkpoint(
            checkpoint,
            original_height=73,
            original_width=97,
            config_path=config,
            device="cpu",
            allow_exploratory=True,
        )
        status = engine.temporal_window_status()
        self.assertFalse(status.supported)
        self.assertEqual(status.validation_status, VALIDATION_STATUS)
        with self.assertRaisesRegex(TemporalInferenceUnsupported, "unsupported"):
            engine.predict_temporal_window(np.zeros((3, 73, 97), dtype=np.uint8))

    def test_floating_frame_intensity_contract_is_fail_closed(self) -> None:
        checkpoint = self.directory / "model.ckpt"
        config = self.directory / "final.yaml"
        _write_checkpoint(checkpoint)
        _write_final_config(config, checkpoint)
        engine = ExploratoryPoseInference.from_checkpoint(
            checkpoint,
            original_height=73,
            original_width=97,
            config_path=config,
            device="cpu",
            allow_exploratory=True,
        )
        with self.assertRaisesRegex(ValueError, "finite and lie in"):
            engine.predict_frame(np.full((73, 97), 255.0, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "finite and lie in"):
            engine.predict_frame(np.full((73, 97), np.nan, dtype=np.float32))
        prediction = engine.predict_frame(np.full((73, 97), 0.5, dtype=np.float32))
        self.assertEqual(len(prediction), 1)

    def test_streamed_output_schema_provenance_and_semantics(self) -> None:
        source = self.directory / "synthetic.h5"
        checkpoint = self.directory / "model.ckpt"
        config = self.directory / "final.yaml"
        output = self.directory / "poses.h5"
        _write_synthetic_source(source)
        _write_checkpoint(checkpoint)
        _write_final_config(config, checkpoint)
        source_hash = sha256_file(source)

        result = infer_hdf5(
            source_path=source,
            dataset_path="/frames",
            checkpoint_path=checkpoint,
            config_path=config,
            output_path=output,
            allow_exploratory=True,
            device="cpu",
            batch_size=2,
            start=1,
            stop=5,
            frame_rate=10.0,
        )

        self.assertEqual(result, output)
        self.assertEqual(validate_output(output), 4)
        self.assertFalse(Path(f"{output}.partial").exists())
        self.assertEqual(sha256_file(source), source_hash)
        with open_completed_output(output) as handle:
            group = handle[GROUP_PATH]
            provenance = group["provenance"].attrs
            self.assertTrue(bool(group.attrs["complete"]))
            self.assertEqual(group.attrs["validation_status"], VALIDATION_STATUS)
            self.assertFalse(bool(group.attrs["temporal_inference_supported"]))
            self.assertEqual(provenance["checkpoint_sha256"], sha256_file(checkpoint))
            self.assertEqual(provenance["config_sha256"], sha256_file(config))
            self.assertEqual(provenance["source_dataset_path"], "/frames")
            self.assertEqual(int(provenance["image_height"]), 73)
            self.assertEqual(int(provenance["image_width"]), 97)
            np.testing.assert_array_equal(group["frame_index"][:], [1, 2, 3, 4])
            np.testing.assert_allclose(group["timestamp"][:], [0.1, 0.2, 0.3, 0.4])
            self.assertIn(
                "uncalibrated sentinel", group["angle_uncertainty"].attrs["semantics"]
            )
            self.assertIn(
                "unknown orientation", group["head_tail_probability"].attrs["semantics"]
            )
            self.assertIn(
                "distinct from geometric FOV",
                group["image_support_probability"].attrs["semantics"],
            )

    def test_inference_failure_preserves_incomplete_partial(self) -> None:
        source = self.directory / "synthetic.h5"
        checkpoint = self.directory / "degenerate.ckpt"
        config = self.directory / "final.yaml"
        output = self.directory / "poses.h5"
        _write_synthetic_source(source, frames=2)
        _write_checkpoint(checkpoint, degenerate=True)
        _write_final_config(config, checkpoint)

        with self.assertRaisesRegex(InferenceContractError, "degenerate"):
            infer_hdf5(
                source_path=source,
                dataset_path="/frames",
                checkpoint_path=checkpoint,
                config_path=config,
                output_path=output,
                allow_exploratory=True,
                device="cpu",
                batch_size=1,
            )

        partial = Path(f"{output}.partial")
        self.assertFalse(output.exists())
        self.assertTrue(partial.exists())
        with h5py.File(partial, "r") as handle:
            self.assertFalse(bool(handle[GROUP_PATH].attrs["complete"]))
            self.assertEqual(handle[GROUP_PATH].attrs["validation_status"], VALIDATION_STATUS)

    def test_source_output_collision_is_refused_without_source_mutation(self) -> None:
        source = self.directory / "synthetic.h5"
        checkpoint = self.directory / "model.ckpt"
        config = self.directory / "final.yaml"
        _write_synthetic_source(source)
        _write_checkpoint(checkpoint)
        _write_final_config(config, checkpoint)
        before = sha256_file(source)

        with self.assertRaisesRegex(ValueError, "aliases the read-only source"):
            infer_hdf5(
                source_path=source,
                dataset_path="/frames",
                checkpoint_path=checkpoint,
                config_path=config,
                output_path=source,
                allow_exploratory=True,
                device="cpu",
                batch_size=2,
            )

        self.assertEqual(sha256_file(source), before)
        with h5py.File(source, "r") as handle:
            self.assertEqual(handle["/frames"].shape, (5, 73, 97))
        self.assertFalse(Path(f"{source}.partial").exists())

    def test_invalid_source_schema_fails_without_output(self) -> None:
        source = self.directory / "bad.h5"
        checkpoint = self.directory / "model.ckpt"
        config = self.directory / "final.yaml"
        output = self.directory / "poses.h5"
        with h5py.File(source, "w") as handle:
            handle.create_dataset("/frames", data=np.zeros((2, 73, 97, 3), dtype=np.uint8))
        _write_checkpoint(checkpoint)
        _write_final_config(config, checkpoint)

        with self.assertRaisesRegex(ValueError, "ndim"):
            infer_hdf5(
                source_path=source,
                dataset_path="/frames",
                checkpoint_path=checkpoint,
                config_path=config,
                output_path=output,
                allow_exploratory=True,
                device="cpu",
            )
        self.assertFalse(output.exists())
        self.assertFalse(Path(f"{output}.partial").exists())

    def test_ambiguous_integer_intensity_schema_fails_closed(self) -> None:
        source = self.directory / "uint16.h5"
        checkpoint = self.directory / "model.ckpt"
        config = self.directory / "final.yaml"
        output = self.directory / "poses.h5"
        with h5py.File(source, "w") as handle:
            handle.create_dataset("/frames", data=np.zeros((2, 73, 97), dtype=np.uint16))
        _write_checkpoint(checkpoint)
        _write_final_config(config, checkpoint)

        with self.assertRaisesRegex(TypeError, "dtype"):
            infer_hdf5(
                source_path=source,
                dataset_path="/frames",
                checkpoint_path=checkpoint,
                config_path=config,
                output_path=output,
                allow_exploratory=True,
                device="cpu",
            )
        self.assertFalse(output.exists())

    def test_declaration_mismatches_fail_before_source_or_output_access(self) -> None:
        checkpoint = self.directory / "model.ckpt"
        _write_checkpoint(checkpoint)
        cases: dict[str, object] = {
            "checkpoint_sha256": "0" * 64,
            "validation_status": "accepted",
            "model_variant": "intrinsic",
            "encoder_pool_output": [4, 4],
            "body_points": 99,
            "input_height": 191,
            "input_width": 255,
        }
        for field, mismatch in cases.items():
            with self.subTest(field=field):
                config = self.directory / f"final-{field}.yaml"
                output = self.directory / f"poses-{field}.h5"
                _write_final_config(config, checkpoint, **{field: mismatch})
                with self.assertRaisesRegex(InferenceContractError, field):
                    infer_hdf5(
                        source_path=self.directory / "source-must-not-be-opened.h5",
                        dataset_path="/frames",
                        checkpoint_path=checkpoint,
                        config_path=config,
                        output_path=output,
                        allow_exploratory=True,
                        device="cpu",
                    )
                self.assertFalse(output.exists())
                self.assertFalse(Path(f"{output}.partial").exists())


if __name__ == "__main__":
    unittest.main()
