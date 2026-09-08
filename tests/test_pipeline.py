"""Pipeline stages over a workspace on a synthetic recording (CPU, no network).

A dark tube on a bright background is written as an HDF5 recording; with
``checkpoint=None`` the segment stage thresholds dark pixels, so the stored
masks are exactly the rendered tube and the fit stage can recover it with
the small schedule of ``tests/test_propagation.py``.
"""

from __future__ import annotations

from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest

import h5py
import numpy as np
import torch

from worm_pose_gen.ambiguity import FLAG_NAMES
from worm_pose_gen.batch_fit import BatchFitConfig
from worm_pose_gen.latent import decode_centerline
from worm_pose_gen.mask_fit import default_width_template, render_tube_segments
from worm_pose_gen import pipeline
from worm_pose_gen.pipeline import (
    STAGES,
    DarkPixelSegmenter,
    FitParams,
    Frames,
    PriorParams,
    PropagateParams,
    SegmentParams,
    TrackParams,
    build_fit_config,
    config_from_dict,
    export_table,
    independent_copies,
    new_arrays,
    read_summary,
    restore_independent_rows,
    run_stage,
    workspace_setup,
    segment_frames,
    stage_command,
    stage_schema,
)
from worm_pose_gen.workspace import Workspace


HEIGHT, WIDTH, FRAMES = 160, 220, 6
# The SMALL schedule of tests/test_propagation.py as fit overrides: two short
# stages, no bounds, priors on the synthetic body's size.
SMALL_OVERRIDES = {
    "stage_downsample": [2, 1],
    "stage_steps": [60, 60],
    "stage_lr_scale": [1.0, 0.3],
    "stage_point_stride": [2, 1],
    "crop_padding": 16,
    "crop_multiple": 8,
    "length_bounds_px": None,
    "width_bounds_px": None,
    "length_prior_px": 150.0,
    "width_prior_px": 12.0,
    "default_length_px": 150.0,
    "default_width_px": 12.0,
    "width_shape_prior_mean": [0.0] * 6,
}
FIT_PARAMS = {
    "prior": "none", "compile": False, "init_workers": 0, "min_worm_pixels": 200, "orient": True, "overrides": SMALL_OVERRIDES,
}
SEGMENT_PARAMS = {"checkpoint": None, "flat_field": False, "min_worm_pixels": 200, "slab": 4}


def _bodies(n: int = FRAMES) -> list[np.ndarray]:
    """Masks of a worm whose second half bends progressively, as in tests/test_propagation.py."""

    template = default_width_template()
    masks = []
    for k in range(n):
        shape = np.zeros(16)
        shape[10:] = 0.25 * k
        latent = np.concatenate((shape, [0.2, 150.0], [WIDTH / 2, HEIGHT / 2]))
        curve = decode_centerline(latent)
        rendered = render_tube_segments(
            torch.as_tensor(curve, dtype=torch.float32)[None], torch.as_tensor(12.0 * template, dtype=torch.float32)[None], HEIGHT, WIDTH
        )[0]
        masks.append((rendered >= 0.5).numpy())
    return masks


def _write_recording(path: Path, n: int = FRAMES) -> list[np.ndarray]:
    """A recording of dark bodies (value 60) on a bright background (200) with mild noise; returns the masks."""

    rng = np.random.default_rng(0)
    masks = _bodies(n)
    stack = np.empty((n, HEIGHT, WIDTH), dtype=np.uint8)
    for k, mask in enumerate(masks):
        image = np.where(mask, 60.0, 200.0) + rng.normal(0, 3, (HEIGHT, WIDTH))
        stack[k] = np.clip(image, 0, 255).astype(np.uint8)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("/img_nir", data=stack)
    return masks


class ParamTests(unittest.TestCase):
    def test_from_dict_ignores_unknown_keys_and_keeps_defaults(self) -> None:
        params = FitParams.from_dict({"preset": "balanced", "threshold": 0.7, "not_a_field": 1})
        self.assertEqual(params.preset, "balanced")
        self.assertEqual(params.init_workers, FitParams().init_workers)
        segment = SegmentParams.from_dict({"threshold": 0.7, "preset": "balanced"})
        self.assertEqual(segment.threshold, 0.7)
        self.assertEqual(SegmentParams.from_dict(None), SegmentParams())
        # One dict drives every stage.
        shared = {"threshold": 0.6, "beam": 5, "track_window": 10, "prior": "none"}
        self.assertEqual(PropagateParams.from_dict(shared).beam, 5)
        self.assertEqual(TrackParams.from_dict(shared).track_window, 10)
        self.assertEqual(PriorParams.from_dict(shared).prior, "none")

    def test_defaults_match_the_script_flags(self) -> None:
        segment = SegmentParams()
        self.assertEqual((segment.threshold, segment.hole_radius, segment.fill_holes, segment.largest_only, segment.min_worm_pixels), (0.5, 8, True, True, 500))
        self.assertTrue(segment.checkpoint.endswith("checkpoints/segmenter/best.ckpt"))
        self.assertEqual(FitParams().preset, "fast")
        self.assertEqual(PriorParams().prior, "bootstrap")
        propagate = PropagateParams()
        self.assertEqual((propagate.min_score, propagate.pad, propagate.max_gap, propagate.beam, propagate.path_inview_weight), (2, 2, 3, 3, 2.0))
        self.assertEqual(TrackParams().track_refit, "clipped-deviating")

    def test_stage_schema_lists_every_field(self) -> None:
        for stage in STAGES:
            schema = stage_schema(stage)
            self.assertEqual([f["name"] for f in schema], list(asdict(pipeline.STAGE_PARAMS[stage]()).keys()))
            for entry in schema:
                self.assertEqual(set(entry), {"name", "type", "default", "help"})
        fit = {f["name"]: f for f in stage_schema("fit")}
        self.assertEqual(fit["preset"]["type"], "str")
        self.assertEqual(fit["padding"]["type"], "int")
        self.assertEqual(fit["overrides"]["type"], "dict")
        self.assertTrue(fit["preset"]["help"])

    def test_fit_config_from_params_and_back(self) -> None:
        config = build_fit_config(FitParams(compile=False, padding=48, overrides=SMALL_OVERRIDES))
        self.assertEqual(config.crop_padding, 16)  # overrides win over the flags
        self.assertEqual(config.stage_steps, (60, 60))
        self.assertIsNone(config.length_bounds_px)
        again = config_from_dict(json.loads(json.dumps(asdict(config))))
        self.assertEqual(again, config)
        with self.assertRaisesRegex(ValueError, "unknown BatchFitConfig field"):
            build_fit_config(FitParams(overrides={"no_such": 1}))

    def test_stage_command_round_trips_the_parameters(self) -> None:
        params = {"preset": "fast", "overrides": {"stage_steps": [1, 2]}, "threshold": 0.4}
        command = stage_command(Path("/tmp/ws"), "fit", params)
        self.assertEqual(command[:3], [".venv/bin/python", "-m", "worm_pose_gen.pipeline"])
        self.assertEqual(command[command.index("--workspace") + 1], "/tmp/ws")
        self.assertEqual(command[command.index("--stage") + 1], "fit")
        self.assertEqual(json.loads(command[command.index("--params") + 1]), params)
        self.assertEqual(json.loads(stage_command("/tmp/ws", "export", None)[-1]), {})


class SegmentationTests(unittest.TestCase):
    def test_dark_pixel_segmenter_and_segment_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recording = Path(directory) / "rec.h5"
            masks = _write_recording(recording)
            frames = Frames(recording, flat_field=False)
            try:
                self.assertEqual((frames.total, frames.shape), (FRAMES, (HEIGHT, WIDTH)))
                self.assertEqual(frames.read([1, 2, 3]).shape, (3, HEIGHT, WIDTH))
                self.assertEqual(frames.read([0, 2]).shape, (2, HEIGHT, WIDTH))
                model = DarkPixelSegmenter("cpu")
                probability = model.predict_probability_batch(frames.read([0]))
                self.assertGreater(probability[0][masks[0]].min(), 0.5)
                self.assertLess(probability[0][~masks[0]].max(), 0.5)
                found, stats, timing = segment_frames(frames, model, [0, 1], SegmentParams(**SEGMENT_PARAMS), torch.device("cpu"))
                self.assertEqual(len(found), 2)
                self.assertTrue(np.array_equal(found[0], masks[0]))
                self.assertEqual(stats[0]["worm_pixels"], int(masks[0].sum()))
                self.assertEqual(stats[0]["components"], 1)
                self.assertEqual(stats[0]["mask_on_border"], 0)
                self.assertEqual(set(timing), {"read", "flat_field", "network", "cleanup"})
            finally:
                frames.close()


class StageTests(unittest.TestCase):
    """The stages in pipeline order over one workspace on the synthetic recording."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        cls.recording = root / "rec.h5"
        cls.masks = _write_recording(cls.recording)
        cls.workspace = Workspace.create(root / "workspaces", "synthetic", cls.recording, 0, FRAMES - 1)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def test_stages_in_order(self) -> None:
        self._segment()
        self._fit()
        self._ambiguity()
        self._propagate()
        self._propagate_again_restores_the_independent_fit()
        self._refit_replaces_stale_ambiguity_and_hypotheses()
        self._track()
        self._export()
        self._cli()
        self._fit_scores_override_masks()
        self._imported_workspace_keeps_its_fit_configuration()
        self._stages_wait_for_the_workspace_lock()

    def _segment(self) -> None:
        messages: list[tuple[float, str]] = []
        result = run_stage(self.workspace, "segment", SEGMENT_PARAMS, device="cpu", progress=lambda p, m: messages.append((p, m)))
        self.assertEqual(result["frames"], FRAMES)
        self.assertEqual(result["frames_with_worm"], FRAMES)
        self.assertEqual(messages[-1][0], 1.0)
        self.assertEqual(self.workspace.mask_rows().tolist(), list(range(FRAMES)))
        self.assertTrue(np.array_equal(self.workspace.get_mask(3), self.masks[3]))
        state = self.workspace.load_state()
        for key in ("worm_pixels", "raw_worm_pixels", "pixels_filled", "components", "pixels_outside_largest", "mask_on_border"):
            self.assertIn(key, state)
        self.assertEqual(state["worm_pixels"].tolist(), [int(m.sum()) for m in self.masks])
        self.assertFalse(state["mask_on_border"].any())
        summary = json.loads((self.workspace.path / "summary.json").read_text())
        self.assertEqual(summary["mask_cleanup"]["min_worm_pixels"], 200)
        self.assertEqual(summary["threshold"], 0.5)
        self.assertIsNone(summary["checkpoint"])
        # Segmentation says nothing about poses, so no provenance yet.
        self.assertEqual(self.workspace.provenance_counts(), {})

    def _fit(self) -> None:
        result = run_stage(self.workspace, "fit", FIT_PARAMS, device="cpu", job="j00000001")
        self.assertEqual(result["frames_fit"], FRAMES)
        self.assertEqual(result["frames_skipped"], {"empty_mask": 0, "small_mask": 0, "no_starts": 0, "fit_error": 0})
        state = self.workspace.load_state()
        self.assertTrue(state["fitted"].all())
        self.assertEqual(state["centerline_xy"].shape, (FRAMES, 100, 2))
        self.assertGreater(float(np.min(state["iou"])), 0.8)
        self.assertTrue(np.all(np.abs(state["body_length_px"] - 150.0) < 15.0))
        self.assertTrue(np.all(state["source"] == 0))
        self.assertTrue(all(state["best_start"]))
        for key in ("iou_independent", "centerline_xy_independent", "width_profile_independent", "body_length_independent"):
            self.assertIn(key, state)
        self.assertTrue(np.array_equal(state["iou_independent"], state["iou"]))
        provenance = self.workspace.load_provenance()
        self.assertEqual(set(provenance["algorithm"].tolist()), {"independent_fit"})
        self.assertEqual(set(provenance["job"].tolist()), {"j00000001"})
        self.assertTrue(np.isfinite(provenance["time"]).all())
        self.assertFalse((self.workspace.path / "recording_prior.json").exists())
        summary = json.loads((self.workspace.path / "summary.json").read_text())
        self.assertEqual(summary["starts"], "skeleton+straight")
        self.assertEqual(config_from_dict(summary["fit_config"]).stage_steps, (60, 60))
        self.assertGreater(summary["iou"]["median"], 0.8)
        self.assertEqual(summary["frames_fitted"], FRAMES)
        self.assertIn("continuity", summary)

    def _ambiguity(self) -> None:
        result = run_stage(self.workspace, "ambiguity", {}, device="cpu")
        self.assertIsNotNone(result["ambiguity"])
        state = self.workspace.load_state()
        for name in FLAG_NAMES:
            self.assertEqual(state[f"flag_{name}"].dtype, np.bool_)
        for key in ("ambiguity_score", "score_independent", "area_ratio", "self_contact_px", "pose_jump_px"):
            self.assertIn(key, state)
        self.assertTrue(np.array_equal(state["score_independent"], state["ambiguity_score"]))
        self.assertTrue(np.isfinite(state["area_ratio"]).all())
        self.assertTrue(np.all(np.abs(state["area_ratio"] - 1.0) < 0.3))
        # A whole body far from the border trips neither the edge nor the fragment flags.
        self.assertFalse(state["flag_edge_inside"].any())
        self.assertFalse(state["flag_fragments"].any())

    def _propagate(self) -> None:
        # No frame reaches the seed score, so the pass finds no stretch and replaces nothing;
        # it still writes the (empty) hypotheses arrays and records itself in the summary.
        result = run_stage(self.workspace, "propagate", {"min_score": 9, "jump_seeds": False, "beam": 2}, device="cpu")
        self.assertEqual(result["stretches"], [])
        self.assertEqual(result["frames_replaced"], 0)
        hypotheses = self.workspace.load_hypotheses()
        self.assertEqual(hypotheses["hypotheses_centerline_xy"].shape, (FRAMES, 5, 100, 2))
        self.assertTrue(np.all(hypotheses["hypotheses_count"] == 0))
        self.assertTrue(np.all(hypotheses["path_index"] == -1))
        state = self.workspace.load_state()
        self.assertNotIn("hypotheses_count", state)
        self.assertTrue(state["fitted"].all())
        summary = json.loads((self.workspace.path / "summary.json").read_text())
        self.assertEqual(summary["propagation"]["stretches"], [])
        self.assertEqual(summary["propagation"]["frames_replaced"], 0)
        self.assertEqual(self.workspace.provenance_counts(), {"independent_fit": FRAMES})
        for key in ("latent_independent", "width_shape_independent", "crop_independent", "best_start_independent"):
            self.assertIn(key, state)
        self.assertTrue(np.array_equal(state["latent_independent"], state["latent"]))

    def _propagate_again_restores_the_independent_fit(self) -> None:
        # Pretend a previous pass replaced row 3 with a chain pose: a rerun starts from the independent fit again.
        state = self.workspace.load_state()
        original = {key: state[key][3].copy() for key in ("latent", "centerline_xy", "iou", "width_px", "crop", "best_start")}
        state["centerline_xy"][3] += 7.0
        state["latent"][3, -2:] += 7.0
        state["iou"][3] = 0.5
        state["width_px"][3] += 2.0
        state["source"][3] = 1
        state["reversed"][3] = True
        self.workspace.save_state(state)
        self.workspace.set_provenance([3], "chain_forward", "jtest")
        result = run_stage(self.workspace, "propagate", {"min_score": 9, "jump_seeds": False, "beam": 2}, device="cpu", job="j00000002")
        self.assertEqual(result["restored_rows"], 1)
        self.assertEqual(result["stretches"], [])
        state = self.workspace.load_state()
        for key, value in original.items():
            self.assertTrue(np.array_equal(state[key][3], value), key)
        self.assertEqual(int(state["source"][3]), 0)
        self.assertFalse(state["reversed"][3])
        self.assertEqual(state["score_independent"][3], state["ambiguity_score"][3])
        provenance = self.workspace.load_provenance()
        self.assertEqual((str(provenance["algorithm"][3]), str(provenance["job"][3])), ("independent_fit", "j00000002"))
        self.assertEqual(json.loads((self.workspace.path / "summary.json").read_text())["propagation"]["restored_rows"], 1)

    def _refit_replaces_stale_ambiguity_and_hypotheses(self) -> None:
        # Scores and candidates of a previous fit describe poses a refit throws away.
        state = self.workspace.load_state()
        before = state["ambiguity_score"].copy()
        state["ambiguity_score"][2:4] = 3
        state["score_independent"][2:4] = 3
        self.workspace.save_state(state)
        hypotheses = self.workspace.load_hypotheses()
        hypotheses["hypotheses_count"][2:4] = 7
        hypotheses["path_index"][2:4] = 1
        hypotheses["hypotheses_energy"][2:4, 0] = 1.0
        self.workspace.save_hypotheses(hypotheses)
        run_stage(self.workspace, "fit", FIT_PARAMS, device="cpu", job="j00000003")
        state = self.workspace.load_state()
        self.assertTrue(np.array_equal(state["ambiguity_score"], before))
        self.assertTrue(np.array_equal(state["score_independent"], before))
        hypotheses = self.workspace.load_hypotheses()
        self.assertEqual(hypotheses["hypotheses_count"][2:4].tolist(), [0, 0])
        self.assertEqual(hypotheses["path_index"][2:4].tolist(), [-1, -1])
        self.assertTrue(np.isnan(hypotheses["hypotheses_energy"][2:4]).all())
        # A propagate pass over the refit finds nothing to fix at the default seed score.
        result = run_stage(self.workspace, "propagate", {"jump_seeds": False, "beam": 2}, device="cpu")
        self.assertEqual(result["stretches"], [])

    def _track(self) -> None:
        result = run_stage(self.workspace, "track", {"track_refit": "clipped-deviating"}, device="cpu")
        self.assertEqual(result["frames_clipped"], 0)
        self.assertEqual(result["frames_refit"], 0)
        state = self.workspace.load_state()
        self.assertIn("track_length_px", state)
        self.assertIn("length_refit", state)
        self.assertTrue(np.isfinite(state["track_length_px"]).all())

    def _export(self) -> None:
        import pyarrow.parquet as pq

        result = run_stage(self.workspace, "export", {"name": "test"}, device="cpu")
        path = Path(result["path"])
        self.assertEqual(path, self.workspace.path / "exports" / "test.parquet")
        table = pq.read_table(path)
        self.assertEqual(table.num_rows, FRAMES)
        expected = {
            "frame_index", "fitted", "iou", "tube_coverage", "body_length_px", "width_px", "points_in_fov", "source", "ambiguity_score",
            "provenance_algorithm", "provenance_job", "provenance_time", "centerline_x", "centerline_y", "width_profile",
            "centroid_x", "centroid_y", "speed_px_per_frame", "mean_abs_curvature",
        } | {f"flag_{name}" for name in FLAG_NAMES}
        self.assertEqual(set(table.column_names), expected)
        rows = table.to_pylist()
        self.assertEqual(rows[0]["frame_index"], 0)
        self.assertEqual(len(rows[0]["centerline_x"]), 100)
        self.assertEqual(len(rows[0]["width_profile"]), 100)
        self.assertEqual(rows[0]["provenance_algorithm"], "independent_fit")
        self.assertIsNone(rows[0]["speed_px_per_frame"])  # no earlier frame
        self.assertTrue(all(np.isfinite(r["speed_px_per_frame"]) for r in rows[1:]))
        self.assertTrue(all(r["mean_abs_curvature"] > 0 for r in rows))
        self.assertAlmostEqual(rows[0]["centroid_x"], float(np.mean(rows[0]["centerline_x"])))

    def _cli(self) -> None:
        # The job command re-runs the ambiguity stage through the module's entry point and reports progress.
        progress_file = self.workspace.path / "progress.json"
        command = stage_command(self.workspace.path, "ambiguity", {})
        previous = os.environ.get("WORM_POSE_PROGRESS_FILE")
        os.environ["WORM_POSE_PROGRESS_FILE"] = str(progress_file)
        try:
            code = pipeline.main(command[command.index("--workspace") :])
        finally:
            if previous is None:
                del os.environ["WORM_POSE_PROGRESS_FILE"]
            else:
                os.environ["WORM_POSE_PROGRESS_FILE"] = previous
        self.assertEqual(code, 0)
        report = json.loads(progress_file.read_text())
        self.assertEqual(report["progress"], 1.0)
        self.assertIsNotNone(report["result"]["ambiguity"])

    def _fit_scores_override_masks(self) -> None:
        # The fit scores against the mask the user drew, not the stored one, and counts its pixels.
        shift = 12
        stored = self.workspace.get_mask(2)
        override = np.roll(stored, shift, axis=1)
        self.workspace.set_override_mask(2, override)
        centroid_before = float(np.mean(self.workspace.load_state()["centerline_xy"][2, :, 0]))
        result = run_stage(self.workspace, "fit", FIT_PARAMS, device="cpu", job="j00000004")
        self.assertEqual(result["frames_fit"], FRAMES)
        state = self.workspace.load_state()
        self.assertEqual(int(state["worm_pixels"][2]), int(override.sum()))
        self.assertAlmostEqual(float(np.mean(state["centerline_xy"][2, :, 0])) - centroid_before, shift, delta=3.0)
        self.assertGreater(float(state["iou"][2]), 0.8)
        # An override on a row is all the fit needs: no stored mask required.
        self.workspace.clear_override_mask(2)
        run_stage(self.workspace, "fit", FIT_PARAMS, device="cpu")
        state = self.workspace.load_state()
        self.assertEqual(int(state["worm_pixels"][2]), int(stored.sum()))
        self.assertAlmostEqual(float(np.mean(state["centerline_xy"][2, :, 0])), centroid_before, delta=3.0)

    def _imported_workspace_keeps_its_fit_configuration(self) -> None:
        # A run directory from this workspace, imported: the first stage on it must not lose the run's fit configuration.
        run_dir = Path(self.directory.name) / "runs" / "demo_run"
        run_dir.mkdir(parents=True)
        np.savez(run_dir / "poses.npz", **self.workspace.load_arrays())
        summary = json.loads((self.workspace.path / "summary.json").read_text())
        summary["starts"] = "skeleton+reversed"
        (run_dir / "summary.json").write_text(json.dumps(summary))
        imported = Workspace.import_run(Path(self.directory.name) / "workspaces", run_dir, "imported")
        self.assertFalse((imported.path / "summary.json").exists())
        self.assertEqual(workspace_setup(imported).start_set, "skeleton+reversed")
        run_stage(imported, "ambiguity", {}, device="cpu")
        self.assertTrue((imported.path / "summary.json").exists())
        merged = read_summary(imported)
        self.assertEqual(config_from_dict(merged["fit_config"]).stage_steps, (60, 60))
        self.assertEqual(merged["starts"], "skeleton+reversed")
        for key in ("threshold", "mask_cleanup", "preset", "fit_params", "ambiguity", "finished_at"):
            self.assertIn(key, merged)
        self.assertEqual(workspace_setup(imported).start_set, "skeleton+reversed")
        self.assertEqual(workspace_setup(imported).config.stage_steps, (60, 60))
        run_stage(imported, "export", {"name": "imported"}, device="cpu")
        self.assertEqual(read_summary(imported)["starts"], "skeleton+reversed")
        shutil.rmtree(imported.path)

    def _stages_wait_for_the_workspace_lock(self) -> None:
        # Another process holding the workspace's lock (a running stage) delays the stage until it lets go.
        handle = open(self.workspace.path / ".lock", "w")
        fcntl.flock(handle, fcntl.LOCK_EX)
        finished: list[float] = []
        thread = threading.Thread(target=lambda: (run_stage(self.workspace, "export", {"name": "locked"}, device="cpu"), finished.append(time.monotonic())))
        thread.start()
        time.sleep(0.4)
        self.assertEqual(finished, [])
        released = time.monotonic()
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()
        thread.join(timeout=30)
        self.assertEqual(len(finished), 1)
        self.assertGreaterEqual(finished[0], released)
        self.assertTrue((self.workspace.path / "exports" / "locked.parquet").exists())


class IndependentCopyTests(unittest.TestCase):
    def test_copies_and_restore_by_provenance(self) -> None:
        config = BatchFitConfig()
        arrays = new_arrays(np.arange(4), config)
        arrays["fitted"][:] = True
        arrays["latent"][:] = np.arange(4)[:, None]
        arrays["iou"][:] = 0.9
        arrays["best_start"][:] = "skeleton_longest_path"
        independent_copies(arrays)
        self.assertTrue(np.array_equal(arrays["latent_independent"], arrays["latent"]))
        # Rows 1 and 2 get chain poses; a refit of row 3 updates only its copy.
        arrays["latent"][1:3] = -1.0
        arrays["iou"][1:3] = 0.4
        arrays["source"][1:3] = (1, 2)
        arrays["latent"][3] = 30.0
        independent_copies(arrays, np.array([3]))
        self.assertEqual(arrays["latent_independent"][3, 0], 30.0)
        self.assertEqual(arrays["latent_independent"][1, 0], 1.0)
        algorithm = np.array(["independent_fit", "chain_forward", "independent_refit", "independent_fit"])
        self.assertEqual(restore_independent_rows(arrays, algorithm), [1, 2])
        self.assertEqual(arrays["latent"][1:3, 0].tolist(), [1.0, 2.0])
        self.assertEqual(arrays["iou"][1:3].tolist(), [0.9, 0.9])
        self.assertEqual(arrays["source"][1:3].tolist(), [0, 0])
        # Nothing to restore: no such provenance, or a workspace predating the full copies.
        self.assertEqual(restore_independent_rows(arrays, np.array(["independent_fit"] * 4)), [])
        del arrays["latent_independent"]
        self.assertEqual(restore_independent_rows(arrays, algorithm), [])


class ExportTableTests(unittest.TestCase):
    def test_kinematics_from_synthetic_arrays(self) -> None:
        config = BatchFitConfig()
        arrays = new_arrays(np.array([10, 11, 13]), config)
        template = default_width_template()
        for row, shift in enumerate((0.0, 3.0, 9.0)):
            latent = np.concatenate((np.zeros(16), [0.0, 100.0], [50.0 + shift, 40.0]))
            arrays["centerline_xy"][row] = decode_centerline(latent)
            arrays["width_profile"][row] = 10.0 * template
            arrays["fitted"][row] = True
            arrays["iou"][row] = 0.9
        arrays["fitted"][1] = False
        table = export_table(arrays, {"algorithm": np.array(["a", "", "b"]), "job": np.array(["j", "", "j"]), "time": np.array([1.0, np.nan, 2.0])})
        rows = table.to_pylist()
        self.assertIsNone(rows[1]["centerline_x"])
        self.assertIsNone(rows[1]["centroid_x"])
        # Row 2 follows row 0 (row 1 is not fitted): 9 px over 3 frames.
        self.assertAlmostEqual(rows[2]["speed_px_per_frame"], 3.0, places=6)
        self.assertAlmostEqual(rows[2]["mean_abs_curvature"], 0.0, places=6)  # a straight body
        self.assertEqual(rows[2]["provenance_algorithm"], "b")
        self.assertEqual(rows[0]["frame_index"], 10)


if __name__ == "__main__":
    unittest.main()
