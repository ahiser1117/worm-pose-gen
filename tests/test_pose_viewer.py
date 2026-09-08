from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import Request, urlopen

import h5py
import numpy as np

from worm_pose_gen.ambiguity import FLAG_NAMES
from worm_pose_gen.latent import decode_centerline
from worm_pose_gen.mask_fit import default_width_template
from worm_pose_gen.pose_viewer import (
    ViewerState,
    classify_frame,
    create_server,
    prior_width_profile,
    run_entry,
    signed_curvature,
)


HEIGHT, WIDTH, FRAMES = 96, 128, 6


def _write_recording(path: Path) -> None:
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    stack = np.empty((FRAMES, HEIGHT, WIDTH), dtype=np.uint8)
    for index in range(FRAMES):
        body = np.abs(yy - (48 + 10 * np.sin(xx / 20 + index))) < 6
        body &= (xx > 12) & (xx < 116)
        image = 190.0 - 80 * body + rng.normal(0, 3, (HEIGHT, WIDTH))
        stack[index] = np.clip(image, 0, 255).astype(np.uint8)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("/img_nir", data=stack)


def _write_run(path: Path, recording: Path, *, first: int = 0, count: int = FRAMES, independent: bool = True) -> None:
    """A run directory with the arrays the fitter stores, frames ``first`` onward."""

    path.mkdir(parents=True)
    n_points = 100
    frame_index = np.arange(first, first + count)
    latent = np.concatenate((np.zeros(16), [0.0, 100.0], [WIDTH / 2, HEIGHT / 2]))
    curve = decode_centerline(latent)
    template = default_width_template(n_points)
    arrays: dict[str, np.ndarray] = {
        "frame_index": frame_index,
        "fitted": np.ones(count, dtype=bool),
        "latent": np.tile(latent, (count, 1)),
        "width_px": np.full(count, 10.0),
        "centerline_xy": np.tile(curve, (count, 1, 1)),
        "width_profile": np.tile(10.0 * template, (count, 1)),
        "width_shape": np.zeros((count, 6)),
        "taper_asymmetry": np.full(count, -0.1),
        "reversed": np.zeros(count, dtype=bool),
        "orientation_gap": np.full(count, 0.01),
        "iou": np.linspace(0.95, 0.85, count),
        "energy": np.full(count, 0.1),
        "total_energy": np.full(count, 0.11),
        "source": np.zeros(count, dtype=np.int8),
        "mask_on_border": np.zeros(count, dtype=bool),
        "points_in_fov": np.full(count, n_points),
        "body_length_px": np.full(count, 100.0),
        "crop": np.tile([0, WIDTH, 0, HEIGHT], (count, 1)),
        "worm_pixels": np.full(count, 900),
        "raw_worm_pixels": np.full(count, 880),
        "pixels_filled": np.full(count, 20),
        "components": np.ones(count, dtype=np.int64),
        "pixels_outside_largest": np.zeros(count, dtype=np.int64),
        "n_starts": np.full(count, 2),
        "width_template": template,
        "area_ratio": np.full(count, 1.0),
        "self_contact_px": np.full(count, 50.0),
        "pose_jump_px": np.full(count, 1.0),
        "length_deviation": np.zeros(count),
        "ambiguity_score": np.zeros(count, dtype=np.int64),
        "iou_independent": np.linspace(0.95, 0.85, count),
        "score_independent": np.zeros(count, dtype=np.int64),
        "best_start": np.array(["skeleton_longest_path"] * count),
    }
    for name in FLAG_NAMES:
        arrays[f"flag_{name}"] = np.zeros(count, dtype=bool)
    # Last frame: a propagated coil-like frame with a low overlap.
    arrays["flag_low_iou"][-1] = True
    arrays["flag_self_contact"][-1] = True
    arrays["ambiguity_score"][-1] = 2
    arrays["source"][-1] = 1
    arrays["best_start"][-1] = "warm_forward"
    if independent:
        # Step 6a chain candidates on the last frame: a forward chain with a prediction.
        arrays["chain_centerline_xy"] = np.full((count, 2, n_points, 2), np.nan)
        arrays["chain_prediction_xy"] = np.full((count, 2, n_points, 2), np.nan)
        arrays["chain_energy"] = np.full((count, 2), np.nan)
        arrays["chain_iou"] = np.full((count, 2), np.nan)
        arrays["chain_start"] = np.full((count, 2), "", dtype="<U32")
        arrays["prediction_distance_px"] = np.full(count, np.nan)
        arrays["chain_centerline_xy"][-1, 0] = curve
        arrays["chain_prediction_xy"][-1, 0] = curve + (1.0, 0.0)
        arrays["chain_energy"][-1, 0] = 0.09
        arrays["chain_iou"][-1, 0] = 0.85
        arrays["chain_start"][-1, 0] = "predicted_forward"
        arrays["prediction_distance_px"][-1] = 1.0
        arrays["centerline_xy_independent"] = arrays["centerline_xy"].copy()
        arrays["centerline_xy_independent"][-1, :, 1] += 8.0
        arrays["width_profile_independent"] = arrays["width_profile"].copy()
        arrays["body_length_independent"] = arrays["body_length_px"].copy()
    np.savez_compressed(path / "poses.npz", **arrays)
    summary = {
        "started_at": f"2026-09-06T1{first}:00:00+00:00",
        "recording": str(recording),
        "frames": [int(frame_index[0]), int(frame_index[-1])],
        "step": 1,
        "frame_count": count,
        "frames_fitted": count,
        "checkpoint": {"path": str(path / "missing.ckpt"), "sha256": "abcdef0123456789"},
        "git": {"commit": "0123456789abcdef", "dirty": False},
        "threshold": 0.5,
        "mask_cleanup": {"fill_holes": True, "fill_holes_radius_px": 8, "largest_component": True, "min_worm_pixels": 500},
        "fit_config": {"coefficients": 16, "n_points": n_points, "width_coefficients": 6, "stage_downsample": [4, 1]},
        "preset": "fast",
        "prior": {"length_px": 100.0, "log_length_sigma": 0.05, "width_px": 10.0, "log_width_sigma": 0.05, "width_shape": [0.1, 0.0, 0.0, 0.0, 0.0, -0.1], "width_shape_sigma": [0.05] * 6, "frames_used": 3, "frames_candidates": 4, "selection": {}},
        "iou": {"median": 0.9, "p10": 0.86, "min": 0.85, "fraction_at_least_0.8": 1.0, "fraction_at_least_0.9": 0.5},
        "body_length_px": {"median": 100.0, "p10": 100.0, "p90": 100.0, "at_upper_bound": None, "beyond_2_sigma_of_prior": 0},
        "ambiguity": {"thresholds": {"low_iou": 0.9, "holes_px": 200}, "flag_counts": {name: 0 for name in FLAG_NAMES}, "frames_with_score_at_least_1": 1, "frames_with_score_at_least_2": 1},
        "propagation": {"stretches": [[count - 3, count - 1]], "frames_in_stretches": 3, "frames_replaced": 1, "replaced_by_source": {"forward": 1, "backward": 0}, "stretch_iou_median_before": 0.5, "stretch_iou_median_after": 0.85},
    }
    (path / "summary.json").write_text(json.dumps(summary))


class PoseViewerHelperTests(unittest.TestCase):
    def test_classification_separates_coils_edges_and_failures(self) -> None:
        clean = classify_frame(True, {}, 0, points_in_fov=100, n_points=100, mask_on_border=False)
        self.assertEqual(clean["kind"], "clean")
        self.assertEqual(clean["tags"], [])
        coil = classify_frame(True, {"self_contact": True, "holes": True}, 2, points_in_fov=100, n_points=100, mask_on_border=False, source=2)
        self.assertEqual(coil["kind"], "ambiguous")
        self.assertIn("coil / self-contact", coil["tags"])
        self.assertIn("propagated backward", coil["tags"])
        self.assertNotIn("fit failure suspected", coil["tags"])
        edge = classify_frame(True, {}, 1, points_in_fov=80, n_points=100, mask_on_border=True)
        self.assertEqual(edge["kind"], "watch")
        self.assertEqual(edge["tags"], ["body at camera edge"])
        failure = classify_frame(True, {"low_iou": True, "pose_jump": True}, 2, points_in_fov=100, n_points=100, mask_on_border=False)
        self.assertEqual(failure["label"], "ambiguous: fit failure suspected")
        self.assertEqual(classify_frame(False, {}, 0, points_in_fov=0, n_points=100, mask_on_border=False)["kind"], "unfitted")

    def test_curvature_of_a_circle_is_its_inverse_radius(self) -> None:
        angle = np.linspace(0, np.pi, 100)
        circle = np.stack((50 * np.cos(angle), 50 * np.sin(angle)), axis=1)
        curvature = signed_curvature(circle)
        np.testing.assert_allclose(curvature[5:-5], 1 / 50, rtol=0.02)
        self.assertTrue(np.all(signed_curvature(np.stack((np.arange(100.0), np.zeros(100)), axis=1)) == 0))

    def test_prior_width_profile_applies_mean_centred_log_correction(self) -> None:
        template = default_width_template(100)
        symmetric = prior_width_profile(10.0, template, None)
        np.testing.assert_allclose(symmetric, 10.0 * template)
        shaped = prior_width_profile(10.0, template, [0.2, 0.0, 0.0, 0.0, 0.0, -0.2])
        self.assertGreater(shaped[10], symmetric[10])
        self.assertLess(shaped[-10], symmetric[-10])
        np.testing.assert_allclose(np.mean(np.log(shaped / (10.0 * template))), 0.0, atol=1e-9)


class PoseViewerServerTests(unittest.TestCase):
    def test_catalog_series_frames_and_notes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recording = root / "rec-a.h5"
            _write_recording(recording)
            _write_run(root / "runs" / "2026-09-06T10-00-00Z_demo", recording)
            _write_run(root / "runs" / "2026-09-06T11-00-00Z_other", recording, first=2, count=3, independent=False)
            (root / "runs" / "not_a_run").mkdir()
            runs = ViewerState.discover(root / "runs")
            self.assertEqual([p.name for p in runs], ["2026-09-06T10-00-00Z_demo", "2026-09-06T11-00-00Z_other"])
            entry = run_entry(runs[0])
            self.assertEqual(entry["mask_cleanup"], "fill + largest")
            self.assertEqual(entry["frames_below_0.9"], 3)
            state = ViewerState(runs, dataset_root=root / "dataset", checkpoint=None, device="cpu", notes=root / "notes.json", runs_root=root / "runs")
            server = create_server(state, "127.0.0.1", 0)
            port = server.server_address[1]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{port}"
            try:
                info = json.loads(urlopen(f"{base}/api/state").read())
                self.assertEqual([r["name"] for r in info["runs"]], ["2026-09-06T11-00-00Z_other", "2026-09-06T10-00-00Z_demo"])
                self.assertEqual(info["flag_groups"]["coil"], ["self_contact", "holes", "area_deficit"])
                _write_run(root / "runs" / "2026-09-06T13-00-00Z_late", recording, first=3, count=2, independent=False)
                rescanned = json.loads(urlopen(f"{base}/api/state?rescan=1").read())
                self.assertEqual(rescanned["added"], 1)
                self.assertEqual(rescanned["runs"][0]["name"], "2026-09-06T13-00-00Z_late")
                page = urlopen(f"{base}/").read().decode()
                self.assertIn("Pose viewer", page)
                self.assertIn("Width along the body", page)
                script = urlopen(f"{base}/app.js").read().decode()
                self.assertIn("buildOverlay", script)
                self.assertIn("drawCharts", script)

                run = json.loads(urlopen(f"{base}/api/run?name=2026-09-06T10-00-00Z_demo").read())
                self.assertTrue(run["recording_readable"])
                self.assertEqual(run["image_shape"], [HEIGHT, WIDTH])
                self.assertTrue(run["has_independent_pose"])
                # Overlapping runs first: "other" covers frames 2-4 (overlap 3), "late" 3-4 (overlap 2).
                self.assertEqual([r["name"] for r in run["compatible_runs"]], ["2026-09-06T11-00-00Z_other", "2026-09-06T13-00-00Z_late"])
                self.assertEqual([r["overlap"] for r in run["compatible_runs"]], [3, 2])
                series = run["series"]
                self.assertEqual(series["frame_index"], list(range(FRAMES)))
                self.assertEqual(series["classification"], ["clean"] * (FRAMES - 1) + ["ambiguous"])
                self.assertEqual(series["flags"]["low_iou"], [0] * (FRAMES - 1) + [1])
                self.assertEqual(len(series["tube_area_px"]), FRAMES)
                self.assertEqual(run["stretches"], [[FRAMES - 3, FRAMES - 1]])

                frame = json.loads(urlopen(f"{base}/api/frame?run=2026-09-06T10-00-00Z_demo&frame={FRAMES - 1}&raw=1").read())
                self.assertEqual((frame["height"], frame["width"]), (HEIGHT, WIDTH))
                self.assertTrue(frame["layers"]["image"].startswith("data:image/jpeg"))
                self.assertTrue(frame["image_raw"].startswith("data:image/jpeg"))
                self.assertTrue(frame["layers"]["tube"].startswith("data:image/png"))
                self.assertTrue(frame["layers"]["tube_independent"].startswith("data:image/png"))
                self.assertNotIn("probability", frame["layers"])  # no checkpoint anywhere
                self.assertTrue(any("checkpoint" in e for e in frame["errors"]))
                stats = frame["stats"]
                self.assertEqual(stats["source_name"], "forward")
                self.assertEqual(stats["classification"]["kind"], "ambiguous")
                self.assertIn("coil / self-contact", stats["classification"]["tags"])
                self.assertEqual(stats["stretch"]["rows"], [FRAMES - 3, FRAMES - 1])
                fired = {f["name"] for f in stats["flags"] if f["fired"]}
                self.assertEqual(fired, {"low_iou", "self_contact"})
                low = next(f for f in stats["flags"] if f["name"] == "low_iou")
                self.assertEqual((low["threshold"], low["test"]), (0.9, "<"))
                self.assertEqual(stats["length_vs_prior_sigmas"], 0.0)
                pose = frame["pose"]
                self.assertEqual(len(pose["centerline_xy"]), 100)
                self.assertEqual(len(pose["curvature"]), 100)
                self.assertEqual(len(pose["width_prior_profile"]), 100)
                self.assertAlmostEqual(pose["independent"]["centerline_xy"][0][1] - pose["centerline_xy"][0][1], 8.0, places=1)
                self.assertEqual(list(pose["chains"]), ["forward"])
                self.assertEqual(pose["chains"]["forward"]["start"], "predicted_forward")
                self.assertEqual(len(pose["chains"]["forward"]["prediction_xy"]), 100)
                self.assertEqual(pose["prediction_distance_px"], 1.0)
                self.assertTrue(run["has_chain_candidates"])
                self.assertEqual(series["prediction_distance_px"][-1], 1.0)

                light = json.loads(urlopen(f"{base}/api/frame?run=2026-09-06T10-00-00Z_demo&frame=2&detail=light").read())
                self.assertEqual(light["detail"], "light")
                self.assertEqual(sorted(light["layers"]), ["image", "tube"])
                self.assertEqual(light["errors"], [])
                self.assertEqual(light["stats"]["frame_index"], 2)
                with self.assertRaises(Exception):
                    urlopen(f"{base}/api/frame?run=2026-09-06T10-00-00Z_demo&frame=2&detail=medium")

                other = json.loads(urlopen(f"{base}/api/pose?run=2026-09-06T11-00-00Z_other&frame=3").read())
                self.assertTrue(other["present"])
                self.assertNotIn("independent", other["pose"])
                missing = json.loads(urlopen(f"{base}/api/pose?run=2026-09-06T11-00-00Z_other&frame=0").read())
                self.assertFalse(missing["present"])

                with self.assertRaises(Exception):
                    urlopen(f"{base}/api/frame?run=2026-09-06T10-00-00Z_demo&frame=99")

                body = json.dumps({"run": "2026-09-06T10-00-00Z_demo", "frame_index": 5, "tags": ["coil"], "comment": "gap closed"}).encode()
                notes = json.loads(urlopen(Request(f"{base}/api/note", data=body, headers={"Content-Type": "application/json"})).read())["notes"]
                self.assertEqual(notes[0]["recording"], "rec-a")
                self.assertEqual(notes[0]["tags"], ["coil"])
                listed = json.loads(urlopen(f"{base}/api/notes").read())
                self.assertEqual(len(listed["notes"]), 1)
                self.assertEqual(json.loads((root / "notes.json").read_text())["notes"][0]["frame_index"], 5)
                body = json.dumps({"index": 0}).encode()
                remaining = json.loads(urlopen(Request(f"{base}/api/note/delete", data=body, headers={"Content-Type": "application/json"})).read())["notes"]
                self.assertEqual(remaining, [])
            finally:
                server.shutdown()
                server.server_close()
                state.close()


if __name__ == "__main__":
    unittest.main()
