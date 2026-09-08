from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from worm_pose_gen.ambiguity import FLAG_NAMES
from worm_pose_gen.latent import decode_centerline
from worm_pose_gen.mask_fit import default_width_template
from worm_pose_gen.workspace import (
    MASK_CHUNK_ROWS,
    Workspace,
    WorkspaceInfo,
    list_workspaces,
    split_arrays,
)


HEIGHT, WIDTH, FRAMES = 96, 128, 6


def _write_recording(path: Path, frames: int = FRAMES, height: int = HEIGHT, width: int = WIDTH) -> None:
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:height, :width]
    stack = np.empty((frames, height, width), dtype=np.uint8)
    for index in range(frames):
        body = np.abs(yy - (height // 2 + 10 * np.sin(xx / 20 + index))) < 6
        image = 190.0 - 80 * body + rng.normal(0, 3, (height, width))
        stack[index] = np.clip(image, 0, 255).astype(np.uint8)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("/img_nir", data=stack)


def _write_run(path: Path, recording: Path, *, first: int = 0, count: int = FRAMES, independent: bool = True, prior: bool = True) -> None:
    """A run directory with the arrays the fitter stores, frames ``first`` onward (after tests/test_pose_viewer.py)."""

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
        "iou": np.linspace(0.95, 0.85, count),
        "energy": np.full(count, 0.1),
        "source": np.zeros(count, dtype=np.int8),
        "mask_on_border": np.zeros(count, dtype=bool),
        "points_in_fov": np.full(count, n_points),
        "body_length_px": np.full(count, 100.0),
        "crop": np.tile([0, WIDTH, 0, HEIGHT], (count, 1)),
        "worm_pixels": np.full(count, 900),
        "ambiguity_score": np.zeros(count, dtype=np.int64),
        "best_start": np.array(["skeleton_longest_path"] * count),
        "width_template": template,
    }
    for name in FLAG_NAMES:
        arrays[f"flag_{name}"] = np.zeros(count, dtype=bool)
    arrays["fitted"][0] = False
    arrays["source"][-1] = 1
    arrays["source"][-2] = 2
    if independent:
        H = 3
        arrays["hypotheses_centerline_xy"] = np.full((count, H, n_points, 2), np.nan)
        arrays["hypotheses_energy"] = np.full((count, H), np.nan)
        arrays["hypotheses_source"] = np.full((count, H), "", dtype="<U12")
        arrays["hypotheses_count"] = np.zeros(count, dtype=np.int64)
        arrays["path_index"] = np.full(count, -1)
        arrays["path_override"] = np.zeros(count, dtype=bool)
        arrays["prediction_xy"] = np.full((count, n_points, 2), np.nan)
        arrays["hypotheses_source"][-1, :2] = ["independent", "forward"]
        arrays["hypotheses_count"][-1] = 2
        arrays["centerline_xy_independent"] = arrays["centerline_xy"].copy()
    np.savez_compressed(path / "poses.npz", **arrays)
    summary = {
        "started_at": "2026-09-06T10:00:00+00:00",
        "recording": str(recording),
        "frames": [int(frame_index[0]), int(frame_index[-1])],
        "step": 1,
        "frame_count": count,
        "threshold": 0.5,
        "mask_cleanup": {"fill_holes": True, "fill_holes_radius_px": 8, "largest_component": True, "min_worm_pixels": 500},
        "preset": "fast",
        "prior": {"length_px": 100.0, "width_px": 10.0},
    }
    (path / "summary.json").write_text(json.dumps(summary))
    if prior:
        (path / "recording_prior.json").write_text(json.dumps({"length_px": 100.0, "width_px": 10.0}))


def _tree_digest(path: Path) -> dict[str, str]:
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(path.rglob("*")) if p.is_file()}


def _blob(height: int, width: int, seed: int) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    return (xx + yy + seed) % 5 == 0


class WorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.recording = self.root / "rec-a.h5"
        _write_recording(self.recording)
        self.workspaces = self.root / "workspaces"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_create_and_open(self) -> None:
        ws = Workspace.create(self.workspaces, "demo", self.recording, 1, 5, step=2, settings={"preset": "fast"})
        self.assertEqual(ws.frame_index.tolist(), [1, 3, 5])
        self.assertEqual(ws.n, 3)
        self.assertEqual(ws.info.frame_count, 3)
        self.assertEqual(ws.image_shape, (HEIGHT, WIDTH))
        self.assertEqual(ws.row_of(3), 1)
        with self.assertRaisesRegex(ValueError, "not in workspace"):
            ws.row_of(2)
        stored = json.loads((self.workspaces / "demo" / "workspace.json").read_text())
        self.assertEqual(stored["image_shape"], [HEIGHT, WIDTH])
        self.assertEqual(stored["settings"], {"preset": "fast"})
        self.assertTrue(stored["created_at"].endswith("+00:00"))

        reopened = Workspace.open(self.workspaces / "demo")
        self.assertEqual(reopened.info, ws.info)
        self.assertEqual(reopened.frame_index.tolist(), [1, 3, 5])
        self.assertEqual(reopened.load_state(), {})
        self.assertEqual(reopened.load_hypotheses(), {})
        self.assertEqual(reopened.edits(), [])
        self.assertFalse(reopened.has_masks())

        with self.assertRaises(FileExistsError):
            Workspace.create(self.workspaces, "demo", self.recording, 0, 1)
        with self.assertRaisesRegex(ValueError, "beyond"):
            Workspace.create(self.workspaces, "too-long", self.recording, 0, FRAMES)
        with self.assertRaisesRegex(ValueError, "invalid workspace name"):
            Workspace.create(self.workspaces, "a/b", self.recording, 0, 1)
        with self.assertRaises(FileNotFoundError):
            Workspace.open(self.workspaces / "missing")
        # An unreadable recording still gives a workspace, with no image shape.
        blind = Workspace.create(self.workspaces, "blind", self.root / "nope.h5", 0, 3)
        self.assertIsNone(blind.image_shape)

    def test_import_run_splits_arrays_and_sets_provenance(self) -> None:
        run = self.root / "runs" / "2026-09-06T10-00-00Z_demo"
        _write_run(run, self.recording)
        before = _tree_digest(run)
        ws = Workspace.import_run(self.workspaces, run)
        self.assertEqual(_tree_digest(run), before)
        self.assertEqual(ws.info.name, run.name)
        self.assertEqual(ws.info.recording, str(self.recording))
        self.assertEqual(ws.info.frames, [0, FRAMES - 1])
        self.assertEqual(ws.info.imported_runs, [run.name])
        self.assertEqual(ws.info.settings["preset"], "fast")
        self.assertTrue((ws.path / "imported_summary.json").exists())
        self.assertTrue((ws.path / "recording_prior.json").exists())

        state, hypotheses = ws.load_state(), ws.load_hypotheses()
        self.assertNotIn("hypotheses_energy", state)
        self.assertNotIn("path_index", state)
        self.assertIn("centerline_xy_independent", state)
        self.assertIn("width_template", state)
        self.assertEqual(set(hypotheses), {"hypotheses_centerline_xy", "hypotheses_energy", "hypotheses_source", "hypotheses_count", "path_index", "path_override", "prediction_xy"})
        self.assertEqual(hypotheses["hypotheses_source"][-1].tolist(), ["independent", "forward", ""])
        with np.load(run / "poses.npz") as archive:
            original = {k: archive[k] for k in archive.files}
        merged = ws.load_arrays()
        self.assertEqual(set(merged), set(original))
        for key, value in original.items():
            np.testing.assert_array_equal(merged[key], value)
        restate, rehyp = split_arrays(merged)
        self.assertEqual(set(restate), set(state))
        self.assertEqual(set(rehyp), set(hypotheses))

        provenance = ws.load_provenance()
        self.assertEqual(provenance["algorithm"].tolist(), ["", "independent_fit", "independent_fit", "independent_fit", "chain_backward", "chain_forward"])
        self.assertEqual(provenance["job"][1], f"import:{run.name}")
        self.assertEqual(provenance["job"][0], "")
        self.assertTrue(np.isnan(provenance["time"][0]))
        self.assertEqual(provenance["time"][1], datetime(2026, 9, 6, 10, tzinfo=timezone.utc).timestamp())
        self.assertEqual(ws.provenance_counts(), {"chain_backward": 1, "chain_forward": 1, "independent_fit": 3})

        named = Workspace.import_run(self.workspaces, run, name="baseline")
        self.assertEqual(named.info.name, "baseline")
        self.assertEqual(named.path, self.workspaces / "baseline")

    def test_import_older_run_without_hypotheses(self) -> None:
        run = self.root / "runs" / "2026-09-06T11-00-00Z_old"
        _write_run(run, self.recording, first=2, count=3, independent=False, prior=False)
        ws = Workspace.import_run(self.workspaces, run)
        self.assertEqual(ws.frame_index.tolist(), [2, 3, 4])
        self.assertFalse((ws.path / "hypotheses.npz").exists())
        self.assertFalse((ws.path / "recording_prior.json").exists())
        self.assertEqual(ws.load_hypotheses(), {})
        self.assertIn("centerline_xy", ws.load_state())
        summary = ws.summary()
        self.assertFalse(summary["has_hypotheses"])
        self.assertFalse(summary["has_prior"])
        self.assertEqual(summary["provenance"], {"chain_backward": 1, "chain_forward": 1})

    def test_state_and_provenance_round_trip(self) -> None:
        ws = Workspace.create(self.workspaces, "demo", self.recording, 0, 5)
        arrays = {"fitted": np.array([1, 0, 1, 1, 0, 1], dtype=bool), "iou": np.array([0.95, np.nan, 0.8, 0.92, np.nan, 0.7]), "label": np.array(["a"] * 6)}
        ws.save_state(arrays)
        loaded = ws.load_state()
        self.assertEqual(set(loaded), set(arrays))
        for key in arrays:
            np.testing.assert_array_equal(loaded[key], arrays[key])
        self.assertEqual([p.name for p in ws.path.iterdir() if p.suffix == ".tmp" or ".tmp" in p.name], [])
        ws.save_hypotheses({"path_index": np.full(6, -1)})
        self.assertEqual(ws.load_hypotheses()["path_index"].tolist(), [-1] * 6)

        empty = ws.load_provenance()
        self.assertEqual(empty["algorithm"].dtype, np.dtype("<U32"))
        self.assertEqual(empty["job"].dtype, np.dtype("<U64"))
        self.assertTrue(np.all(np.isnan(empty["time"])))
        ws.set_provenance([0, 2], "independent_fit", "j00000001")
        ws.set_provenance(np.array([3]), "chain_forward", "j00000002", time=123.0)
        provenance = Workspace.open(ws.path).load_provenance()
        self.assertEqual(provenance["algorithm"].tolist(), ["independent_fit", "", "independent_fit", "chain_forward", "", ""])
        self.assertEqual(provenance["job"][3], "j00000002")
        self.assertEqual(provenance["time"][3], 123.0)
        self.assertGreater(provenance["time"][0], 1.7e9)
        with self.assertRaisesRegex(ValueError, "out of range"):
            ws.set_provenance([6], "x", "y")

        summary = ws.summary()
        self.assertEqual(summary["frame_count"], 6)
        self.assertEqual(summary["fitted"], 4)
        self.assertAlmostEqual(summary["iou"]["median"], 0.86)
        self.assertEqual(summary["iou"]["min"], 0.7)
        self.assertEqual(summary["iou"]["frames_below_0.9"], 2)
        self.assertTrue(summary["has_hypotheses"])
        self.assertEqual(summary["provenance"], {"chain_forward": 1, "independent_fit": 2})

    def test_masks_chunk_across_the_boundary(self) -> None:
        long_recording = self.root / "long.h5"
        count = MASK_CHUNK_ROWS + 80
        _write_recording(long_recording, frames=count, height=8, width=12)
        ws = Workspace.create(self.workspaces, "long", long_recording, 0, count - 1)
        rows = [5, 1023, 1024, 1030]
        masks = [_blob(8, 12, r) for r in rows]
        ws.set_masks(rows, masks)
        self.assertEqual(sorted(p.name for p in ws.masks_dir.iterdir()), ["chunk_00000.npz", "chunk_00001.npz"])
        self.assertTrue(ws.has_masks())
        self.assertEqual(ws.mask_rows().tolist(), rows)
        for row, mask in zip(rows, masks):
            np.testing.assert_array_equal(ws.get_mask(row), mask)
        self.assertIsNone(ws.get_mask(6))
        self.assertIsNone(ws.get_mask(2000))
        got = ws.get_masks([1023, 1024, 7])
        self.assertEqual(sorted(got), [1023, 1024])
        with np.load(ws.masks_dir / "chunk_00001.npz") as archive:
            self.assertEqual(archive["rows"].tolist(), [1024, 1030])
            self.assertEqual(archive["packed"].shape, (2, 12))
            self.assertEqual(archive["shape"].tolist(), [8, 12])

        # Appending to chunk 0 rewrites only chunk 0, keeps its rows, and replaces an existing row.
        chunk1_before = (ws.masks_dir / "chunk_00001.npz").read_bytes()
        replacement = ~masks[0]
        ws.set_masks([7, 5], [_blob(8, 12, 7), replacement])
        self.assertEqual((ws.masks_dir / "chunk_00001.npz").read_bytes(), chunk1_before)
        self.assertEqual(ws.mask_rows().tolist(), [5, 7, 1023, 1024, 1030])
        np.testing.assert_array_equal(ws.get_mask(5), replacement)
        np.testing.assert_array_equal(ws.get_mask(1023), masks[1])
        # A fresh handle (empty cache) reads the same.
        np.testing.assert_array_equal(Workspace.open(ws.path).get_mask(7), _blob(8, 12, 7))
        self.assertEqual([p.name for p in ws.masks_dir.iterdir() if "tmp" in p.name], [])

        with self.assertRaisesRegex(ValueError, "shape"):
            ws.set_masks([8], [np.zeros((9, 12), dtype=bool)])
        with self.assertRaisesRegex(ValueError, "out of range"):
            ws.set_masks([count], [masks[0]])
        with self.assertRaisesRegex(ValueError, "length"):
            ws.set_masks([1, 2], [masks[0]])

    def test_override_mask_takes_precedence(self) -> None:
        ws = Workspace.create(self.workspaces, "demo", self.recording, 0, 5)
        stored = _blob(HEIGHT, WIDTH, 1)
        ws.set_masks([2], [stored])
        self.assertIsNone(ws.get_override_mask(2))
        np.testing.assert_array_equal(ws.effective_mask(2), stored)
        override = _blob(HEIGHT, WIDTH, 3)
        ws.set_override_mask(2, override)
        self.assertTrue((ws.path / "overrides" / "masks" / "0000002.npz").exists())
        np.testing.assert_array_equal(ws.get_override_mask(2), override)
        np.testing.assert_array_equal(ws.effective_mask(2), override)
        np.testing.assert_array_equal(ws.get_mask(2), stored)
        # An override on a row without a stored mask counts as a mask row.
        ws.set_override_mask(4, override)
        self.assertEqual(ws.mask_rows().tolist(), [2, 4])
        self.assertEqual(ws.stored_mask_rows().tolist(), [2])
        np.testing.assert_array_equal(ws.effective_mask(4), override)
        self.assertIsNone(ws.effective_mask(3))
        self.assertTrue(ws.clear_override_mask(2))
        self.assertFalse(ws.clear_override_mask(2))
        np.testing.assert_array_equal(ws.effective_mask(2), stored)
        self.assertEqual(ws.summary()["override_rows"], 1)

    def test_leftover_temporaries_do_not_break_the_listings(self) -> None:
        ws = Workspace.create(self.workspaces, "demo", self.recording, 0, 5)
        ws.set_masks([1], [_blob(HEIGHT, WIDTH, 1)])
        ws.set_override_mask(3, _blob(HEIGHT, WIDTH, 3))
        overrides = ws.path / "overrides" / "masks"
        # No temporary survives a completed write, and its name is not a row's.
        self.assertEqual(sorted(p.name for p in overrides.iterdir()), ["0000003.npz"])
        self.assertEqual(sorted(p.name for p in ws.masks_dir.iterdir()), ["chunk_00000.npz"])
        # Temporaries left by a crash mid-write (the current and the old naming) are ignored.
        (overrides / ".0000004.npz.tmp").write_bytes(b"partial")
        (overrides / "0000004.npz.tmp.npz").write_bytes(b"partial")
        (ws.masks_dir / ".chunk_00001.npz.tmp").write_bytes(b"partial")
        self.assertEqual(ws.override_rows(), [3])
        self.assertEqual(ws.mask_rows().tolist(), [1, 3])
        self.assertTrue(ws.has_masks())
        self.assertEqual(ws.summary()["override_rows"], 1)

    def test_edits_log_is_append_only_with_a_counter(self) -> None:
        ws = Workspace.create(self.workspaces, "demo", self.recording, 0, 5)
        first = ws.append_edit("pick_hypothesis", {"row": 3, "index": 1})
        second = ws.append_edit("flip", {"rows": [1, 2]})
        self.assertEqual((first, second), ("e000001", "e000002"))
        third = Workspace.open(ws.path).append_edit("note", {"text": "coil"})
        self.assertEqual(third, "e000003")
        edits = ws.edits()
        self.assertEqual([e["id"] for e in edits], ["e000001", "e000002", "e000003"])
        self.assertEqual(edits[0]["kind"], "pick_hypothesis")
        self.assertEqual(edits[0]["payload"], {"row": 3, "index": 1})
        self.assertTrue(edits[0]["time"].endswith("+00:00"))
        self.assertEqual(len((ws.path / "edits.jsonl").read_text().splitlines()), 3)
        self.assertEqual(ws.summary()["edits"], 3)

    def test_snapshot_copies_state_hypotheses_and_provenance(self) -> None:
        ws = Workspace.create(self.workspaces, "demo", self.recording, 0, 5)
        ws.save_state({"fitted": np.ones(6, dtype=bool)})
        ws.set_provenance([0], "independent_fit", "j1")
        path = ws.snapshot("before fix / v1")
        self.assertEqual(path.parent, ws.path / "snapshots")
        self.assertTrue(path.name.endswith("_before-fix-v1"))
        self.assertEqual(sorted(p.name for p in path.iterdir()), ["provenance.npz", "snapshot.json", "state.npz"])
        self.assertEqual(json.loads((path / "snapshot.json").read_text())["label"], "before fix / v1")
        with np.load(path / "state.npz") as archive:
            self.assertEqual(archive["fitted"].tolist(), [True] * 6)
        again = ws.snapshot("before fix / v1")
        self.assertNotEqual(again, path)
        self.assertEqual(ws.summary()["snapshots"], sorted([path.name, again.name]))
        # The snapshot is a copy: later state changes do not touch it.
        ws.save_state({"fitted": np.zeros(6, dtype=bool)})
        with np.load(path / "state.npz") as archive:
            self.assertEqual(archive["fitted"].tolist(), [True] * 6)

    def test_list_workspaces_newest_first(self) -> None:
        self.assertEqual(list_workspaces(self.workspaces), [])
        stamps = {"a": "2026-09-01T00:00:00+00:00", "b": "2026-09-03T00:00:00+00:00", "c": "2026-09-02T00:00:00+00:00"}
        for name, stamp in stamps.items():
            ws = Workspace.create(self.workspaces, name, self.recording, 0, 2)
            ws.info.created_at = stamp
            ws.save_info()
        (self.workspaces / "not_a_workspace").mkdir()
        (self.workspaces / "broken").mkdir()
        (self.workspaces / "broken" / "workspace.json").write_text("{")
        infos = list_workspaces(self.workspaces)
        self.assertEqual([i.name for i in infos], ["b", "c", "a"])
        self.assertTrue(all(isinstance(i, WorkspaceInfo) for i in infos))
        self.assertEqual(infos[0].path, str(self.workspaces / "b"))
        self.assertEqual(infos[0].frame_count, 3)
        self.assertEqual(infos[0].image_shape, [HEIGHT, WIDTH])


if __name__ == "__main__":
    unittest.main()
