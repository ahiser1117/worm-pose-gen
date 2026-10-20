from __future__ import annotations

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


def _fit_arrays(*, first: int = 0, count: int = FRAMES, independent: bool = True) -> dict[str, np.ndarray]:
    """The per-frame arrays the fitter stores (``poses.npz`` layout), frames ``first`` onward (after tests/test_frame_view.py)."""

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
    return arrays


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

    def test_split_arrays_separates_hypotheses_from_the_state(self) -> None:
        arrays = _fit_arrays()
        state, hypotheses = split_arrays(arrays)
        self.assertNotIn("hypotheses_energy", state)
        self.assertNotIn("path_index", state)
        self.assertIn("centerline_xy_independent", state)
        self.assertIn("width_template", state)
        self.assertEqual(set(hypotheses), {"hypotheses_centerline_xy", "hypotheses_energy", "hypotheses_source", "hypotheses_count", "path_index", "path_override", "prediction_xy"})
        ws = Workspace.create(self.workspaces, "demo", self.recording, 0, FRAMES - 1)
        ws.save_state(state)
        ws.save_hypotheses(hypotheses)
        merged = ws.load_arrays()
        self.assertEqual(set(merged), set(arrays))
        for key, value in arrays.items():
            np.testing.assert_array_equal(merged[key], value)
        self.assertEqual(split_arrays(_fit_arrays(independent=False))[1], {})

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
