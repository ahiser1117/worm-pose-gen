"""Body-field records: flip, review, staleness, atomic writes, and the app's /api/body-fields routes."""

import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from worm_pose_gen import body_fields
from worm_pose_gen.app import AppConfig, create_app
from worm_pose_gen.body_targets import render_body_targets
from worm_pose_gen.corpus import CorpusStore


SHAPE = (64, 96)
MAX_LAG = 2


def straight_worm():
    centerline = np.stack((np.linspace(10, 90, 50), np.full(50, 32.0)), 1)
    yy, xx = np.mgrid[: SHAPE[0], : SHAPE[1]]
    mask = (np.abs(yy - 32) <= 4) & (xx >= 10) & (xx <= 90)
    return centerline, np.linspace(6.0, 9.0, 50), mask


def write_record(root, record, *, fit_iou=0.97, has_body=True):
    """A record as the builder writes it, from a straight worm (head at x=10)."""

    centerline, widths, mask = straight_worm()
    image = np.where(mask, 60, 200).astype(np.uint8)
    context = np.stack([np.roll(image, shift, axis=1) for shift in range(-MAX_LAG, MAX_LAG + 1)])
    meta = {"sample_id": record.sample_id, "mask_revision": record.revision, "max_lag": MAX_LAG,
            "fit_preset": "reference", "has_body": has_body}
    arrays = {"context": context, "context_valid": np.array([False, True, True, True, True])}
    if has_body:
        targets = render_body_targets(mask, centerline, widths)
        meta.update(orientation="nose", nose_offset=0, orientation_margin=3.5, fit_iou=fit_iou, overlap_px=0)
        arrays.update(nose_xy=np.array([8.0, 32.0]), centerline_xy=centerline, width_profile=widths,
                      ap=targets.ap.astype(np.float16), overlap=targets.overlap, head_xy=targets.head_xy,
                      tail_xy=targets.tail_xy, diameter_px=np.float64(targets.diameter_px))
    body_fields.fields_dir(root).mkdir(parents=True, exist_ok=True)
    body_fields.save(body_fields.field_path(root, record.sample_id), meta, arrays)
    return image, mask


def decode(url):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("L"))


class BodyFieldRecordTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.store = CorpusStore(self.root)
        *_, mask = straight_worm()
        self.record = self.store.save("rec", 3, np.full(SHAPE, 100, np.uint8), mask.astype(np.uint8),
                                      source_path="rec.h5", label_source="manual", split="train")
        write_record(self.root, self.record)
        self.path = body_fields.field_path(self.root, self.record.sample_id)

    def test_flip_reverses_the_tube_and_keeps_every_other_key(self):
        before, meta_before = body_fields.load(self.path)
        meta = body_fields.flip(self.root, self.record.sample_id)
        after, stored = body_fields.load(self.path)
        self.assertEqual(stored, meta)
        self.assertEqual(meta["orientation"], "manual")
        self.assertEqual(meta["orientation_margin"], -3.5)
        self.assertEqual(set(after), set(before))
        np.testing.assert_array_equal(after["centerline_xy"], before["centerline_xy"][::-1])
        np.testing.assert_array_equal(after["width_profile"], before["width_profile"][::-1])
        np.testing.assert_array_equal(after["head_xy"], before["tail_xy"])
        np.testing.assert_array_equal(after["tail_xy"], before["head_xy"])
        self.assertEqual(after["ap"].dtype, np.float16)
        np.testing.assert_allclose(after["ap"], 1 - before["ap"], atol=1e-3)
        np.testing.assert_array_equal(np.isnan(after["ap"]), np.isnan(before["ap"]))
        for name in ("context", "context_valid", "overlap", "nose_xy", "diameter_px"):
            np.testing.assert_array_equal(after[name], before[name])
        self.assertEqual({k: v for k, v in meta.items() if k not in ("orientation", "orientation_margin")},
                         {k: v for k, v in meta_before.items() if k not in ("orientation", "orientation_margin")})
        # The tube rendered from the flipped centerline gives the flipped field.
        _, _, mask = straight_worm()
        rendered = render_body_targets(mask, after["centerline_xy"], after["width_profile"]).ap
        np.testing.assert_allclose(after["ap"], rendered, atol=2e-3)

    def test_flip_redraws_an_existing_review_png_and_needs_a_body(self):
        review = body_fields.fields_dir(self.root) / body_fields.REVIEW_DIR
        review.mkdir()
        (review / f"{self.record.sample_id}.png").write_bytes(b"old")
        body_fields.flip(self.root, self.record.sample_id)
        with Image.open(review / f"{self.record.sample_id}.png") as picture:
            self.assertEqual(picture.mode, "RGB")
        empty = self.store.save("rec", 4, np.full(SHAPE, 100, np.uint8), np.zeros(SHAPE, np.uint8),
                                source_path="rec.h5", label_source="manual")
        write_record(self.root, empty, has_body=False)
        with self.assertRaisesRegex(ValueError, "no body"):
            body_fields.flip(self.root, empty.sample_id)

    def test_review_status_and_time(self):
        self.assertEqual(body_fields.review_status(body_fields.read_meta(self.path)), "unreviewed")
        meta = body_fields.set_review(self.root, self.record.sample_id, "rejected")
        self.assertEqual(meta["review"], "rejected")
        self.assertRegex(meta["reviewed_at"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00$")
        self.assertEqual(body_fields.read_meta(self.path), meta)
        with self.assertRaisesRegex(ValueError, "unknown review status"):
            body_fields.set_review(self.root, self.record.sample_id, "maybe")
        with self.assertRaises(FileNotFoundError):
            body_fields.set_review(self.root, "rec_f000099", "accepted")

    def test_mask_edit_makes_the_record_stale(self):
        meta = body_fields.read_meta(self.path)
        self.assertFalse(body_fields.is_stale(meta, self.record))
        self.assertTrue(body_fields.is_current(self.path, self.record.revision, MAX_LAG))
        *_, mask = straight_worm()
        edited = self.store.update_label(self.record.sample_id, np.roll(mask, 1, axis=0).astype(np.uint8))
        self.assertEqual(edited.revision, self.record.revision + 1)
        self.assertTrue(body_fields.is_stale(meta, edited))
        self.assertFalse(body_fields.is_current(self.path, edited.revision, MAX_LAG))
        self.assertFalse(body_fields.is_current(self.path, self.record.revision, MAX_LAG + 1))

    def test_failed_write_leaves_the_record_untouched(self):
        original = self.path.read_bytes()
        with mock.patch.object(np, "savez_compressed", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                body_fields.set_review(self.root, self.record.sample_id, "accepted")
        self.assertEqual(self.path.read_bytes(), original)
        body_fields.set_review(self.root, self.record.sample_id, "accepted")
        self.assertEqual(sorted(p.name for p in self.path.parent.glob("*.npz*")), [self.path.name])

    def test_choose_nose_prefers_the_centre_then_the_nearest(self):
        xy = np.arange(10, dtype=float)[:, None].repeat(2, 1)
        valid = np.zeros(5, bool)
        self.assertEqual(body_fields.choose_nose(xy[:5], valid, 2), (None, None))
        valid[[0, 3]] = True
        nose, offset = body_fields.choose_nose(xy[:5], valid, 2)
        self.assertEqual((offset, nose[0]), (1, 3.0))
        valid[2] = True
        self.assertEqual(body_fields.choose_nose(xy[:5], valid, 2)[1], 0)


LOOP_SHAPE = (300, 420)
# Right along y=150, down, back left, then up through the first run at (240, 150).
LOOP_CORNERS = np.array([[80, 150], [340, 150], [340, 250], [240, 250], [240, 60]], float)


def looped_worm():
    pieces = [np.linspace(a, b, 60, endpoint=False) for a, b in zip(LOOP_CORNERS[:-1], LOOP_CORNERS[1:])]
    centerline = np.concatenate(pieces + [LOOP_CORNERS[-1:]])
    yy, xx = np.mgrid[: LOOP_SHAPE[0], : LOOP_SHAPE[1]]
    points = np.stack((xx.ravel(), yy.ravel()), 1).astype(float)
    distance = np.full(len(points), np.inf)
    for a, b in zip(centerline[:-1], centerline[1:]):
        t = np.clip(((points - a) @ (b - a)) / max(float((b - a) @ (b - a)), 1e-9), 0, 1)
        distance = np.minimum(distance, np.linalg.norm(points - (a + t[:, None] * (b - a)), axis=1))
    return centerline, (distance <= 14).reshape(LOOP_SHAPE)


class TraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.centerline, cls.mask = looped_worm()

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.store = CorpusStore(self.root)
        image = np.where(self.mask, 60, 200).astype(np.uint8)
        self.record = self.store.save("rec", 3, image, self.mask.astype(np.uint8),
                                      source_path="rec.h5", label_source="manual", split="train")
        targets = render_body_targets(self.mask, self.centerline, np.full(len(self.centerline), 28.0))
        meta = {"sample_id": self.record.sample_id, "mask_revision": self.record.revision, "max_lag": 0,
                "fit_preset": "reference", "has_body": True, "orientation": "nose", "fit_iou": 0.5, "overlap_px": 0}
        arrays = {"context": image[None], "context_valid": np.array([True]), "centerline_xy": self.centerline,
                  "width_profile": np.full(len(self.centerline), 28.0), "ap": targets.ap.astype(np.float16),
                  "overlap": targets.overlap, "head_xy": targets.head_xy, "tail_xy": targets.tail_xy,
                  "diameter_px": np.float64(28.0)}
        body_fields.fields_dir(self.root).mkdir(parents=True)
        self.path = body_fields.field_path(self.root, self.record.sample_id)
        body_fields.save(self.path, meta, arrays)

    def test_trace_ending_at_the_border_continues_off_camera(self):
        trace = np.array([[50.0, 30.0], [80.0, 30.0], [415.0, 30.0]])
        extended = body_fields.extend_trace(trace, LOOP_SHAPE, 500.0)
        self.assertEqual(len(extended), 4)
        np.testing.assert_allclose(extended[-1], [550.0, 30.0])
        inside = np.array([[50.0, 30.0], [80.0, 30.0]])
        np.testing.assert_array_equal(body_fields.extend_trace(inside, LOOP_SHAPE, 500.0), inside)
        np.testing.assert_array_equal(body_fields.extend_trace(trace, LOOP_SHAPE, None), trace)

    def test_trace_through_the_crossing_sets_the_ap_field_and_preview_writes_nothing(self):
        # Rough clicks along the loop, a few pixels off the midline.
        trace = np.array([[82, 148], [180, 153], [290, 147], [337, 175], [336, 245],
                          [290, 253], [243, 247], [238, 180], [244, 110], [240, 63]], float)
        stored_before = self.path.read_bytes()
        meta, arrays = body_fields.apply_trace(self.store, self.record.sample_id, trace, device="cpu")
        self.assertEqual(self.path.read_bytes(), stored_before)
        self.assertEqual((meta["fit_method"], meta["review"], meta["orientation"]), ("traced", "accepted", "manual"))
        self.assertEqual(meta["auto_fit_iou"], 0.5)
        # Square corners and flat ends that a smooth tube cannot match cap the overlap.
        self.assertGreater(meta["fit_iou"], 0.75)
        np.testing.assert_array_equal(arrays["trace_xy"], trace)
        self.assertLess(np.linalg.norm(arrays["head_xy"] - LOOP_CORNERS[0]), 15.0)
        # Before the crossing on the first run, and after it on the last.
        self.assertLess(float(arrays["ap"][150, 150]), 0.2)
        self.assertGreater(float(arrays["ap"][90, 240]), 0.85)
        meta, _ = body_fields.apply_trace(self.store, self.record.sample_id, trace, commit=True, device="cpu")
        stored, stored_meta = body_fields.load(self.path)
        self.assertEqual(stored_meta, meta)
        self.assertIn("context", stored)

    def test_as_drawn_keeps_the_trace_as_midline(self):
        trace = LOOP_CORNERS.copy()
        meta, arrays = body_fields.apply_trace(self.store, self.record.sample_id, trace, as_drawn=True, device="cpu")
        self.assertEqual(meta["fit_method"], "trace_as_drawn")
        np.testing.assert_allclose(arrays["centerline_xy"][[0, -1]], trace[[0, -1]], atol=1e-6)


class BodyFieldApiTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config = AppConfig(workspaces_root=self.root / "workspaces", recording_roots=(self.root,),
                                poses_root=self.root / "poses", corpus_root=self.root / "corpus",
                                dataset_root=self.root / "flat-field-cache", checkpoint=None, prior_cache=None,
                                notes=self.root / "notes.json", device="cpu", gpus=())
        self.app = create_app(self.config)
        # No lifespan: rebuild submission is tested while the queue remains stopped.
        self.client = TestClient(self.app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)
        self.addCleanup(self.app.state.app_state.close)
        store = CorpusStore(self.config.corpus_root)
        *_, mask = straight_worm()
        image = np.full(SHAPE, 100, np.uint8)
        self.good = store.save("rec", 1, image, mask.astype(np.uint8), source_path="rec.h5", label_source="manual", split="train")
        self.poor = store.save("rec", 2, image, mask.astype(np.uint8), source_path="rec.h5", label_source="manual", split="val")
        self.missing = store.save("rec", 3, image, mask.astype(np.uint8), source_path="rec.h5", label_source="manual", split="test")
        write_record(self.config.corpus_root, self.good)
        write_record(self.config.corpus_root, self.poor, fit_iou=0.6)
        self.url = f"/api/body-fields/{self.good.sample_id}"

    def listing(self, **query):
        response = self.client.get("/api/body-fields", params=query)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_listing_status_and_filters(self):
        payload = self.listing()
        self.assertEqual(payload["total"], 3)
        self.assertEqual(payload["counts"]["status"], {"current": 2, "stale": 0, "missing": 1})
        rows = {row["sample_id"]: row for row in payload["samples"]}
        self.assertEqual(rows[self.missing.sample_id]["status"], "missing")
        self.assertEqual(rows[self.good.sample_id]["review"], "unreviewed")
        self.assertFalse(rows[self.good.sample_id]["self_contact"])
        self.assertEqual(payload["facets"]["orientations"], ["nose"])
        ids = lambda **query: [row["sample_id"] for row in self.listing(**query)["samples"]]
        self.assertEqual(ids(max_iou="0.9"), [self.poor.sample_id])
        self.assertEqual(ids(min_iou="0.9"), [self.good.sample_id])
        self.assertEqual(ids(split="val"), [self.poor.sample_id])
        self.assertEqual(ids(status="missing"), [self.missing.sample_id])
        self.assertEqual(ids(contact="yes"), [])
        self.assertEqual(ids(orientation="taper"), [])
        self.assertEqual(self.client.get("/api/body-fields", params={"review": "maybe"}).status_code, 400)

    def test_detail_layers_and_context(self):
        detail = self.client.get(self.url).json()
        self.assertEqual((detail["width"], detail["height"]), (SHAPE[1], SHAPE[0]))
        self.assertEqual(detail["meta"]["orientation"], "nose")
        self.assertEqual(detail["nose_xy"], [8.0, 32.0])
        self.assertEqual(detail["head_xy"], [10.0, 32.0])
        _, _, mask = straight_worm()
        self.assertTrue(np.array_equal(decode(detail["mask"]) == 255, mask))
        ap = decode(detail["ap"])
        self.assertTrue(np.array_equal(ap > 0, mask))
        self.assertLess(ap[32, 12], 20)
        self.assertGreater(ap[32, 88], 235)
        context = self.client.get(self.url + "/context").json()
        self.assertEqual(context["valid"], [False, True, True, True, True])
        self.assertEqual(len(context["frames"]), 2 * MAX_LAG + 1)
        self.assertEqual(decode(context["frames"][0]).shape, SHAPE)
        missing = self.client.get(f"/api/body-fields/{self.missing.sample_id}").json()
        self.assertIsNone(missing["meta"])
        self.assertEqual(self.client.get(f"/api/body-fields/{self.missing.sample_id}/context").status_code, 404)
        self.assertEqual(self.client.get("/api/body-fields/nope").status_code, 404)

    def test_flip_review_and_stale_after_a_label_edit(self):
        flipped = self.client.post(self.url + "/flip")
        self.assertEqual(flipped.status_code, 200, flipped.text)
        self.assertEqual(flipped.json()["meta"]["orientation"], "manual")
        self.assertEqual(flipped.json()["head_xy"], [90.0, 32.0])
        reviewed = self.client.post(self.url + "/review", json={"status": "accepted"})
        self.assertEqual(reviewed.status_code, 200, reviewed.text)
        self.assertEqual(reviewed.json()["sample"]["review"], "accepted")
        self.assertEqual(self.listing(review="accepted")["samples"][0]["sample_id"], self.good.sample_id)
        self.assertEqual(self.client.post(self.url + "/review", json={"status": "bad"}).status_code, 400)
        self.assertEqual(self.client.post(f"/api/body-fields/{self.missing.sample_id}/flip").status_code, 404)
        *_, mask = straight_worm()
        CorpusStore(self.config.corpus_root).update_label(self.good.sample_id, np.roll(mask, 1, axis=0).astype(np.uint8))
        row = self.client.get(self.url).json()["sample"]
        self.assertEqual((row["status"], row["fields_revision"], row["mask_revision"]), ("stale", 1, 2))

    def test_rebuild_queues_one_job_and_blocks_edits_until_it_finishes(self):
        response = self.client.post(self.url + "/rebuild")
        self.assertEqual(response.status_code, 200, response.text)
        job = response.json()
        self.assertEqual(job["spec"]["kind"], "body_fields")
        self.assertEqual(job["spec"]["params"]["sample_id"], self.good.sample_id)
        self.assertEqual(job["spec"]["gpus"], 0)
        command = job["command"]
        self.assertTrue(command[1].endswith("scripts/build_body_fields.py"))
        self.assertEqual(command[2:], ["--dataset-root", str(self.config.corpus_root.resolve()),
                                       "--samples", self.good.sample_id, "--force"])
        self.assertEqual(self.client.get(self.url).json()["sample"]["job"], job["id"])
        for path, body in (("/rebuild", {}), ("/flip", None), ("/review", {"status": "accepted"}),
                           ("/trace", {"points": [[10, 32], [90, 32]]})):
            blocked = self.client.post(self.url + path, json=body)
            self.assertEqual(blocked.status_code, 400, path)
            self.assertIn("rebuild job", blocked.json()["error"])
        self.app.state.app_state.runner.cancel(job["id"])
        self.assertIsNone(self.client.get(self.url).json()["sample"]["job"])

    def test_trace_preview_then_commit(self):
        # The stored record runs head at x=10; trace it the other way round.
        points = [[90, 32], [70, 32], [50, 32], [30, 32], [10, 32]]
        stored = body_fields.field_path(self.config.corpus_root, self.good.sample_id).read_bytes()
        preview = self.client.post(self.url + "/trace", json={"points": points})
        self.assertEqual(preview.status_code, 200, preview.text)
        body = preview.json()
        self.assertFalse(body["committed"])
        self.assertEqual((body["meta"]["fit_method"], body["meta"]["auto_fit_iou"]), ("traced", 0.97))
        self.assertGreater(body["meta"]["fit_iou"], 0.0)
        self.assertLess(abs(body["head_xy"][0] - 90), 6)
        self.assertEqual(body["trace_xy"], points)
        self.assertEqual(decode(body["ap"]).shape, SHAPE)
        self.assertEqual(body_fields.field_path(self.config.corpus_root, self.good.sample_id).read_bytes(), stored)
        self.assertIsNone(self.client.get(self.url).json()["trace_xy"])
        drawn = self.client.post(self.url + "/trace", json={"points": points, "as_drawn": True}).json()
        self.assertEqual(drawn["meta"]["fit_method"], "trace_as_drawn")
        committed = self.client.post(self.url + "/trace", json={"points": points, "commit": True}).json()
        self.assertTrue(committed["committed"])
        self.assertEqual((committed["sample"]["fit_method"], committed["sample"]["review"]), ("traced", "accepted"))
        detail = self.client.get(self.url).json()
        self.assertEqual((detail["trace_xy"], detail["meta"]["orientation"]), (points, "manual"))
        listing = self.listing(method="traced")
        self.assertEqual([row["sample_id"] for row in listing["samples"]], [self.good.sample_id])
        self.assertIn("traced", listing["facets"]["methods"])
        self.assertEqual(self.client.post(self.url + "/trace", json={"points": [[10, 32]]}).status_code, 400)
        self.assertEqual(self.client.post(self.url + "/trace", json={"points": "no"}).status_code, 400)
        self.assertEqual(self.client.post(f"/api/body-fields/{self.missing.sample_id}/trace", json={"points": points}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
