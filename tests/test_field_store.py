"""The body-field network's fields kept in a workspace (``field_store``, ``pipeline.WorkspaceFields``)."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from tests.test_body_proposal import SHAPE, looped_body, prediction_from
from worm_pose_gen import pipeline
from worm_pose_gen.body_proposal import FieldPrediction, crop_fields, field_evidence, propose_trace
from worm_pose_gen.field_store import FIELD_CHUNK_ROWS, FieldStore


class FieldCropTests(unittest.TestCase):
    def setUp(self) -> None:
        self.centerline, self.mask = looped_body()
        self.prediction = prediction_from(self.centerline, self.mask, noise=0.03)

    def test_trace_and_evidence_match_the_full_prediction(self) -> None:
        crop = crop_fields(self.prediction, self.mask)
        self.assertTrue(crop.covers(self.mask))
        full = self.prediction
        np.testing.assert_array_equal(propose_trace(crop, self.mask), propose_trace(full, self.mask))
        ours, theirs = field_evidence(crop, self.mask), field_evidence(full, self.mask)
        np.testing.assert_array_equal(ours.ap, theirs.ap)
        np.testing.assert_array_equal(ours.head_xy, theirs.head_xy)
        np.testing.assert_array_equal(ours.tail_xy, theirs.tail_xy)

    def test_covers_masks_within_the_margin_only(self) -> None:
        crop = crop_fields(self.prediction, self.mask)
        shifted = np.roll(self.mask, 1, axis=1)
        self.assertTrue(crop.covers(shifted))
        grown = self.mask.copy()
        grown[:20, :20] = True
        self.assertFalse(crop.covers(grown))

    def test_an_empty_mask_keeps_the_peaks(self) -> None:
        crop = crop_fields(self.prediction, np.zeros(SHAPE, bool))
        self.assertEqual(crop.support.size, 0)
        self.assertEqual(crop.head, self.prediction.peak("head"))
        self.assertIsNone(field_evidence(crop, np.zeros(SHAPE, bool)).head_xy)


class FieldStoreTests(unittest.TestCase):
    def test_round_trip_flush_and_another_key(self) -> None:
        centerline, mask = looped_body()
        crop = crop_fields(prediction_from(centerline, mask, noise=0.03), mask)
        empty = crop_fields(prediction_from(centerline, mask), np.zeros(SHAPE, bool))
        with tempfile.TemporaryDirectory() as directory:
            store = FieldStore(directory, "a")
            store.put({3: crop, FIELD_CHUNK_ROWS + 1: empty})
            # Waiting rows are served before they are written.
            self.assertIs(store.get([3])[3], crop)
            store.flush()
            loaded = FieldStore(directory, "a").get([3, 4, FIELD_CHUNK_ROWS + 1])
            self.assertEqual(sorted(loaded), [3, FIELD_CHUNK_ROWS + 1])
            for name in ("support", "ap", "crossing"):
                np.testing.assert_array_equal(getattr(loaded[3], name), getattr(crop, name))
            self.assertEqual((loaded[3].box, loaded[3].head, loaded[3].tail), (crop.box, crop.head, crop.tail))
            self.assertEqual(loaded[FIELD_CHUNK_ROWS + 1].support.size, 0)
            # A later write merges into the chunk.
            store.put({5: crop})
            store.flush()
            self.assertEqual(sorted(FieldStore(directory, "a").get(range(10))), [3, 5])
            # Fields of another network read as missing, and its write replaces them.
            other = FieldStore(directory, "b")
            self.assertEqual(other.get([3, 5]), {})
            other.put({7: crop})
            other.flush()
            self.assertEqual(sorted(FieldStore(directory, "a").get(range(10))), [])
            self.assertEqual(sorted(FieldStore(directory, "b").get(range(10))), [7])


class _Predictor:
    def __init__(self, prediction: FieldPrediction) -> None:
        self.prediction = prediction
        self.requested: list[list[int]] = []
        self.frames = mock.Mock()

    def predict(self, frame_indices):
        self.requested.append([int(f) for f in frame_indices])
        return [self.prediction for _ in frame_indices]


class WorkspaceFieldsTests(unittest.TestCase):
    def test_predicts_once_and_again_for_a_mask_past_the_stored_region(self) -> None:
        centerline, mask = looped_body()
        predictor = _Predictor(prediction_from(centerline, mask))
        with tempfile.TemporaryDirectory() as directory:
            workspace = mock.Mock(path=Path(directory), frame_index=np.arange(100, 110), info=mock.Mock(settings={}))
            params = pipeline.SegmentParams(flat_field=False)
            with mock.patch.object(pipeline, "field_predictor", return_value=predictor) as load, mock.patch.object(pipeline, "workspace_frames"):
                fields = pipeline.WorkspaceFields(workspace, "stub.ckpt", params, "cpu")
                first = fields([2, 3], {2: mask, 3: mask})
                fields.close()
                self.assertEqual(predictor.requested, [[102, 103]])
                # A new reader serves both rows from the store without loading the network.
                again = pipeline.WorkspaceFields(workspace, "stub.ckpt", params, "cpu")
                second = again([2, 3], {2: mask, 3: mask})
                self.assertEqual((again.stored, again.predicted), (2, 0))
                self.assertEqual(load.call_count, 1)
                np.testing.assert_array_equal(second[0].ap, first[0].ap)
                # A mask edited past the stored region is predicted again, and its fields replace the stored ones.
                edited = mask.copy()
                edited[:20, :20] = True
                third = again([2, 3], {2: mask, 3: edited})
                self.assertEqual(predictor.requested[-1], [103])
                self.assertTrue(third[1].covers(edited))
                again.close()
                self.assertTrue(FieldStore(directory, fields.store.key).get([3])[3].covers(edited))


if __name__ == "__main__":
    unittest.main()
