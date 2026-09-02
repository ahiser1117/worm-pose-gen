from __future__ import annotations

import base64
import io
import json
import tempfile
import threading
import unittest
from urllib.request import Request, urlopen

import h5py
import numpy as np
from PIL import Image

from worm_pose_gen.label_app import (
    LabelState,
    Proposer,
    RecordingSource,
    create_server,
    data_url,
    decode_mask_data_url,
    mask_to_png_values,
    png_values_to_mask,
)
from worm_pose_gen.segmentation_dataset import SegmentationStore
from worm_pose_gen.segmenter import IGNORE_LABEL


def _write_recording(path: str, frames: int = 6, height: int = 96, width: int = 128) -> None:
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:height, :width]
    stack = np.empty((frames, height, width), dtype=np.uint8)
    for index in range(frames):
        background = np.full((height, width), 190.0)
        body = np.abs(yy - (48 + 10 * np.sin(xx / 20 + index))) < 6
        body &= (xx > 12) & (xx < 116)
        image = background - 80 * body + rng.normal(0, 3, (height, width))
        stack[index] = np.clip(image, 0, 255).astype(np.uint8)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("/img_nir", data=stack)


class LabelAppTests(unittest.TestCase):
    def test_png_label_conventions_round_trip(self) -> None:
        mask = np.array([[0, 1, IGNORE_LABEL]], dtype=np.uint8)
        png = mask_to_png_values(mask)
        self.assertEqual(png.tolist(), [[0, 255, 128]])
        self.assertTrue(np.array_equal(png_values_to_mask(png), mask))
        decoded = decode_mask_data_url(data_url(png), (1, 3))
        self.assertTrue(np.array_equal(decoded, mask))
        with self.assertRaisesRegex(ValueError, "shape"):
            decode_mask_data_url(data_url(png), (2, 3))

    def test_refinements_preserve_ignore_and_change_worm(self) -> None:
        mask = np.zeros((40, 60), dtype=np.uint8)
        mask[10:30, 10:50] = 1
        mask[18:22, 28:32] = 0  # small hole
        mask[2:4, 2:4] = 1  # debris
        mask[35, 35] = IGNORE_LABEL
        filled, info = Proposer.refine(mask, "fill_holes", "cpu")
        self.assertTrue(filled[18:22, 28:32].all())
        self.assertEqual(filled[35, 35], IGNORE_LABEL)
        self.assertEqual(info["pixels_added"], 16)
        largest, info = Proposer.refine(mask, "largest_component", "cpu")
        self.assertFalse(largest[2:4, 2:4].any())
        self.assertEqual(info["components_removed"], 1)
        grown, _ = Proposer.refine(mask, "dilate", "cpu")
        self.assertGreater(int((grown == 1).sum()), int((mask == 1).sum()))
        shrunk, _ = Proposer.refine(mask, "erode", "cpu")
        self.assertLess(int((shrunk == 1).sum()), int((mask == 1).sum()))
        with self.assertRaisesRegex(ValueError, "unknown refinement"):
            Proposer.refine(mask, "nope", "cpu")

    def test_server_serves_frames_and_saves_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recording = f"{directory}/rec-a.h5"
            _write_recording(recording)
            store = SegmentationStore(f"{directory}/dataset")
            state = LabelState([recording], store, Proposer(None, "cpu"))
            server = create_server(state, "127.0.0.1", 0)
            port = server.server_address[1]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                info = json.loads(urlopen(f"http://127.0.0.1:{port}/api/state").read())
                self.assertEqual(info["recordings"][0]["name"], "rec-a")
                self.assertIsNone(info["checkpoint"])
                page = urlopen(f"http://127.0.0.1:{port}/").read().decode()
                self.assertIn("Worm labeler", page)
                frame = json.loads(urlopen(f"http://127.0.0.1:{port}/api/frame?recording=rec-a&index=2").read())
                self.assertEqual(frame["height"], 96)
                self.assertIsNone(frame["proposals"]["network"])
                self.assertTrue(frame["proposals"]["classical"].startswith("data:image/png"))
                self.assertIsNone(frame["existing_mask"])
                # Save the classical proposal back as the label.
                body = json.dumps(
                    {"recording": "rec-a", "frame_index": 2, "mask": frame["proposals"]["classical"], "label_source": "manual"}
                ).encode()
                request = Request(f"http://127.0.0.1:{port}/api/save", data=body, headers={"Content-Type": "application/json"})
                saved = json.loads(urlopen(request).read())
                self.assertEqual(saved["record"]["split"], "train")
                self.assertEqual(saved["counts"]["train"], 1)
                again = json.loads(urlopen(f"http://127.0.0.1:{port}/api/frame?recording=rec-a&index=2").read())
                self.assertIsNotNone(again["existing_mask"])
                # Next-frame modes.
                request = Request(
                    f"http://127.0.0.1:{port}/api/next",
                    data=json.dumps({"mode": "sequential", "recording": "rec-a", "current_index": 2, "stride": 3}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                self.assertEqual(json.loads(urlopen(request).read())["frame_index"], 5)
                request = Request(
                    f"http://127.0.0.1:{port}/api/next", data=json.dumps({"mode": "random"}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                chosen = json.loads(urlopen(request).read())
                self.assertNotEqual(chosen["frame_index"], 2)
                # An all-background label is accepted (empty frames are valid negatives).
                empty = np.zeros((96, 128), dtype=np.uint8)
                request = Request(
                    f"http://127.0.0.1:{port}/api/save",
                    data=json.dumps({"recording": "rec-a", "frame_index": 3, "mask": data_url(empty)}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                saved_empty = json.loads(urlopen(request).read())
                self.assertEqual(saved_empty["record"]["foreground_fraction"], 0.0)
                # A malformed mask is refused with a JSON error.
                request = Request(
                    f"http://127.0.0.1:{port}/api/save",
                    data=json.dumps({"recording": "rec-a", "frame_index": 3, "mask": "not-a-png"}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with self.assertRaises(Exception):
                    urlopen(request)
                # Flat field was cached on disk.
                self.assertTrue((store.root / "flat_fields" / "rec-a.npz").exists())
            finally:
                server.shutdown()
                server.server_close()
                state.close()

    def test_recording_source_reads_and_corrects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recording = f"{directory}/rec-b.h5"
            _write_recording(recording, frames=4)
            source = RecordingSource(recording, f"{directory}/ff")
            raw, corrected = source.corrected(1)
            self.assertEqual(raw.shape, (96, 128))
            self.assertEqual(corrected.dtype, np.uint8)
            with self.assertRaises(IndexError):
                source.read(10)
            source.close()


if __name__ == "__main__":
    unittest.main()
