"""The Workspace page's model-output layers: GET /api/workspaces/{name}/network-fields with a stub network, and the mask model's mask-probability."""

import base64
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient
import h5py
import numpy as np
from PIL import Image

from tests.test_body_proposal import StubModule
from worm_pose_gen.app import AppConfig, create_app
from worm_pose_gen.body_proposal import FieldPrediction
from worm_pose_gen.body_targets import point_heatmap
from worm_pose_gen.workspace import Workspace


H, W, FRAMES = 64, 96, 4


def decode(url):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("L"))


def prediction(head_peak=1.0):
    """A straight body along y=32 from x=10 to 85, A-P rising with x, a crossing at x 40-45, the tail near the right edge."""

    mask = np.zeros((H, W), np.float32)
    mask[26:39, 10:86] = 1
    ap = np.tile(np.linspace(0, 1, W, dtype=np.float32), (H, 1))
    overlap = np.zeros((H, W), np.float32)
    overlap[26:39, 40:46] = 1
    head = point_heatmap((H, W), np.array([30.0, 32.0]), 3.0) * head_peak
    tail = point_heatmap((H, W), np.array([88.0, 32.0]), 3.0)
    return FieldPrediction(mask=mask, ap=ap, head=head, tail=tail, overlap=overlap)


class RecordingStub(StubModule):
    """The stub network with lags, keeping every input batch it is given."""

    def __init__(self, fields, lags=()):
        super().__init__(fields)
        self.lags = tuple(lags)
        self.inputs = []

    def forward(self, images):
        self.inputs.append(images.clone())
        return self.logits[None].expand(len(images), *self.logits.shape)


class NetworkFieldsApiTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.recording = self.root / "movie.h5"
        rng = np.random.default_rng(0)
        frames = np.clip(rng.normal(190, 4, (FRAMES, H, W)), 0, 255).astype(np.uint8)
        for k in range(FRAMES):
            frames[k, 28:37, 10 + 3 * k : 80 + 3 * k] = 70  # the body moves right by 3 px per frame
        with h5py.File(self.recording, "w") as handle:
            handle.create_dataset("/img_nir", data=frames)
        self.checkpoint = self.root / "body_net.ckpt"
        self.checkpoint.write_bytes(b"stub")
        self.workspace = Workspace.create(self.root / "workspaces", "demo", self.recording, 0, FRAMES - 1, 1)

    def client(self, body_net):
        """The app, with ``body_net`` as the body model the workspace was analysed with (``None``: none)."""

        self.workspace.info.settings["body_net"] = None if body_net is None else str(body_net)
        self.workspace.save_info()
        config = AppConfig(workspaces_root=self.root / "workspaces", dataset_root=self.root / "cache", device="cpu", gpus=(),
                           lab_library=self.root / "lab", library=self.root / "mine")
        app = create_app(config)
        client = TestClient(app, raise_server_exceptions=False)
        self.addCleanup(client.close)
        self.addCleanup(app.state.app_state.close)
        return app, client

    def get(self, client, frame, status=200):
        response = client.get("/api/workspaces/demo/network-fields", params={"frame": frame})
        self.assertEqual(response.status_code, status, response.text)
        return response.json()

    def test_without_a_network_the_route_says_so(self):
        _, client = self.client(None)
        self.assertIn("without a body-field model", self.get(client, 1, 400)["error"])
        _, client = self.client(self.root / "absent.ckpt")
        self.assertIn("without a body-field model", self.get(client, 1, 400)["error"])

    def test_encoding_ends_and_cache(self):
        app, client = self.client(self.checkpoint)
        stub = RecordingStub(prediction())
        with mock.patch("worm_pose_gen.body_net.load_body_net", return_value=stub) as load:
            first = self.get(client, 1)
            again = self.get(client, 1)
        load.assert_called_once()
        self.assertEqual((first["cached"], again["cached"], len(stub.inputs)), (False, True, 1))
        self.assertEqual((first["width"], first["height"], first["model"], first["row"]), (W, H, "stub.ckpt", 1))
        ap, overlap = decode(first["ap"]), decode(first["overlap"])
        self.assertEqual(ap[5, 50], 0, "undefined off the predicted mask")
        self.assertEqual(ap[32, 42], 0, "undefined on a predicted crossing")
        self.assertEqual(overlap[32, 42], 255)
        self.assertEqual(overlap[5, 42], 0)
        self.assertAlmostEqual((int(ap[32, 60]) - 1) / 254, 60 / (W - 1), delta=0.01)
        # The head is used; the tail peak is strong but within END_BORDER_PX of the edge, so the fitter drops it.
        self.assertEqual(first["head_xy"], [30.0, 32.0])
        self.assertIsNone(first["tail_xy"])
        self.assertGreater(first["tail_peak"], 0.9)
        self.assertEqual(first["mask_source"], "predicted")
        # A mask edit is a new revision: predicted again, and the ends are judged against that mask.
        labels = np.zeros((H, W), np.uint8)
        labels[26:39, 60:86] = 1
        self.workspace.set_override_mask(1, labels)
        edited = self.get(client, 1)
        self.assertEqual((edited["cached"], len(stub.inputs), edited["mask_source"]), (False, 2, "workspace"))
        self.assertIsNone(edited["head_xy"], "a head 30 px from the current mask is dropped")
        # The LRU stays bounded.
        app.state.app_state.network_fields.entries = 2
        for frame in (0, 2, 3):
            self.get(client, frame)
        self.assertEqual(len(app.state.app_state.network_fields._cache), 2)
        self.assertIn("is not in workspace", self.get(client, 99, 400)["error"])

    def test_raw_outputs_on_request(self):
        _, client = self.client(self.checkpoint)
        with mock.patch("worm_pose_gen.body_net.load_body_net", return_value=RecordingStub(prediction(head_peak=0.5))):
            plain = self.get(client, 1)
            response = client.get("/api/workspaces/demo/network-fields", params={"frame": 1, "outputs": 1})
        self.assertNotIn("outputs", plain)
        raw = response.json()
        self.assertTrue(raw["cached"], "the raw channels come from the cached prediction")
        self.assertEqual(sorted(raw["outputs"]), ["ap", "head", "mask", "overlap", "tail"])
        outputs = {name: decode(url) for name, url in raw["outputs"].items()}
        # Unthresholded and unmasked: the A-P field is there off the body, the overlap is not cut from the mask.
        self.assertAlmostEqual(outputs["ap"][5, 50] / 255, 50 / (W - 1), delta=0.01)
        self.assertEqual((outputs["mask"][32, 50], outputs["mask"][5, 50], outputs["overlap"][32, 42]), (255, 0, 255))
        self.assertAlmostEqual(outputs["head"][32, 30] / 255, 0.5, delta=0.01)
        self.assertAlmostEqual(raw["peaks"]["head"], 0.5, places=3)
        self.assertAlmostEqual(raw["peaks"]["mask"], 1.0, places=3)

    def test_the_mask_models_probability_and_one_model_for_both(self):
        segmenter = self.root / "segmenter.ckpt"
        segmenter.write_bytes(b"stub")
        self.workspace.info.settings["checkpoint"] = str(segmenter)
        app, client = self.client(self.checkpoint)
        probability = np.zeros((H, W), np.float32)
        probability[26:39, 10:86] = 0.6
        with mock.patch.object(app.state.app_state.segmenters, "probability", return_value=(probability, str(segmenter))) as segment:
            answer = client.get("/api/workspaces/demo/mask-probability", params={"frame": 2}).json()
        self.assertEqual(segment.call_args.args[0], str(segmenter))
        self.assertEqual((answer["row"], answer["model"]), (2, "segmenter.ckpt"))
        self.assertAlmostEqual(answer["peak"], 0.6, places=5)
        values = decode(answer["probability"])
        self.assertEqual((values[32, 50], values[5, 50]), (153, 0))
        status = client.get("/api/workspaces/demo/status").json()
        self.assertEqual((status["has_mask_model"], status["has_body_model"]), (True, True))
        # Masks from the body model: it is the only model, and there is no mask model's probability.
        self.workspace.info.settings["mask_source"] = "body_net"
        _, client = self.client(self.checkpoint)
        status = client.get("/api/workspaces/demo/status").json()
        self.assertEqual((status["has_mask_model"], status["has_body_model"]), (False, True))
        refused = client.get("/api/workspaces/demo/mask-probability", params={"frame": 2})
        self.assertEqual(refused.status_code, 400)
        self.assertIn("has no mask model", refused.json()["error"])

    def test_a_weak_head_peak_is_no_head(self):
        _, client = self.client(self.checkpoint)
        with mock.patch("worm_pose_gen.body_net.load_body_net", return_value=RecordingStub(prediction(head_peak=0.2))):
            body = self.get(client, 2)
        self.assertIsNone(body["head_xy"])
        self.assertAlmostEqual(body["head_peak"], 0.2, places=3)

    def test_neighbours_outside_the_recording_give_zero_lag_channels(self):
        _, client = self.client(self.checkpoint)
        stub = RecordingStub(prediction(), lags=(1, 2))
        with mock.patch("worm_pose_gen.body_net.load_body_net", return_value=stub):
            self.get(client, 0)
            self.get(client, 1)
            self.get(client, 3)
        first, second, last = (inputs[0].numpy() for inputs in stub.inputs)
        self.assertEqual(first.shape, (3, H, W))
        self.assertFalse(first[1].any() or first[2].any(), "frames -1 and -2 do not exist")
        self.assertTrue(second[1].any(), "frames 2 and 0 exist and differ")
        self.assertFalse(second[2].any(), "frame -1 does not exist")
        self.assertFalse(last[1].any() or last[2].any(), "frames 4 and 5 do not exist")
        self.assertGreater(float(np.abs(second[0]).max()), 0.0)


if __name__ == "__main__":
    unittest.main()
