import tempfile
import unittest
from pathlib import Path

import torch

from worm_pose_gen.body_ensemble import BodyFieldEnsemble, FLIPS, save_ensemble
from worm_pose_gen.body_net import OUTPUTS, BodyFieldModule, load_body_net


def save_checkpoint(module: BodyFieldModule, path: Path) -> None:
    """A checkpoint with the keys Lightning's load_from_checkpoint reads."""

    torch.save({"state_dict": module.state_dict(), "hyper_parameters": dict(module.hparams),
                "pytorch-lightning_version": "2.0.0"}, str(path))


class Symmetric(torch.nn.Module):
    """A flip-equivariant toy network: a pointwise map of the input channels to the outputs."""

    def __init__(self) -> None:
        super().__init__()
        self.lags = (1,)
        self.conv = torch.nn.Conv2d(2, len(OUTPUTS), 1)

    def forward(self, images):
        return self.conv(images)


class EnsembleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.inputs = torch.randn(2, 3, 40, 64)

    def test_flip_tta_of_equivariant_network_equals_network(self):
        network = Symmetric()
        tta = BodyFieldEnsemble([network], tuple(FLIPS))
        images = torch.randn(2, 2, 12, 20)
        torch.testing.assert_close(torch.sigmoid(tta(images)), torch.sigmoid(network(images)), atol=1e-5, rtol=0)

    def test_ensemble_of_identical_members_equals_member(self):
        member = BodyFieldModule(lags=(1, 4), pretrained=False).eval()
        ensemble = BodyFieldEnsemble([member, member, member]).eval()
        with torch.no_grad():
            torch.testing.assert_close(torch.sigmoid(ensemble(self.inputs)), torch.sigmoid(member(self.inputs)), atol=1e-5, rtol=0)

    def test_mean_of_two_members_and_flipped_view(self):
        first, second = (BodyFieldModule(lags=(1, 4), pretrained=False).eval() for _ in range(2))
        ensemble = BodyFieldEnsemble([first, second], ("none", "h")).eval()
        with torch.no_grad():
            views = [torch.sigmoid(m(x)).flip(d) if d else torch.sigmoid(m(x))
                     for m in (first, second) for x, d in ((self.inputs, ()), (self.inputs.flip(-1), (-1,)))]
            torch.testing.assert_close(torch.sigmoid(ensemble(self.inputs)), torch.stack(views).mean(0), atol=1e-5, rtol=0)

    def test_members_must_share_lags(self):
        with self.assertRaises(ValueError):
            BodyFieldEnsemble([BodyFieldModule(lags=(1,), pretrained=False), BodyFieldModule(lags=(2,), pretrained=False)])
        with self.assertRaises(ValueError):
            BodyFieldEnsemble([BodyFieldModule(lags=(1,), pretrained=False)], ("none", "diagonal"))

    def test_checkpoint_round_trip_through_load_body_net(self):
        members = [BodyFieldModule(lags=(1, 4), pretrained=False).eval() for _ in range(2)]
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f"member{k}.ckpt" for k in range(2)]
            for member, path in zip(members, paths):
                save_checkpoint(member, path)
            single = load_body_net(paths[0], "cpu")
            self.assertIsInstance(single, BodyFieldModule)
            save_ensemble(Path(tmp) / "ensemble.ckpt", paths, ("none", "hv"))
            loaded = load_body_net(Path(tmp) / "ensemble.ckpt", "cpu")
        self.assertIsInstance(loaded, BodyFieldEnsemble)
        self.assertEqual(loaded.lags, (1, 4))
        self.assertEqual(loaded.device, torch.device("cpu"))
        self.assertFalse(loaded.training)
        self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
        expected = BodyFieldEnsemble(members, ("none", "hv")).eval()
        with torch.no_grad():
            torch.testing.assert_close(loaded(self.inputs), expected(self.inputs))
        with torch.no_grad():
            torch.testing.assert_close(single(self.inputs), members[0](self.inputs))


if __name__ == "__main__":
    unittest.main()
