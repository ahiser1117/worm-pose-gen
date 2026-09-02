from __future__ import annotations

from importlib import resources
import unittest

import numpy as np

from worm_pose_gen.classical import ClassicalConfig
from worm_pose_gen.heuristic_tuner import analyze_image, encode_png, parse_config


class HeuristicTunerTests(unittest.TestCase):
    def test_pipeline_defaults_remain_frozen_evidence_setting(self) -> None:
        # The interactively tuned 61/3/4.25/2.05/8 setting was evaluated on the
        # 30-frame stress set and rejected; the frozen defaults stay in force.
        config = ClassicalConfig()
        self.assertEqual(config.local_radius, 31)
        self.assertEqual(config.smooth_radius, 2)
        self.assertEqual(config.foreground_z, 2.6)
        self.assertIsNone(config.connected_foreground_z)
        self.assertEqual(config.close_radius, 2)

    def test_parse_config_enables_connected_hysteresis(self) -> None:
        config = parse_config({
            "local_radius": 29,
            "smooth_radius": 1,
            "foreground_z": 2.7,
            "connected_enabled": True,
            "connected_foreground_z": 1.6,
            "close_radius": 3,
        })
        self.assertEqual(config.local_radius, 29)
        self.assertEqual(config.connected_foreground_z, 1.6)

    def test_parse_config_disables_connected_hysteresis(self) -> None:
        config = parse_config({
            "foreground_z": 2.6,
            "connected_enabled": False,
            "connected_foreground_z": 5.0,
        })
        self.assertIsNone(config.connected_foreground_z)

    def test_parse_config_rejects_inverted_cutoffs(self) -> None:
        with self.assertRaisesRegex(ValueError, "below the primary cutoff"):
            parse_config({
                "foreground_z": 1.5,
                "connected_enabled": True,
                "connected_foreground_z": 1.8,
            })

    def test_png_encoder_supports_rgba(self) -> None:
        image = np.zeros((9, 13, 4), dtype=np.uint8)
        image[2:5, 4:8] = (10, 20, 30, 200)
        encoded = encode_png(image)
        self.assertTrue(encoded.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertGreater(len(encoded), 40)

    def test_analysis_returns_all_preview_layers(self) -> None:
        image = np.full((45, 60), 180, dtype=np.uint8)
        image[19:26, 10:48] = 90
        payload = analyze_image(
            image,
            ClassicalConfig(
                local_radius=9,
                smooth_radius=1,
                foreground_z=1.5,
                connected_foreground_z=0.8,
                close_radius=1,
            ),
        )
        self.assertEqual(set(payload["images"]), {"frame", "score", "candidate", "kept"})
        self.assertTrue(
            all(
                value.startswith("data:image/png;base64,")
                for value in payload["images"].values()
            )
        )
        self.assertIn("retained_component_area", payload["metrics"])

    def test_ui_assets_include_primary_controls(self) -> None:
        root = resources.files("worm_pose_gen.heuristic_tuner_ui")
        html = root.joinpath("index.html").read_text()
        script = root.joinpath("app.js").read_text()
        self.assertIn('id="foreground-z"', html)
        self.assertIn('id="connected-foreground-z"', html)
        self.assertIn('fetch("/api/analyze"', script)


if __name__ == "__main__":
    unittest.main()
