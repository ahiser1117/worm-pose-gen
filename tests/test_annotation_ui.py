from __future__ import annotations

from importlib import resources
import unittest


class AnnotationUITests(unittest.TestCase):
    def test_hidden_state_overrides_loading_overlay_display(self) -> None:
        css = resources.files("worm_pose_gen.annotation_ui").joinpath("style.css").read_text()
        self.assertIn("[hidden] { display: none !important; }", css)


if __name__ == "__main__":
    unittest.main()
