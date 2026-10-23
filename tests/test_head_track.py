"""The head track's Viterbi decode (``head_track``) on synthetic candidate sequences."""

import unittest

import numpy as np

from worm_pose_gen.head_track import HIDDEN_COST, track_heads

K = 4


def walking(n, start=(100.0, 100.0), step=(3.0, 0.0), value=0.7):
    """``n`` rows of modes with the true head walking ``step`` px a frame, as mode 0."""

    modes = np.full((n, K, 3), np.nan)
    modes[:, 0, :2] = np.asarray(start) + np.arange(n)[:, None] * np.asarray(step)
    modes[:, 0, 2] = value
    return modes


class HeadTrackTests(unittest.TestCase):
    def test_a_stronger_mode_on_the_other_turn_does_not_take_the_head(self):
        modes = walking(30, value=0.45)
        flips = [10, 11, 20]
        for t in range(30):
            # A second mode about 95 px away all along, stronger on the flip frames (2023-03-30-01, frames 4720, 4725).
            modes[t, 1] = (modes[t, 0, 0] - 95.0, modes[t, 0, 1] + 10.0, 0.6 if t in flips else 0.2)
        track = track_heads(np.arange(30), modes)
        np.testing.assert_array_equal(track.xy, modes[:, 0, :2])
        np.testing.assert_array_equal(track.candidate, np.zeros(30))
        argmax = np.where((modes[:, 1, 2] > modes[:, 0, 2])[:, None], modes[:, 1, :2], modes[:, 0, :2])
        self.assertEqual(int(np.sum(np.linalg.norm(np.diff(argmax, axis=0), axis=1) > 44)), 4)

    def test_a_head_that_leaves_view_is_none_until_it_returns(self):
        modes = walking(40)
        modes[12:25] = np.nan
        modes[12:25, 0] = (400.0, 400.0, 0.05)  # weak, far off
        track = track_heads(np.arange(40), modes)
        self.assertTrue(np.isnan(track.xy[12:25]).all())
        np.testing.assert_array_equal(track.candidate[12:25], -1)
        np.testing.assert_array_equal(track.xy[:12], modes[:12, 0, :2])
        np.testing.assert_array_equal(track.xy[25:], modes[25:, 0, :2])

    def test_a_weak_mode_on_the_track_bridges_a_dip_at_the_interpolated_place(self):
        modes = walking(30)
        modes[14:17, 0, 1] += 4.0  # 4 px off the walk
        modes[14:17, 0, 2] = 0.15  # below END_THRESHOLD
        modes[14:17, 1] = (400.0, 300.0, 0.25)  # stronger, far off
        track = track_heads(np.arange(30), modes)
        np.testing.assert_array_equal(track.candidate, np.zeros(30))
        expected = walking(30)[:, 0, :2]
        np.testing.assert_allclose(track.xy, expected)
        np.testing.assert_array_equal(np.flatnonzero(track.interpolated), [14, 15, 16])

    def test_weak_heads_interpolate_in_substrate_coordinates(self):
        modes = walking(30, step=(0.0, 0.0), value=0.6)
        shift = np.zeros((30, 2))
        # The stage moves 4 px a frame from frame 10 on; the head stays put on the agar.
        shift[10:, 0] = -4.0 * np.arange(1, 21)
        modes[:, 0, 0] -= shift[:, 0]
        modes[12:18, 0, 2] = 0.2
        track = track_heads(np.arange(30), modes, shift_px=shift)
        np.testing.assert_allclose(track.xy, modes[:, 0, :2])

    def test_a_weak_head_beside_a_stretch_out_of_view_is_dropped(self):
        modes = walking(30)
        modes[10:15] = np.nan  # out of view
        modes[15:18, 0, 2] = 0.2  # weak on the way back in
        track = track_heads(np.arange(30), modes)
        self.assertTrue(np.isnan(track.xy[10:18]).all())
        np.testing.assert_array_equal(track.candidate[15:18], -1)
        np.testing.assert_array_equal(track.xy[18:], modes[18:, 0, :2])

    def test_no_heads_anywhere(self):
        modes = np.full((5, K, 3), np.nan)
        track = track_heads(np.arange(5), modes)
        self.assertTrue(np.isnan(track.xy).all())
        self.assertEqual(len(track_heads(np.arange(0), np.zeros((0, K, 3))).xy), 0)

    def test_the_head_starts_and_ends_out_of_view(self):
        modes = walking(20)
        modes[:5] = np.nan
        modes[15:] = np.nan
        track = track_heads(np.arange(20), modes)
        np.testing.assert_array_equal(np.isfinite(track.xy[:, 0]), np.arange(20) // 5 % 3 != 0)

    def test_a_mode_on_a_stronger_tail_is_not_the_head(self):
        modes = walking(20, value=0.4)
        tail = np.zeros((20, 3))
        tail[:, :2] = modes[:, 0, :2] + (300.0, 0.0)
        tail[:, 2] = 0.9
        # From frame 8 on the head heatmap fires only on the tail (the head inside a coil).
        modes[8:, 0] = np.c_[tail[8:, :2] + 2.0, np.full(12, 0.45)]
        track = track_heads(np.arange(20), modes, tail=tail)
        self.assertTrue(np.isnan(track.xy[8:]).all())
        without = track_heads(np.arange(20), modes)
        self.assertTrue(np.isfinite(without.xy[8:]).all(), "without the tail the mode on it would have been taken")

    def test_the_nose_bridges_frames_without_a_mode(self):
        modes = walking(30)
        nose = modes[:, 0, :2] + 3.0
        nose[::2] = np.nan  # confident on every other frame only
        modes[10:20] = np.nan
        track = track_heads(np.arange(30), modes, nose_xy=nose)
        bridged = np.arange(10, 20)
        self.assertTrue(np.isfinite(track.xy[bridged[bridged % 2 == 1]]).all())
        self.assertTrue((track.candidate[bridged[bridged % 2 == 1]] == K).all())
        np.testing.assert_array_equal(track.candidate[:10], 0)

    def test_missing_frames_stretch_the_motion_scale(self):
        # Rows are frames 0-9 and 60-69: the head walked 150 px while the frames were not fit.
        frames = np.r_[np.arange(10), np.arange(60, 70)]
        modes = walking(70, step=(2.5, 0.0), value=0.5)[frames]
        # A decoy as strong, 300 px off the walk, after the gap.
        modes[10:, 1] = np.c_[modes[10:, 0, :2] + (0.0, 300.0), np.full(10, 0.5)]
        track = track_heads(frames, modes)
        np.testing.assert_array_equal(track.xy, modes[:, 0, :2])

    def test_a_substrate_shift_undoes_a_stage_move(self):
        modes = walking(30, value=0.5)
        shift = np.zeros((30, 2))
        # The stage moves at frame 15: the head jumps 120 px in the image, and a decoy appears where it was.
        modes[15:, 0, 0] += 120.0
        shift[15:, 0] = -120.0
        modes[15:, 1] = np.c_[modes[15:, 0, :2] - (120.0, 0.0), np.full(15, 0.55)]
        np.testing.assert_array_equal(track_heads(np.arange(30), modes, shift_px=shift).xy, modes[:, 0, :2])
        unshifted = track_heads(np.arange(30), modes).xy
        np.testing.assert_array_equal(unshifted[15:], modes[15:, 1, :2])

    def test_a_long_gap_forgets_the_position(self):
        n = 300
        modes = walking(n, step=(0.0, 0.0))
        modes[50:250] = np.nan
        modes[250:, 0, :2] = (700.0, 500.0)  # back far away after 200 frames out of view
        track = track_heads(np.arange(n), modes)
        np.testing.assert_array_equal(np.isfinite(track.xy[:, 0]), (np.arange(n) < 50) | (np.arange(n) >= 250))
        self.assertGreater(HIDDEN_COST, -np.log(0.3))


if __name__ == "__main__":
    unittest.main()
