import unittest

import numpy as np

from exp_flf2v.prepare_input_direction_calibration import matched_random, realize


class DirectionPreparationTests(unittest.TestCase):
    def test_realize_clips_and_rounds_like_codec(self):
        original = np.array([[[0, 100, 255]]], dtype=np.uint8)
        delta = np.array([[[[-2 / 255, 1 / 255, 2 / 255]]]], dtype=np.float32)
        self.assertEqual(realize(original[None], delta).tolist(), [[[[0, 1, 0]]]])

    def test_random_matches_realized_magnitude_with_saturation(self):
        original = np.array([[[[0, 255], [100, 101]]]], dtype=np.uint8)
        changed = np.array([[[[2, -2], [1, -1]]]], dtype=np.int16)
        control = matched_random(original, changed, seed=42)
        np.testing.assert_array_equal(np.abs(control), np.abs(changed))
        self.assertTrue(np.all((original.astype(np.int16) + control) >= 0))
        self.assertTrue(np.all((original.astype(np.int16) + control) <= 255))
        np.testing.assert_array_equal(control, matched_random(original, changed, seed=42))


if __name__ == "__main__":
    unittest.main()
