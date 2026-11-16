import unittest
import numpy as np
from exp_flf2v.rgb_direction_audit import recover_first_direction, assert_same_submitted_pixels


class DirectionRecoveryTest(unittest.TestCase):
    def test_submitted_equality_accepts_float_roundoff(self):
        original = np.array([0., 127/255, 1.], dtype=np.float32)
        expected = np.array([2/255, -2/255, -1/255], dtype=np.float32)
        actual = expected + np.float32(3e-8)
        self.assertFalse(np.array_equal(actual, expected))
        result = assert_same_submitted_pixels(original, actual, expected, 'boundary')
        self.assertEqual(result['changed_submitted_channels'], 0)
        self.assertGreater(result['max_float_delta_difference'], 0)

    def test_submitted_equality_rejects_one_level_and_rounding_boundary(self):
        original = np.array([100/255], dtype=np.float32)
        for actual, expected in ((1/255, 0.), (.501/255, .499/255)):
            with self.subTest(actual=actual), self.assertRaisesRegex(RuntimeError, '1 submitted RGB channels'):
                assert_same_submitted_pixels(original, np.array([actual]), np.array([expected]), 'boundary')

    def test_submitted_equality_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            assert_same_submitted_pixels(np.zeros(2), np.zeros(1), np.zeros(2), 'boundary')
        with self.assertRaises(ValueError):
            assert_same_submitted_pixels(np.zeros(1), np.array([np.nan]), np.zeros(1), 'boundary')

    def inputs(self, scale=1.):
        report = dict(status='completed', fingerprint='abc',
                      configuration=dict(objective='final-rgb', iterations=1, alpha=1., epsilon=4.),
                      history=[dict(iteration=1, accepted_scale=scale)])
        checkpoint = dict(done=np.array(1), fingerprint=np.array('abc'),
                          controls=np.array([-1., 0., 1.], dtype=np.float32) * (scale / 255))
        return report, checkpoint

    def test_exact_single_step_direction(self):
        for scale in (1., .5):
            report, checkpoint = self.inputs(scale)
            direction, step = recover_first_direction(report, checkpoint)
            np.testing.assert_array_equal(direction, [-1, 0, 1])
            self.assertEqual(step, scale)

    def test_rejects_no_update_multiple_updates_and_wrong_checkpoint(self):
        for field, value in (('done', np.array(2)), ('fingerprint', np.array('wrong'))):
            report, checkpoint = self.inputs()
            checkpoint[field] = value
            with self.assertRaises(ValueError):
                recover_first_direction(report, checkpoint)
        report, checkpoint = self.inputs(0.)
        with self.assertRaises(ValueError):
            recover_first_direction(report, checkpoint)

    def test_rejects_non_sign_controls(self):
        report, checkpoint = self.inputs()
        checkpoint['controls'][0] *= .6
        with self.assertRaises(ValueError):
            recover_first_direction(report, checkpoint)


if __name__ == '__main__':
    unittest.main()
