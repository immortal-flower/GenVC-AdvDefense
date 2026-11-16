import unittest
import numpy as np
from exp_flf2v.rgb_direction_audit import recover_first_direction


class DirectionRecoveryTest(unittest.TestCase):
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
