import unittest
import numpy as np
from exp_flf2v.analyze_vae_error_inheritance import association


class AssociationTest(unittest.TestCase):
    def test_positive(self):
        result = association(np.arange(33),np.arange(33)*2)
        self.assertAlmostEqual(result['Pearson'],1)
        self.assertEqual(result['BothPositiveFrames'],32)

    def test_constant_is_not_correlation(self):
        self.assertIsNone(association(np.ones(33),np.ones(33))['Pearson'])

    def test_invalid(self):
        with self.assertRaises(ValueError):
            association(np.zeros(32),np.zeros(32))
        with self.assertRaises(ValueError):
            association(np.full(33,np.nan),np.zeros(33))


if __name__ == '__main__':
    unittest.main()
