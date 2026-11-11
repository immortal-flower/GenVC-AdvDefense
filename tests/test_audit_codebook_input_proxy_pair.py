import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from exp_flf2v.audit_codebook_input_proxy_pair import (
    classify_pair, history_summary, quantize_asset, real_encoder_cell,
)


class ProxyPairAuditTest(unittest.TestCase):
    def test_triage_stops_when_proxy_does_not_beat_random(self):
        self.assertEqual(classify_pair(-0.01, 0.02),
                         "candidate_did_not_beat_random_under_its_own_proxy")

    def test_history_is_pre_update(self):
        report = history_summary([0.8, 0.9, 0.85])
        self.assertAlmostEqual(report["pre_update_change"], 0.05)
        self.assertEqual(report["upward_intervals"], 1)

    def test_real_hard_ratio(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.npz"
            residual = np.ones((1, 1, 9, 1, 2), dtype=np.float32)
            correction = np.full_like(residual, 0.5)
            np.savez(path, residual_step1=residual, correction_step1=correction)
            report = real_encoder_cell(path, 3)
            self.assertAlmostEqual(report["hard_uncovered_ratio"], 0.25)

    def test_quantization_matches_codec_rounding(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "asset.npz"
            original = np.array([[[[100, 101, 102]]]], dtype=np.uint8)
            delta = np.array([[[[1 / 255, -2 / 255, 0]]]], dtype=np.float32)
            np.savez(path, delta=delta)
            actual = quantize_asset(original, path)
            np.testing.assert_array_equal(actual, [[[[101, 99, 102]]]])


if __name__ == "__main__":
    unittest.main()
