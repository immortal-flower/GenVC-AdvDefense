import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from exp_flf2v.analyze_codebook_input_pilot import assess_pair, trace_shift_summary


class PairedScreenTest(unittest.TestCase):
    def test_trace_shift_reports_target_position_only(self):
        with TemporaryDirectory() as tmp:
            clean_dir = Path(tmp) / "clean"
            run_dir = Path(tmp) / "run"
            clean_dir.mkdir()
            run_dir.mkdir()
            shape = (1, 2, 9, 3, 4)
            clean = np.zeros(shape, dtype=np.float32)
            changed = clean.copy()
            changed[:, :, 3] = 2.0
            changed[:, :, 4] = 9.0
            np.savez(clean_dir / "input_attack_trace.npz", target_latent=clean)
            np.savez(run_dir / "input_attack_trace.npz", target_latent=changed)
            result = trace_shift_summary(clean_dir, run_dir)
            self.assertEqual(result["target_latent_cell3_MSE"], 4.0)

    def test_promising_requires_both_metrics_and_submitted_reference(self):
        random = dict(input_MSE=0.00004, input_Linf_pixel=2.0,
                      PSNR_dB=33.2, LPIPS=0.08, PSNR_vs_submitted_dB=33.0)
        candidate = dict(input_MSE=0.00004, input_Linf_pixel=2.0,
                         PSNR_dB=32.7, LPIPS=0.09, PSNR_vs_submitted_dB=32.8)
        self.assertEqual(assess_pair(random, candidate)["screening_result"],
                         "promising_single_GOP")
        candidate["LPIPS"] = 0.07
        self.assertEqual(assess_pair(random, candidate)["screening_result"],
                         "not_yet_consistent")

    def test_rejects_unmatched_input_magnitude(self):
        random = dict(input_MSE=0.00004, input_Linf_pixel=2.0,
                      PSNR_dB=33.2, LPIPS=0.08, PSNR_vs_submitted_dB=33.0)
        candidate = dict(input_MSE=0.00008, input_Linf_pixel=2.0,
                         PSNR_dB=32.7, LPIPS=0.09, PSNR_vs_submitted_dB=32.8)
        self.assertFalse(assess_pair(random, candidate)["input_MSE_matched_within_1pct"])

    def test_small_positive_differences_are_not_called_promising(self):
        random = dict(input_MSE=0.00004, input_Linf_pixel=2.0,
                      PSNR_dB=33.44, LPIPS=0.0757, PSNR_vs_submitted_dB=33.365)
        candidate = dict(input_MSE=0.00004, input_Linf_pixel=2.0,
                         PSNR_dB=33.34, LPIPS=0.0758, PSNR_vs_submitted_dB=33.273)
        self.assertEqual(assess_pair(random, candidate)["screening_result"],
                         "weak_single_GOP_signal")


if __name__ == "__main__":
    unittest.main()
