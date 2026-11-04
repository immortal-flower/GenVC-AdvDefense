import json
import tempfile
import unittest
from pathlib import Path

from exp_flf2v.plot_sign_uep_matched_ber import (
    collect_rows,
    protected_run_name,
    unprotected_run_name,
)


class MatchedBerPlotTests(unittest.TestCase):
    def test_run_names_cover_legacy_protected_result(self):
        self.assertEqual(protected_run_name(64, 42),
                         "sign64_hotspot_sparse_rep3")
        self.assertEqual(
            unprotected_run_name(21, 123, "Jockey"),
            "sign_uep_unprotected_count21_aseed123_jockey")
        self.assertEqual(
            protected_run_name(64, 123, "Jockey", 1),
            "sign_uep_rep3_count64_aseed123_gop1_jockey")
        self.assertEqual(
            unprotected_run_name(21, 123, "Jockey", 1),
            "sign_uep_unprotected_count21_aseed123_gop1_jockey")

    def test_collect_rows_matches_local_error_rates(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            for protected_budget, unprotected_count in (
                    (32, 11), (64, 21), (96, 32), (128, 43), (192, 64)):
                for seed in (42, 123, 2026):
                    for run_name, psnr, lpips, residual in (
                        (protected_run_name(protected_budget, seed),
                         33.0, 0.08, 2),
                        (unprotected_run_name(unprotected_count, seed, "Jockey"),
                         32.0, 0.10, unprotected_count),
                    ):
                        path = root / run_name / "Jockey" / "gop0" / "metrics.json"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps({
                            "PSNR_dB": psnr,
                            "LPIPS": lpips,
                            "stream_attack": {"changed_symbols": residual},
                        }), encoding="utf-8")
            rows, missing = collect_rows(
                root, "Jockey", 0, (42, 123, 2026), 33.47, 0.0757)
        self.assertFalse(missing)
        self.assertEqual(len(rows), 5)
        self.assertAlmostEqual(rows[0]["local_ber_percent"], 32 / 576 * 100)
        self.assertAlmostEqual(rows[0]["psnr_recovery_gain_db"], 1.0)


if __name__ == "__main__":
    unittest.main()
