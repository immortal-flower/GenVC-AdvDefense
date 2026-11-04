import json
import tempfile
import unittest
from pathlib import Path

from exp_flf2v.plot_sign_uep_burst_interleave import collect, run_name


class BurstInterleavePlotTests(unittest.TestCase):
    def test_run_name_matches_batch_output(self):
        self.assertEqual(
            run_name(64, 42, 128, "Jockey", 0),
            "sign_uep_rep3_burst64_ileave42_start128_jockey")

    def test_collect_aggregates_three_burst_starts(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            for interleave_seed in (0, 42):
                for count in (32, 64, 96):
                    for start in (0, 128, 256):
                        name = run_name(
                            count, interleave_seed, start, "Jockey", 0)
                        path = root / name / "Jockey" / "gop0" / "metrics.json"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps({
                            "PSNR_dB": 33.0,
                            "LPIPS": 0.08,
                            "stream_attack": {
                                "changed_symbols": 2,
                                "corrected_logical_symbols": count - 2,
                            },
                        }), encoding="utf-8")
            detail, aggregate, missing = collect(
                root, "Jockey", 0, (32, 64, 96), (0, 128, 256),
                (0, 42), 33.47, 0.0757)
        self.assertFalse(missing)
        self.assertEqual(len(detail), 18)
        self.assertEqual(len(aggregate), 6)
        self.assertAlmostEqual(aggregate[0]["PSNR_drop_mean_dB"], 0.47)


if __name__ == "__main__":
    unittest.main()
