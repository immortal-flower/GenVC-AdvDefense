import json
import tempfile
import unittest
from pathlib import Path

from exp_flf2v.audit_experiment_evidence import build_ledger


class EvidenceAuditTests(unittest.TestCase):
    def test_legacy_and_formal_runs_are_distinguished(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "legacy" / "Clip" / "gop0"
            legacy.mkdir(parents=True)
            (legacy / "metrics.json").write_text(json.dumps({
                "sequence": "Clip", "gop": 0, "PSNR_dB": 30.0,
                "LPIPS": 0.1, "per_frame_PSNR_dB": [30.0],
                "codebook_bits": 10, "boundary_bytes": 2,
            }), encoding="utf-8")
            formal = root / "formal" / "Clip" / "gop0"
            formal.mkdir(parents=True)
            (formal / "metrics.json").write_text(json.dumps({
                "sequence": "Clip", "gop": 0, "PSNR_dB": 30.0,
                "LPIPS": 0.1, "per_frame_PSNR_dB": [30.0],
                "codebook_bits": 10, "boundary_bytes": 2,
                "provenance": {
                    "configuration_id": "abc123456789",
                    "model": {"model_seed": 42},
                    "source": {"gop_segment_sha256": "deadbeef"},
                },
            }), encoding="utf-8")
            rows = build_ledger(root)
        by_run = {row["run"]: row for row in rows}
        self.assertFalse(by_run["legacy"]["formal_ready"])
        self.assertTrue(by_run["formal"]["formal_ready"])
        self.assertIn("missing_configuration_id", by_run["legacy"]["issues"])


if __name__ == "__main__":
    unittest.main()
