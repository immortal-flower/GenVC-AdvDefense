import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from exp_flf2v.codebook_validation_plan import ValidationJob
from exp_flf2v.experiment_provenance import (
    build_job_provenance,
    build_shared_validation_provenance,
    configuration_id,
    formal_run_name,
    metrics_matches_provenance,
)


class ExperimentProvenanceTests(unittest.TestCase):
    def make_args(self, directory):
        directory = Path(directory)
        data = directory / "Clip_2x2.yuv"
        # 2x2 YUV420 is 6 bytes/frame; two frames are sufficient here.
        data.write_bytes(bytes(range(12)))
        codebook = directory / "codebook.tdcm"
        codebook.write_bytes(b"TDC2-demo")
        metrics = directory / "metrics.json"
        metrics.write_text('{"PSNR_dB": 30.0}', encoding="utf-8")
        return SimpleNamespace(
            data=str(data), clean_codebook=str(codebook),
            clean_metrics=str(metrics), sequence="Clip", gop_index=0,
            num_frames=2, height=2, width=2, model_seed=42,
            wan_ckpt="checkpoints/Wan2.1", K=16, M=2, steps=4,
            ddim_tail=1, g_scale=3.0, flow_shift=5.0,
            ref_codec="compressai", ref_quality=4)

    def test_configuration_changes_with_attack_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            shared = build_shared_validation_provenance(
                self.make_args(directory))
            first = build_job_provenance(
                shared, ValidationJob("hot", 1, 3, 16, 42))
            second = build_job_provenance(
                shared, ValidationJob("hot", 1, 3, 16, 123))
        self.assertNotEqual(first["configuration_id"],
                            second["configuration_id"])

    def test_configuration_is_path_independent(self):
        records = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as directory:
                shared = build_shared_validation_provenance(
                    self.make_args(directory))
                records.append(build_job_provenance(
                    shared, ValidationJob("hot", 1, 3, 16, 42)))
        self.assertEqual(records[0]["configuration_id"],
                         records[1]["configuration_id"])

    def test_formal_name_and_strict_metrics_match(self):
        config = {"schema": "x", "value": 1}
        config["configuration_id"] = configuration_id(config)
        name = formal_run_name("run", config, "Main Experiment")
        self.assertIn("main_experiment_cfg", name)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text(json.dumps({"provenance": config}),
                            encoding="utf-8")
            self.assertTrue(metrics_matches_provenance(path, config))
            other = dict(config, configuration_id="deadbeef0000")
            self.assertFalse(metrics_matches_provenance(path, other))


if __name__ == "__main__":
    unittest.main()
