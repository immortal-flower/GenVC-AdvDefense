import importlib.util
import math
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
ANALYSIS_PATH = ROOT / "exp_flf2v" / "analyze_combinatorial_index_robustness.py"
analysis_spec = importlib.util.spec_from_file_location("comb_analysis", ANALYSIS_PATH)
analysis = importlib.util.module_from_spec(analysis_spec)
analysis_spec.loader.exec_module(analysis)

# bitstream_attacks only needs these three functions at import time.  Supplying
# a lightweight module keeps this format test CPU-only and independent of torch.
fake_codebook = types.ModuleType("sde_rf_wan.turbo_codebook")
fake_codebook.combinatorial_index_bits = analysis.combinatorial_index_bits
fake_codebook.unrank_combination = analysis.unrank_combination
sys.modules["sde_rf_wan.turbo_codebook"] = fake_codebook

ATTACK_PATH = ROOT / "exp_flf2v" / "bitstream_attacks.py"
attack_spec = importlib.util.spec_from_file_location("rank_attacks", ATTACK_PATH)
attacks = importlib.util.module_from_spec(attack_spec)
attack_spec.loader.exec_module(attacks)


class CombinatorialRankAttackTest(unittest.TestCase):
    def test_exact_rank_bit_changes_real_tdc2_payload(self):
        K, M = 32, 5
        indices = [0, 1, 2, 3, 4]
        rank = analysis.rank_combination(indices, K)
        rank_bits = analysis.combinatorial_index_bits(K, M)
        data = bytearray(b"TDC2")
        data.extend(struct.pack("<10I", K, M, 1, 0, 1, 42, 3, 9, 16, 24))
        data.extend(struct.pack("<I", 0))
        data.extend(struct.pack("<3I", 2, 3, 4))
        data.extend(struct.pack("<H", M))
        data.extend(rank.to_bytes((rank_bits + 7) // 8, "little"))
        data.append(0b10101)
        with tempfile.TemporaryDirectory() as tmp:
            clean = Path(tmp) / "clean.tdcm"
            attacked = Path(tmp) / "attacked.tdcm"
            clean.write_bytes(data)
            metadata = attacks.apply_serialized_codebook_attack(
                clean, attacked, attack="rank-bitflip", count=1,
                target_steps=[0], target_frames=[0],
                rank_region="low", rank_bits=[0])
            parsed = attacks._parse_tdcm_payload_layout(
                bytearray(attacked.read_bytes()))
        self.assertEqual(metadata["channel_flipped_bits"], 1)
        self.assertEqual(metadata["changed_symbols"], 1)
        self.assertEqual(metadata["examples"][0]["flipped_bit"], 0)
        self.assertEqual(parsed["rank_entries"][0]["rank"], 1)
        self.assertEqual(parsed["rank_entries"][0]["indices"], [0, 1, 2, 3, 5])


if __name__ == "__main__":
    unittest.main()
