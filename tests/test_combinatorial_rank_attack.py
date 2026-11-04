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
    @staticmethod
    def _tiny_tdc2():
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
        return data, rank_bits

    def test_exact_rank_bit_changes_real_tdc2_payload(self):
        data, _ = self._tiny_tdc2()
        with tempfile.TemporaryDirectory() as tmp:
            clean = Path(tmp) / "clean.tdcm"
            attacked = Path(tmp) / "attacked.tdcm"
            clean.write_bytes(data)
            metadata = attacks.apply_serialized_codebook_attack(
                clean, attacked, attack="rank-bitflip", count=1,
                target_cells=[(0, 0)],
                rank_region="low", rank_bits=[0])
            parsed = attacks._parse_tdcm_payload_layout(
                bytearray(attacked.read_bytes()))
        self.assertEqual(metadata["channel_flipped_bits"], 1)
        self.assertEqual(metadata["changed_symbols"], 1)
        self.assertEqual(metadata["examples"][0]["flipped_bit"], 0)
        self.assertEqual(metadata["target_cells_zero_based"], [[0, 0]])
        self.assertEqual(parsed["rank_entries"][0]["rank"], 1)
        self.assertEqual(parsed["rank_entries"][0]["indices"], [0, 1, 2, 3, 5])

    def test_rank_repetition_corrects_one_physical_error(self):
        data, rank_bits = self._tiny_tdc2()
        high_start = 2 * rank_bits // 3
        with tempfile.TemporaryDirectory() as tmp:
            clean = Path(tmp) / "clean.tdcm"
            protected = Path(tmp) / "clean.tdr2"
            disturbed = Path(tmp) / "disturbed.tdr2"
            recovered = Path(tmp) / "recovered.tdcm"
            clean.write_bytes(data)
            metadata = attacks.apply_serialized_rank_repetition_disturbance(
                clean, protected, disturbed, recovered,
                repetitions=3, protected_cells=[(0, 0)],
                rank_region="high", rank_bits=[high_start], count=1,
                target_steps=[0], target_frames=[0], seed=42,
                disturbance_mode="random")
            recovered_bytes = recovered.read_bytes()
        self.assertEqual(recovered_bytes, bytes(data))
        self.assertEqual(metadata["channel_flipped_bits"], 1)
        self.assertEqual(metadata["changed_symbols"], 0)
        self.assertEqual(metadata["corrected_logical_symbols"], 1)
        self.assertEqual(
            metadata["protection_overhead_bits"],
            (rank_bits - high_start) * 2)

    def test_sparse_sign_repetition_corrects_one_physical_error(self):
        data, _ = self._tiny_tdc2()
        with tempfile.TemporaryDirectory() as tmp:
            clean = Path(tmp) / "clean.tdcm"
            protected = Path(tmp) / "clean.tds2"
            disturbed = Path(tmp) / "disturbed.tds2"
            recovered = Path(tmp) / "recovered.tdcm"
            clean.write_bytes(data)
            metadata = attacks.apply_serialized_repetition_disturbance(
                clean, protected, disturbed, recovered,
                repetitions=3, protected_cells=[(0, 0)],
                count=1, seed=42, disturbance_mode="random")
            recovered_bytes = recovered.read_bytes()
        self.assertEqual(recovered_bytes, bytes(data))
        self.assertEqual(metadata["protected_cells"], [[0, 0]])
        self.assertEqual(metadata["changed_symbols"], 0)
        self.assertEqual(metadata["corrected_logical_symbols"], 1)
        self.assertEqual(metadata["protection_overhead_bits"], 10)

    def test_same_budget_slot_policies_use_one_global_channel(self):
        data, _ = self._tiny_tdc2()
        with tempfile.TemporaryDirectory() as tmp:
            clean = Path(tmp) / "clean.tdcm"
            clean.write_bytes(data)
            eep = attacks.select_sign_protection_slots(
                clean, "eep", 2)
            random_slots = attacks.select_sign_protection_slots(
                clean, "random", 2, policy_seed=7)
            self.assertEqual(len(eep), 2)
            self.assertEqual(len(random_slots), 2)

            protected = Path(tmp) / "clean.tds3"
            disturbed = Path(tmp) / "disturbed.tds3"
            recovered = Path(tmp) / "recovered.tdcm"
            metadata = attacks.apply_serialized_slot_repetition_bsc(
                clean, protected, disturbed, recovered,
                protected_slots=eep, repetitions=3, count=1, seed=42,
                policy_name="eep", policy_seed=0)
            parsed = attacks._parse_tdcm_payload_layout(
                bytearray(recovered.read_bytes()))
            serialized_bits = len(protected.read_bytes()) * 8
        self.assertEqual(parsed["n_sde"], 1)
        self.assertEqual(metadata["protected_sign_symbols"], 2)
        self.assertEqual(metadata["protection_overhead_bits"], 4)
        self.assertEqual(metadata["eligible_physical_bits"], 9)
        self.assertEqual(metadata["channel_flipped_bits"], 1)
        self.assertEqual(
            metadata["transmitted_container_bits"],
            serialized_bits)

    def test_frozen_joint_selection_is_cell_bounded(self):
        data, _ = self._tiny_tdc2()
        with tempfile.TemporaryDirectory() as tmp:
            clean = Path(tmp) / "clean.tdcm"
            clean.write_bytes(data)
            chosen = attacks.select_sign_protection_slots(
                clean, "frozen-joint", 3, cells=[(0, 0)])
        self.assertEqual(len(chosen), 3)
        self.assertTrue(all((step, frame) == (0, 0)
                            for step, frame, _ in chosen))


if __name__ == "__main__":
    unittest.main()
