import importlib.util
import math
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path

# Load the bitstream module directly so these CPU-only format tests do not
# require the full Wan/PyTorch runtime.
try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    sys.modules["torch"] = types.SimpleNamespace(
        Tensor=object, Generator=object, device=lambda value: value,
    )

MODULE_PATH = Path(__file__).parents[1] / "sde_rf_wan" / "turbo_codebook.py"
SPEC = importlib.util.spec_from_file_location("turbo_codebook_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
TurboBitstream = MODULE.TurboBitstream
combinatorial_index_bits = MODULE.combinatorial_index_bits
rank_combination = MODULE.rank_combination
unrank_combination = MODULE.unrank_combination


class CombinatorialCodingTest(unittest.TestCase):
    def test_rank_round_trip(self):
        for K in range(1, 14):
            for M in range(K + 1):
                total = math.comb(K, M)
                for rank in range(total):
                    indices = unrank_combination(rank, K, M)
                    self.assertEqual(rank_combination(indices, K), rank)

    def test_meeting_configuration_uses_664_bits(self):
        self.assertEqual(combinatorial_index_bits(16384, 64), 600)
        new_bits = combinatorial_index_bits(16384, 64) + 64
        old_bits = 64 * (math.ceil(math.log2(16384)) + 1)
        self.assertEqual(new_bits, 664)
        self.assertEqual(old_bits, 960)
        self.assertAlmostEqual(1 - new_bits / old_bits, 0.30833333333333335)

    def test_tdc2_round_trip_with_variable_m(self):
        step_data = [
            [([0, 3, 7], [1, -1, 1]), ([1, 2, 6, 9], [-1, 1, -1, 1])],
            [([2, 4, 8], [-1, -1, 1]), ([0, 5, 7, 9], [1, 1, -1, -1])],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "roundtrip.tdcm"
            TurboBitstream.save(
                str(path), step_data, K=10, M=3,
                num_sde_steps=2, num_ddim_steps=1,
                num_latent_frames=2, seed=42,
                frame_shape=(2, 3, 4), prompt="test",
                num_frames=9, height=16, width=24,
            )
            self.assertEqual(path.read_bytes()[:4], b"TDC2")
            loaded = TurboBitstream.load(str(path))
        self.assertEqual(loaded["step_data"], step_data)
        self.assertEqual(loaded["index_coding"], "combinatorial")

    def test_legacy_tdcm_remains_readable(self):
        step_data = [[([1, 4, 8], [1, -1, 1])]]
        data = bytearray(b"TDCM")
        data.extend(struct.pack("<10I", 10, 3, 1, 0, 1, 42, 3, 9, 16, 24))
        data.extend(struct.pack("<I", 0))
        data.extend(struct.pack("<3I", 2, 3, 4))
        data.extend(struct.pack("<H", 3))
        for index in step_data[0][0][0]:
            data.extend(struct.pack("<H", index))
        data.extend(bytes([0b101]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.tdcm"
            path.write_bytes(data)
            loaded = TurboBitstream.load(str(path))
        self.assertEqual(loaded["step_data"], step_data)
        self.assertEqual(loaded["index_coding"], "fixed-width")


if __name__ == "__main__":
    unittest.main()
