import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (Path(__file__).parents[1] / "exp_flf2v" /
               "analyze_combinatorial_index_robustness.py")
SPEC = importlib.util.spec_from_file_location("combination_robustness", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CombinatorialIndexRobustnessTest(unittest.TestCase):
    def test_rank_round_trip(self):
        for K in range(2, 12):
            for M in range(1, K):
                for rank in range(__import__("math").comb(K, M)):
                    indices = MODULE.unrank_combination(rank, K, M)
                    self.assertEqual(MODULE.rank_combination(indices, K), rank)

    def test_single_flip_statistics_are_consistent(self):
        K, M = 32, 5
        indices = [1, 3, 8, 13, 27]
        records = [{"M": M, "rank": MODULE.rank_combination(indices, K),
                    "indices": indices}]
        rows = MODULE.analyze(records, K, bits_per_region=2, seed=7)
        self.assertEqual(len(rows), 6)
        for row in rows:
            if row["valid_rank"]:
                self.assertEqual(row["shared_atoms"] + row["replaced_atoms"], M)
                self.assertGreaterEqual(row["jaccard"], 0.0)
                self.assertLessEqual(row["jaccard"], 1.0)

    def test_csv_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.csv"
            MODULE.write_csv(path, [{"a": 1, "b": 2}])
            self.assertIn("a,b", path.read_text(encoding="utf-8-sig"))

    def test_legacy_tdcm_indices_can_be_analyzed_as_tdc2_subset(self):
        indices = [1, 4, 8]
        signs = [1, -1, 1]
        data = bytearray(b"TDCM")
        data.extend(struct.pack("<10I", 10, 3, 1, 0, 1, 42, 3,
                                9, 16, 24))
        data.extend(struct.pack("<I", 0))
        data.extend(struct.pack("<3I", 2, 3, 4))
        data.extend(struct.pack("<H", len(indices)))
        for index in indices:
            data.extend(struct.pack("<H", index))
        packed_signs = sum((sign > 0) << bit
                           for bit, sign in enumerate(signs))
        data.append(packed_signs)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.tdcm"
            path.write_bytes(data)
            K, M, source_format, records = MODULE.read_codebook_records(path)
        self.assertEqual((K, M, source_format), (10, 3, "TDCM"))
        self.assertEqual(records[0]["indices"], indices)
        self.assertEqual(records[0]["rank"],
                         MODULE.rank_combination(indices, K))


if __name__ == "__main__":
    unittest.main()
