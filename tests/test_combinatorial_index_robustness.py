import importlib.util
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


if __name__ == "__main__":
    unittest.main()
