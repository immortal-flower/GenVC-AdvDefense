import csv
import tempfile
import unittest
from pathlib import Path

from exp_flf2v.codebook_validation_plan import expand_jobs, load_cells


class ValidationPlanTests(unittest.TestCase):
    def temporary_directory(self):
        return tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)

    def test_load_and_expand(self):
        with self.temporary_directory() as directory:
            path = Path(directory) / "plan.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["role", "sde_step", "latent_position"])
                writer.writeheader()
                writer.writerow({"role": "hot", "sde_step": 1, "latent_position": 3})
                writer.writerow({"role": "control", "sde_step": 9, "latent_position": 3})
            cells = load_cells(path)
            jobs = expand_jobs(cells, [16], [42, 123, 2026])
        self.assertEqual(len(jobs), 6)
        self.assertEqual(jobs[0].run_name("Jockey"),
                         "sensitivity_grid_sign16_step1_frame3_aseed42_jockey")
        self.assertEqual(jobs[-1].attack_seed, 2026)

    def test_duplicate_cells_are_rejected(self):
        with self.temporary_directory() as directory:
            path = Path(directory) / "plan.csv"
            path.write_text(
                "role,sde_step,latent_position\na,1,3\nb,1,3\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_cells(path)

    def test_invalid_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Bit count"):
            expand_jobs([("hot", 1, 3)], [65], [42])


if __name__ == "__main__":
    unittest.main()
