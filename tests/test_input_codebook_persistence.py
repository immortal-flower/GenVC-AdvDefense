import unittest

from exp_flf2v.analyze_input_codebook_persistence import compare_cell


class CodebookPersistenceTests(unittest.TestCase):
    def test_replacement_and_shared_sign_change_are_separate(self):
        clean = {'M': 3, 'indices': [1, 2, 3], 'signs': [1, -1, 1]}
        changed = {'M': 3, 'indices': [1, 2, 4], 'signs': [-1, -1, 1]}
        result = compare_cell(clean, changed)
        self.assertEqual(result['replaced_atoms'], 1)
        self.assertEqual(result['shared_sign_changes'], 1)
        self.assertAlmostEqual(result['signed_change_fraction'], 2 / 3)

    def test_rejects_different_number_of_transmitted_atoms(self):
        with self.assertRaises(ValueError):
            compare_cell({'M': 1}, {'M': 2})


if __name__ == '__main__':
    unittest.main()
