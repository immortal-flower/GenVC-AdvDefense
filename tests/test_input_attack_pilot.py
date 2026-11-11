import unittest
from exp_flf2v.analyze_input_attack_pilot import compare_records


class InputAttackPilotTests(unittest.TestCase):
    def stream(self, indices, signs):
        return dict(K=8, num_sde_steps=1, num_latent_frames=1,
                    step_data=[[(indices, signs)]])

    def test_unchanged(self):
        stream = self.stream([1, 2], [1, -1])
        row = compare_records(stream, stream)[0]
        self.assertEqual(row['replaced_atoms'], 0)
        self.assertEqual(row['index_jaccard'], 1)
        self.assertEqual(row['shared_atoms_sign_flipped'], 0)

    def test_signs_compared_by_atom_identity(self):
        row = compare_records(self.stream([1, 2], [1, -1]),
                              self.stream([2, 3], [1, 1]))[0]
        self.assertEqual(row['replaced_atoms'], 1)
        self.assertAlmostEqual(row['index_jaccard'], 1/3)
        self.assertEqual(row['shared_atoms_sign_flipped'], 1)

    def test_incompatible(self):
        left = self.stream([1], [1])
        right = self.stream([1], [1])
        right['K'] = 16
        with self.assertRaises(ValueError):
            compare_records(left, right)


if __name__ == '__main__':
    unittest.main()
