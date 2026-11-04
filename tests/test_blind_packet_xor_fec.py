import unittest

from exp_flf2v.analyze_blind_packet_xor_fec import (
    exact_recovery,
    group_assignment,
)


class BlindPacketXorFecTests(unittest.TestCase):
    def test_single_erasure_is_always_recoverable(self):
        assignment, groups = group_assignment(8, 4, "contiguous-groups")
        for packet in range(8 + groups):
            self.assertTrue(exact_recovery(
                8, assignment, groups, [packet]))

    def test_two_data_losses_in_one_group_fail(self):
        assignment, groups = group_assignment(8, 4, "contiguous-groups")
        self.assertFalse(exact_recovery(8, assignment, groups, [0, 1]))

    def test_shuffle_assignment_is_deterministic(self):
        first, _ = group_assignment(20, 4, "shared-shuffle", 42)
        second, _ = group_assignment(20, 4, "shared-shuffle", 42)
        self.assertEqual(first.tolist(), second.tolist())

    def test_two_parities_recover_two_data_losses(self):
        assignment, groups = group_assignment(8, 4, "contiguous-groups")
        self.assertTrue(exact_recovery(
            8, assignment, groups, [0, 1], parity_per_group=2))

    def test_data_and_parity_loss_can_exceed_mds_budget(self):
        assignment, groups = group_assignment(8, 4, "contiguous-groups")
        first_group_parity = 8
        self.assertFalse(exact_recovery(
            8, assignment, groups, [0, 1, first_group_parity],
            parity_per_group=2))


if __name__ == "__main__":
    unittest.main()
