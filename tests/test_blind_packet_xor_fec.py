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


if __name__ == "__main__":
    unittest.main()
