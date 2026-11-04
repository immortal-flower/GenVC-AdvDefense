import unittest

import numpy as np

from exp_flf2v.analyze_sign_packet_erasure import (
    physical_positions,
    simulate_erasure,
)


class SignPacketErasureTests(unittest.TestCase):
    def test_sequential_single_packet_can_remove_complete_replicas(self):
        positions = physical_positions(4, 3, "sequential")
        result = simulate_erasure(
            positions, np.ones(4, dtype=np.uint8), packet_bits=6,
            loss_packets=1, start_packet=0)
        self.assertEqual(result["unresolved_symbols"], 2)
        self.assertEqual(result["changed_symbols_zero_fill"], 2)

    def test_replica_stripe_survives_one_packet(self):
        positions = physical_positions(4, 3, "replica-stripe")
        result = simulate_erasure(
            positions, np.ones(4, dtype=np.uint8), packet_bits=4,
            loss_packets=1, start_packet=0)
        self.assertEqual(result["unresolved_symbols"], 0)

    def test_shared_shuffle_is_deterministic(self):
        first = physical_positions(20, 3, "shared-shuffle", seed=42)
        second = physical_positions(20, 3, "shared-shuffle", seed=42)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
