import unittest
import struct

import numpy as np

from exp_flf2v.analyze_sign_packet_erasure import (
    parse_tdcm_layout,
    physical_positions,
    simulate_erasure,
)


class SignPacketErasureTests(unittest.TestCase):
    @staticmethod
    def minimal_tdcm():
        raw = bytearray(b"TDCM")
        raw.extend(struct.pack(
            "<10I", 16, 2, 1, 0, 1, 42, 1, 1, 8, 8))
        raw.extend(struct.pack("<I", 0))
        raw.extend(struct.pack("<I", 1))
        raw.extend(struct.pack("<H", 2))
        raw.extend(struct.pack("<2H", 3, 7))
        raw.append(0b00000001)
        return raw

    def test_torch_free_tdcm_parser_locates_signs(self):
        layout = parse_tdcm_layout(self.minimal_tdcm())
        self.assertEqual(layout["index_coding"], "fixed-width")
        self.assertEqual(layout["n_sde"], 1)
        self.assertEqual(layout["n_lat"], 1)
        self.assertEqual([slot["index"] for slot in layout["slots"]], [3, 7])
        self.assertEqual([slot["sign_mask"] for slot in layout["slots"]], [1, 2])

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
