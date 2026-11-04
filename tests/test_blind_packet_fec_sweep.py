import unittest

from exp_flf2v.analyze_blind_packet_fec_sweep import (
    parse_config,
    recovery_rate,
)


class BlindPacketFecSweepTests(unittest.TestCase):
    def test_parse_config(self):
        self.assertEqual(parse_config("16:2"), (16, 2))

    def test_two_parities_recover_any_two_packet_burst(self):
        for seed in range(10):
            self.assertEqual(recovery_rate(32, 16, 2, 2, seed), 1.0)


if __name__ == "__main__":
    unittest.main()
