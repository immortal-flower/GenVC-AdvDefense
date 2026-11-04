import unittest

import numpy as np

from exp_flf2v.packet_cauchy_rs import encode_group, recover_group


class PacketCauchyRsTests(unittest.TestCase):
    def test_recovers_three_missing_packets(self):
        rng = np.random.default_rng(42)
        data = [rng.integers(0, 256, 64, dtype=np.uint8) for _ in range(10)]
        parity = encode_group(data, 3)
        damaged = list(data)
        for index in (1, 4, 8):
            damaged[index] = None
        recovered = recover_group(damaged, parity)
        for expected, actual in zip(data, recovered):
            np.testing.assert_array_equal(expected, actual)

    def test_fails_when_too_few_parities_survive(self):
        data = [np.full(16, value, dtype=np.uint8) for value in range(4)]
        parity = encode_group(data, 2)
        damaged = [None, None, data[2], data[3]]
        with self.assertRaises(ValueError):
            recover_group(damaged, [parity[0], None])


if __name__ == "__main__":
    unittest.main()
