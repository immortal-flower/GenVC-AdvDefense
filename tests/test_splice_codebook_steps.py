import struct
import unittest

from exp_flf2v.splice_codebook_steps import make_hybrids, split_stream


def stream(values, magic=b"TDC2"):
    header = magic + struct.pack("<10I", 16, 1, 3, 0, 1, 42, 1, 33, 720, 1280)
    header += struct.pack("<I", 0) + struct.pack("<I", 16)
    records = []
    for value in values:
        records.append(struct.pack("<H", 1) + bytes([value, 1]))
    return header + b"".join(records)


class SpliceCodebookStepsTest(unittest.TestCase):
    def test_early_and_late_swap(self):
        clean = stream([1, 2, 3])
        attack = stream([4, 5, 6])
        early, late = make_hybrids(clean, attack, 2)
        self.assertEqual(split_stream(early)[1], split_stream(stream([4, 5, 3]))[1])
        self.assertEqual(split_stream(late)[1], split_stream(stream([1, 2, 6]))[1])

    def test_rejects_mismatch_and_truncation(self):
        with self.assertRaises(ValueError):
            make_hybrids(stream([1, 2, 3]), stream([4, 5, 6], magic=b"TDCM"), 2)
        with self.assertRaises(ValueError):
            split_stream(stream([1, 2, 3])[:-1])


if __name__ == "__main__":
    unittest.main()
