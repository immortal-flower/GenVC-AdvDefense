import unittest

from exp_flf2v.analyze_blind_stream_packet_erasure import analyze_loss


class BlindStreamPacketErasureTests(unittest.TestCase):
    def setUp(self):
        self.info = {"prefix_bytes": 4, "payload_end": 12}
        self.copy_bytes = [[6, 12, 14], [7, 13, 15]]
        self.recoverable = {2, 3}

    def test_header_loss_is_structural(self):
        result = analyze_loss(
            self.info, self.copy_bytes, self.recoverable,
            packet_bytes=4, start_packet=0, lost_packets=1,
            stream_bytes=16)
        self.assertFalse(result["syntax_recoverable"])

    def test_protected_sign_bytes_can_be_recovered(self):
        result = analyze_loss(
            self.info, self.copy_bytes, self.recoverable,
            packet_bytes=2, start_packet=3, lost_packets=1,
            stream_bytes=16)
        self.assertTrue(result["syntax_recoverable"])
        self.assertTrue(result["exact_recovery"])

    def test_losing_all_copies_leaves_unresolved_sign(self):
        result = analyze_loss(
            self.info, [[6, 7, 8]], {2, 3, 4}, packet_bytes=3,
            start_packet=2, lost_packets=1, stream_bytes=12)
        self.assertTrue(result["syntax_recoverable"])
        self.assertEqual(result["unresolved_signs"], 1)


if __name__ == "__main__":
    unittest.main()
