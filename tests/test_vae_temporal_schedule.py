import unittest
from exp_flf2v.vae_temporal_schedule import crop_start


class TemporalScheduleTest(unittest.TestCase):
    def test_original_schedule(self):
        for r in range(44):
            for tile in range(72):
                self.assertEqual(crop_start(tile,r,5),(tile*7+(8+r)*5)%29)

    def test_windows_cover_start_range(self):
        for window in (5,9,17):
            starts = {crop_start(t,r,window) for t in range(72) for r in range(20)}
            self.assertEqual(starts,set(range(34-window)))
            self.assertTrue(all(0<=t and t+window<=33 for t in starts))

    def test_invalid(self):
        for args in ((0,0,6),(-1,0,9),(0,-1,17)):
            with self.assertRaises(ValueError):
                crop_start(*args)
