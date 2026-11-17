import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from exp_flf2v.search_native_rgb_input import (
    choose_best, main, project_energy, read_metrics, realize, sse,
)


class NativeQuerySearchTest(unittest.TestCase):
    def test_energy_projection_preserves_boundary_bound_and_mse(self):
        rng = np.random.default_rng(12)
        raw = rng.integers(0, 256, (5, 32, 32, 3), dtype=np.uint8)
        initial = realize(raw, rng.choice([-2., 2.], size=raw.shape), 1., 4,
                          np.zeros((2, 32, 32, 3), dtype=np.int8))
        vector = initial.astype(np.float32) + rng.standard_normal(raw.shape).astype(np.float32)
        actual = project_energy(raw, vector, initial, 4)
        self.assertLessEqual(sse(actual), sse(initial))
        self.assertGreaterEqual(sse(actual), .99*sse(initial))
        np.testing.assert_array_equal(actual[[0,-1]], initial[[0,-1]])
        self.assertLessEqual(np.abs(actual.astype(np.int16)).max(), 4)
        pixels = raw.astype(np.int16)+actual
        self.assertGreaterEqual(pixels.min(), 0)
        self.assertLessEqual(pixels.max(), 255)

    def test_rejects_impossible_energy_match(self):
        raw = np.full((5, 4, 4, 3), 128, np.uint8)
        initial = np.full(raw.shape, 2, np.int8)
        with self.assertRaises(ValueError):
            project_energy(raw, np.zeros(raw.shape, np.float32), initial, 4)

    def test_selects_psnr_only_and_preserves_parent_on_tie(self):
        parent = dict(PSNR=32.5, LPIPS=.1)
        self.assertIs(choose_best(parent, [dict(PSNR=32.51), dict(PSNR=32.4999)]), parent)
        candidate = dict(PSNR=32.4, LPIPS=.05)
        self.assertIs(choose_best(parent, [candidate]), candidate)

    def test_precise_metric_uses_frame_average_not_rounded_display(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'m.json'
            data = dict(per_frame_PSNR_dB=[32.5032]*33, PSNR_dB=32.5, LPIPS=.1,
                        input_MSE=.00004, input_Linf_pixel=4., BPP=.01,
                        reconstruction_vs_submitted_PSNR_dB=32.2)
            path.write_text(json.dumps(data), encoding='utf-8')
            self.assertAlmostEqual(read_metrics(path)['PSNR'], 32.5032)

    def test_mocked_end_to_end_search_resumes_without_codec_relaunch(self):
        # Native oracle is mocked; this test never imports torch or uses a GPU.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root/'Jockey_720p.yuv'
            data.write_bytes(b'fixture')
            ckpt = root/'weights'
            ckpt.mkdir()
            (ckpt/'Wan2.1_VAE.pth').write_bytes(b'fixture')
            raw = np.full((5, 32, 32, 3), 128, np.uint8)
            init = root/'init.npz'
            np.savez_compressed(init, delta=np.full(raw.shape, 2/255, np.float32))
            argv = ['search', '--data', str(data), '--wan_ckpt', str(ckpt),
                    '--initial_attack', str(init), '--results_root', str(root), '--rounds', '1']
            calls = []

            class FakeProcess:
                def __init__(self, cmd, **kwargs):
                    calls.append(cmd)
                    self.stdout = io.StringIO('mock native completed\n')
                    values = {cmd[i]:cmd[i+1] for i in range(2,len(cmd)-1) if cmd[i].startswith('--')}
                    self_name = values['--run_name']
                    with np.load(values['--attack_file'], allow_pickle=False) as saved:
                        delta = saved['delta']
                    score = 32. if ('r1_plus' in self_name or 'holdout_best' in self_name) else 33.
                    result = dict(per_frame_PSNR_dB=[score]*33, LPIPS=.1,
                                  input_MSE=float(np.mean(delta**2)), input_Linf_pixel=float(np.abs(delta).max()*255),
                                  BPP=.01, reconstruction_vs_submitted_PSNR_dB=score-.1)
                    metric = Path(values['--output_dir'])/self_name/'Jockey/gop0/metrics.json'
                    metric.parent.mkdir(parents=True, exist_ok=True)
                    metric.write_text(json.dumps(result), encoding='utf-8')
                def wait(self):
                    return 0

            def direction(shape, seed, grid):
                return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)

            with patch('sys.argv', argv), patch('exp_flf2v.search_native_rgb_input.load_source', return_value=raw), \
                    patch('exp_flf2v.search_native_rgb_input.smooth_direction', side_effect=direction), \
                    patch('exp_flf2v.search_native_rgb_input.subprocess.Popen', FakeProcess), \
                    contextlib.redirect_stdout(io.StringIO()):
                main()
                count = len(calls)
                main()
            self.assertEqual(len(calls), count)
            self.assertEqual(count, 5)  # initial + pair + two held-out inputs
            for cmd in calls:
                self.assertNotIn('--reuse_codebook', cmd)
                self.assertEqual(cmd[cmd.index('--height')+1], '720')
            state = json.loads((root/'native_query_jockey_gop0_energy_matched_v1_search/search.json').read_text())
            self.assertEqual(state['status'], 'completed')
            self.assertFalse(state['heldout']['used_for_selection'])


if __name__ == '__main__':
    unittest.main()
