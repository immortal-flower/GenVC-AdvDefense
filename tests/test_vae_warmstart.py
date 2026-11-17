import unittest
import json
import tempfile
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from exp_flf2v.vae_warmstart_utils import bounded_delta, crop_starts, quantized_pair, temporal_weights
from exp_flf2v.run_vae_warmstart_batch import codec_command, compare
from exp_flf2v import run_vae_warmstart_batch as batch


class WarmstartTest(unittest.TestCase):
    def test_temporal_weights(self):
        w = temporal_weights(33)
        torch.testing.assert_close(w.sum(1), torch.ones(33))
        torch.testing.assert_close(w[[0,16,32]], torch.eye(3))

    def test_gradient_and_bounds(self):
        c = torch.zeros(3,3,2,2, requires_grad=True)
        base = torch.ones(3,5,2,2)/255
        d = bounded_delta(base,c,temporal_weights(5),2/255)
        torch.testing.assert_close(d,base)
        d.square().sum().backward()
        self.assertTrue(torch.all(c.grad.abs().sum((1,2,3)) > 0))
        d = bounded_delta(base,torch.ones_like(c),temporal_weights(5),4/255)
        self.assertLessEqual(float(d.max()),4/255+1e-8)

    def test_realized_pair_with_saturation(self):
        rng = np.random.default_rng(42)
        x = rng.integers(0,256,(5,8,8,3),dtype=np.uint8)
        x[0] = 0
        x[-1] = 255
        d = rng.uniform(-4/255,4/255,x.shape).astype(np.float32)
        c,r = quantized_pair(x,d,42)
        np.testing.assert_array_equal(np.abs(c),np.abs(r))
        for delta in (c,r):
            pixels = x.astype(np.float32)+delta*255
            self.assertGreaterEqual(float(pixels.min()),0)
            self.assertLessEqual(float(pixels.max()),255)

    def test_crop_coverage(self):
        for n in (720,1280):
            coverage = np.zeros(n)
            for i in crop_starts(n):
                coverage[i:i+192] += 1
            self.assertTrue(np.all(coverage>0))

    def test_fresh_codec_command(self):
        cmd = codec_command('python','codec.py',Path('data'),Path('ckpt'),Path('root'),'run',Path('attack.npz'),4)
        self.assertNotIn('--reuse_codebook',cmd)
        self.assertIn('--input_attack_diagnostics',cmd)
        self.assertEqual(cmd[cmd.index('--height')+1],'720')

    def test_pair_signs_and_energy_guard(self):
        rows = [dict(name='initial',PSNR=33.,LPIPS=.1)]
        for eps in (2,4):
            for rounds in (1,4):
                for kind,psnr,lpips in [('candidate',32.,.2),('random',32.5,.15)]:
                    rows.append(dict(name=f'eps{eps}_r{rounds}_{kind}',PSNR=psnr,LPIPS=lpips,
                                     InputMSE=.0001,InputLinfPixel=eps,BPP=.01))
        paired = compare(rows)
        self.assertEqual(paired[0]['ExtraDropVsInitial'],1.)
        self.assertEqual(paired[0]['ExtraDropVsRandom'],.5)
        self.assertAlmostEqual(paired[0]['ExtraLPIPSVsRandom'],.05)
        rows[-1]['InputMSE'] = .001
        with self.assertRaises(ValueError):
            compare(rows)

    def test_batch_resume_and_corrupt_metric_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt = root/'ckpt'
            ckpt.mkdir()
            for path in (root/'source.yuv', root/'initial.npz', ckpt/'Wan2.1_VAE.pth'):
                path.write_bytes(b'test')
            argv = ['batch', '--data', str(root/'source.yuv'), '--initial', str(root/'initial.npz'),
                    '--wan_ckpt', str(ckpt), '--results_root', str(root), '--tag', 'test']
            launches = []
            def launch(cmd, check):
                self.assertTrue(check)
                if 'generate_vae_warmstart.py' in cmd[1]:
                    assets = root/'test_assets'
                    assets.mkdir(exist_ok=True)
                    items = []
                    names = [('initial',2)] + [(f'eps{e}_r{r}_{k}',e)
                             for e in (2,4) for r in (1,4) for k in ('candidate','random')]
                    for name,eps in names:
                        path = assets/(name+'.npz')
                        path.write_bytes(name.encode())
                        items.append(dict(name=name,file=path.name,epsilon=eps,
                                          sha256=batch.sha256(path),input_MSE=.0001))
                    (assets/'generation.json').write_text(json.dumps(dict(status='completed',
                        fingerprint='fixed',assets=items)), encoding='utf-8')
                else:
                    self.assertNotIn('--reuse_codebook',cmd)
                    name = cmd[cmd.index('--run_name')+1]
                    launches.append(name)
                    metrics = root/name/'Jockey/gop0/metrics.json'
                    metrics.parent.mkdir(parents=True,exist_ok=True)
                    metrics.write_text('{}',encoding='utf-8')
            values = dict(PSNR=32.,LPIPS=.1,InputMSE=.0001,InputLinfPixel=2.,BPP=.01,PSNRvsSubmitted=32.)
            with patch('sys.argv',argv), patch.object(batch.subprocess,'run',side_effect=launch), \
                 patch.object(batch,'read_metrics',return_value=values):
                batch.main()
                self.assertEqual(len(launches),9)
                batch.main()
                self.assertEqual(len(launches),9)
                (root/'test_initial/Jockey/gop0/metrics.json').write_text('changed')
                batch.main()
                self.assertEqual(len(launches),10)


if __name__ == '__main__':
    unittest.main()
