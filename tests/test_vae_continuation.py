import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from exp_flf2v import run_vae_warmstart_batch as batch
from exp_flf2v.generate_vae_warmstart import validate_extension,load_source_base
from exp_flf2v.vae_warmstart_utils import bounded_delta,clipping_audit,temporal_weights,quantized_pair,vae_attack_loss
import torch


class ContinuationTest(unittest.TestCase):
    def test_source_checkpoint_protocol(self):
        keys = ('data_sha256','initial_sha256','vae_path','vae_size','geometry','crop','stride',
                'anchors','alpha_pixel','momentum','seed','schedule_offset','objective','parameterization','boundary')
        config = dict.fromkeys(keys,'same')
        config['saved_rounds'] = [1,4]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            items = []
            for kind in ('candidate','random'):
                f = source/(kind+'.npz')
                f.write_bytes(kind.encode())
                items.append(dict(name='eps4_r4_'+kind,file=f.name,sha256=batch.sha256(f)))
            report = dict(status='completed',configuration=config,fingerprint='old',assets=items)
            (source/'generation.json').write_text(json.dumps(report))
            np.savez(source/'eps4_optimizer.npz',done=4,fingerprint='old')
            self.assertEqual(validate_extension(source,config)['fingerprint'],'old')
            with self.assertRaisesRegex(ValueError,'differs: momentum'):
                validate_extension(source,dict(config,momentum='changed'))
            self.assertEqual(validate_extension(source,dict(config,momentum='changed'),rebase=True)['fingerprint'],'old')
            with self.assertRaisesRegex(ValueError,'differs: data_sha256'):
                validate_extension(source,dict(config,data_sha256='changed'),rebase=True)
            np.savez(source/'eps4_optimizer.npz',done=8,fingerprint='old')
            with self.assertRaisesRegex(ValueError,'completed source round'):
                validate_extension(source,config)

    def test_four_new_native_runs_and_resume(self):
        self._check_native_runs(4,(8,12))

    def test_round12_to20_and28_native_runs_and_resume(self):
        self._check_native_runs(12,(20,28))

    def test_refine28_to32_and36_runs_and_resume(self):
        self._check_native_runs(28,(32,36),refine=True)

    def test_direct_rgb36_to40_and44_runs_and_resume(self):
        self._check_native_runs(36,(40,44),refine=True,direct_rgb=True)

    def test_direct_rgb_objective_changes_gradient_reference(self):
        x = torch.tensor([0.],requires_grad=True)
        clean_vae = torch.tensor([2.],requires_grad=True)
        recon = torch.tensor([1.],requires_grad=True)
        direct = vae_attack_loss(recon,x,clean_vae,direct_rgb=True)
        deviation = vae_attack_loss(recon,x,clean_vae,direct_rgb=False)
        gd, = torch.autograd.grad(direct,recon)
        gv, = torch.autograd.grad(deviation,recon)
        self.assertEqual(float(gd),2.)
        self.assertEqual(float(gv),-2.)
        self.assertIsNone(x.grad)
        self.assertIsNone(clean_vae.grad)

    def test_load_rebased_source_initializer_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            f = folder/'initial.npz'
            saved = np.full((5,2,2,3),4/255,dtype=np.float32)
            np.savez(f,delta=saved)
            report = dict(assets=[dict(name='initial',file=f.name,sha256=batch.sha256(f))])
            np.testing.assert_array_equal(load_source_base(folder,report),saved)
            f.write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'base hash mismatch'):
                load_source_base(folder,report)

    def test_surrogate_forward_bound_and_gradient(self):
        w = temporal_weights(5)
        base = torch.zeros(3,5,2,2)
        c = torch.full((3,3,2,2),6/255,requires_grad=True)
        exact = bounded_delta(base,c,w,4/255)
        surrogate = bounded_delta(base,c,w,4/255,straight_through=True)
        torch.testing.assert_close(exact,surrogate,rtol=0,atol=0)
        g0, = torch.autograd.grad(exact.sum(),c)
        g1, = torch.autograd.grad(surrogate.sum(),c)
        self.assertEqual(float(g0.abs().max()),0.)
        self.assertGreater(float(g1.abs().min()),0.)
        audit = clipping_audit(base,c.detach(),w,4/255)
        self.assertEqual(audit['clamp_zero_derivative_fraction'],1.)
        self.assertAlmostEqual(audit['max_overshoot_pixel'],2.,places=5)

    def test_rebase_preserves_saved_input(self):
        raw = np.full((5,2,2,3),128,dtype=np.uint8)
        base = torch.zeros(3,5,2,2)
        c = torch.full((3,3,2,2),6/255)
        w = temporal_weights(5)
        old = bounded_delta(base,c,w,4/255).permute(1,2,3,0).numpy()
        saved,random = quantized_pair(raw,old,42)
        rebased = torch.from_numpy(saved).permute(3,0,1,2)
        new = bounded_delta(rebased,torch.zeros_like(c),w,4/255).permute(1,2,3,0).numpy()
        actual,reference = quantized_pair(raw,new,42)
        np.testing.assert_array_equal(actual,saved)
        np.testing.assert_array_equal(reference,random)

    def _check_native_runs(self,source_round,target_rounds,refine=False,direct_rgb=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt = root/'ckpt'
            ckpt.mkdir()
            data,initial = root/'data.yuv',root/'initial.npz'
            for f in (data,initial,ckpt/'Wan2.1_VAE.pth'):
                f.write_bytes(b'fake')
            source_assets = root/'old_assets'
            source_assets.mkdir()
            (source_assets/'generation.json').write_text(json.dumps(dict(status='completed',
                configuration=dict(saved_rounds=[source_round]))))
            source_eval = root/'old_evaluation'
            source_eval.mkdir()
            script = Path(batch.__file__).with_name('run_flf2v_experiment.py').resolve()
            def write_metrics(path,psnr):
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(dict(per_frame_PSNR_dB=[psnr]*33,LPIPS=.2,
                    input_MSE=.0001,input_Linf_pixel=4.,BPP=.01,reconstruction_vs_submitted_PSNR_dB=psnr)))
            for kind in ('candidate','random'):
                name = f'eps4_r{source_round}_'+kind
                f = source_assets/(name+'.npz')
                f.write_bytes(name.encode())
                metrics = root/('old_'+name)/'Jockey/gop0/metrics.json'
                write_metrics(metrics,31. if kind=='candidate' else 33.)
                cmd = batch.codec_command(batch.sys.executable,script,data,ckpt,root,'old_'+name,f,4)
                (source_eval/(name+'.json')).write_text(json.dumps(dict(metrics_sha256=batch.sha256(metrics),
                    configuration=dict(asset=batch.sha256(f),script=batch.sha256(script),command=cmd))))
            argv = ['batch','--data',str(data),'--initial',str(initial),'--wan_ckpt',str(ckpt),
                    '--results_root',str(root),'--extend_tag','old','--tag','new',
                    '--extension_rounds',*map(str,target_rounds)]
            if refine:
                argv += ['--refine']
            if direct_rgb:
                argv += ['--direct_rgb']
            launches = []
            def launch(cmd,check):
                if 'generate_vae_warmstart.py' in cmd[1]:
                    self.assertIn('--extend_from',cmd)
                    self.assertEqual('--refine' in cmd,refine)
                    self.assertEqual('--direct_rgb' in cmd,direct_rgb)
                    assets = root/'new_assets'
                    assets.mkdir(exist_ok=True)
                    rows = [dict(name='initial',kind='initial')]
                    for r in (source_round,*target_rounds):
                        for kind in ('candidate','random'):
                            name = f'eps4_r{r}_{kind}'
                            path = assets/(name+'.npz')
                            path.write_bytes(name.encode())
                            rows.append(dict(name=name,kind=kind,round=r,epsilon=4,file=path.name,
                                             sha256=batch.sha256(path),input_MSE=.0001))
                    (assets/'generation.json').write_text(json.dumps(dict(status='completed',fingerprint='new',assets=rows)))
                else:
                    self.assertNotIn('--reuse_codebook',cmd)
                    run = cmd[cmd.index('--run_name')+1]
                    launches.append(run)
                    write_metrics(root/run/'Jockey/gop0/metrics.json',30. if run.endswith('candidate') else 32.)
            with patch('sys.argv',argv),patch.object(batch.subprocess,'run',side_effect=launch), \
                 contextlib.redirect_stdout(io.StringIO()):
                batch.main()
                self.assertEqual(len(launches),4)
                result = json.loads((root/'new_evaluation/completed.json').read_text())
                self.assertEqual(result['comparisons'][0]['Baseline'],f'eps4_r{source_round}_candidate')
                self.assertEqual(result['comparisons'][0]['ExtraDropVsInitial'],1.)
                batch.main()
                self.assertEqual(len(launches),4)


if __name__ == '__main__':
    unittest.main()
