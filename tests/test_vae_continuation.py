import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from exp_flf2v import run_vae_warmstart_batch as batch
from exp_flf2v.generate_vae_warmstart import validate_extension


class ContinuationTest(unittest.TestCase):
    def test_source_checkpoint_protocol(self):
        keys = ('data_sha256','initial_sha256','vae_path','vae_size','geometry','crop','stride',
                'anchors','alpha_pixel','momentum','seed','schedule_offset','objective','parameterization','boundary')
        config = dict.fromkeys(keys,'same')
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
            np.savez(source/'eps4_optimizer.npz',done=8,fingerprint='old')
            with self.assertRaisesRegex(ValueError,'four-round'):
                validate_extension(source,config)

    def test_four_new_native_runs_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt = root/'ckpt'
            ckpt.mkdir()
            data,initial = root/'data.yuv',root/'initial.npz'
            for f in (data,initial,ckpt/'Wan2.1_VAE.pth'):
                f.write_bytes(b'fake')
            source_assets = root/'old_assets'
            source_assets.mkdir()
            source_eval = root/'old_evaluation'
            source_eval.mkdir()
            script = Path(batch.__file__).with_name('run_flf2v_experiment.py').resolve()
            def write_metrics(path,psnr):
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(dict(per_frame_PSNR_dB=[psnr]*33,LPIPS=.2,
                    input_MSE=.0001,input_Linf_pixel=4.,BPP=.01,reconstruction_vs_submitted_PSNR_dB=psnr)))
            for kind in ('candidate','random'):
                name = 'eps4_r4_'+kind
                f = source_assets/(name+'.npz')
                f.write_bytes(name.encode())
                metrics = root/('old_'+name)/'Jockey/gop0/metrics.json'
                write_metrics(metrics,31. if kind=='candidate' else 33.)
                cmd = batch.codec_command(batch.sys.executable,script,data,ckpt,root,'old_'+name,f,4)
                (source_eval/(name+'.json')).write_text(json.dumps(dict(metrics_sha256=batch.sha256(metrics),
                    configuration=dict(asset=batch.sha256(f),script=batch.sha256(script),command=cmd))))
            argv = ['batch','--data',str(data),'--initial',str(initial),'--wan_ckpt',str(ckpt),
                    '--results_root',str(root),'--extend_tag','old','--tag','new']
            launches = []
            def launch(cmd,check):
                if 'generate_vae_warmstart.py' in cmd[1]:
                    self.assertIn('--extend_from',cmd)
                    assets = root/'new_assets'
                    assets.mkdir(exist_ok=True)
                    rows = [dict(name='initial',kind='initial')]
                    for r in (4,8,12):
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
                self.assertEqual(result['comparisons'][0]['Baseline'],'eps4_r4_candidate')
                self.assertEqual(result['comparisons'][0]['ExtraDropVsInitial'],1.)
                batch.main()
                self.assertEqual(len(launches),4)


if __name__ == '__main__':
    unittest.main()
