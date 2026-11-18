import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from exp_flf2v import validate_vae_warmstart_seeds as batch


class SeedValidationTest(unittest.TestCase):
    def test_clean_and_attacked_commands(self):
        for seed in (123,2026):
            for asset in (None,Path('fixed.npz')):
                cmd = batch.validation_command('codec','data','ckpt','root','run',asset,seed)
                self.assertEqual(cmd[cmd.index('--seed')+1],str(seed))
                self.assertNotIn('--reuse_codebook',cmd)
                self.assertEqual(cmd[cmd.index('--attack')+1], 'none' if asset is None else 'vae-pgd')
                self.assertEqual('--attack_file' in cmd, asset is not None)

    def test_complete_batch_resume_and_hash_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt = root/'ckpt'
            ckpt.mkdir()
            (ckpt/'Wan2.1_VAE.pth').write_bytes(b'fake')
            data = root/'data.yuv'
            data.write_bytes(b'source')
            assets = root/'source_assets'
            assets.mkdir()
            items = []
            for kind in ('candidate','random'):
                path = assets/(kind+'.npz')
                path.write_bytes(kind.encode())
                items.append(dict(name='eps4_r4_'+kind,file=path.name,epsilon=4,round=4,
                                  input_MSE=.0001,input_Linf_pixel=4.,sha256=batch.sha256(path)))
            (assets/'generation.json').write_text(json.dumps(dict(status='completed',
                fingerprint='fixed',configuration=dict(data_sha256=batch.sha256(data)),assets=items)))
            argv = ['batch','--data',str(data),'--wan_ckpt',str(ckpt),'--results_root',str(root),
                    '--source_tag','source','--tag','heldout']
            launches = []
            def launch(cmd,check):
                self.assertTrue(check)
                self.assertIn('run_flf2v_experiment.py',cmd[1])
                run = cmd[cmd.index('--run_name')+1]
                kind = run.rsplit('_',1)[1]
                launches.append(run)
                metrics = root/run/'Jockey/gop0/metrics.json'
                metrics.parent.mkdir(parents=True,exist_ok=True)
                psnr = dict(clean=33.,candidate=30.,random=32.)[kind]
                lpips = dict(clean=.08,candidate=.2,random=.1)[kind]
                metrics.write_text(json.dumps(dict(per_frame_PSNR_dB=[psnr]*33,LPIPS=lpips,
                    input_MSE=0. if kind=='clean' else .0001,
                    input_Linf_pixel=0. if kind=='clean' else 4., BPP=.01,
                    reconstruction_vs_submitted_PSNR_dB=psnr)))
            with patch('sys.argv',argv),patch.object(batch.subprocess,'run',side_effect=launch), \
                 contextlib.redirect_stdout(io.StringIO()):
                batch.main()
                self.assertEqual(len(launches),6)
                result = json.loads((root/'heldout_evaluation/completed.json').read_text())
                self.assertEqual(result['paired'][0]['CandidateDropVsClean'],3.)
                self.assertEqual(result['paired'][0]['ExtraDropVsRandom'],2.)
                self.assertAlmostEqual(result['paired'][0]['ExtraLPIPSVsRandom'],.1)
                batch.main()
                self.assertEqual(len(launches),6)
                metrics = root/'heldout_seed123_clean/Jockey/gop0/metrics.json'
                metrics.write_text('corrupted')
                batch.main()
                self.assertEqual(len(launches),7)
                (assets/'candidate.npz').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'modified source asset'):
                    batch.main()
                self.assertEqual(len(launches),7)

    def test_missing_pair_rejected(self):
        with self.assertRaises(ValueError):
            batch.paired_results([])


if __name__ == '__main__':
    unittest.main()
