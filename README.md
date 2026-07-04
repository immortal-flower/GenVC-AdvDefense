# GenVC-AdvDefense

Generative video compression adversarial attack and defense experiments built on GVCC.

This repository extends **GVCC: Zero-Shot Video Compression via Codebook-Driven Stochastic Rectified Flow** with an experimental framework for robustness evaluation. The current focus is FLF2V on UVG, including clean compression baselines, image-space adversarial attacks, preprocessing defenses, and combined attack-defense evaluation.

## Adversarial / Defense Experiments

The FLF2V experiment entry point is:

```bash
python exp_flf2v/run_flf2v_experiment.py
```

Useful options:

```bash
--attack none/uap/ftuap/pgd-d/pgd-r/pgd-rd/keyframe/gop-shared
--defense none/jpeg/median/jpeg-median
--epsilon 4
--attack_steps 8
--attack_alpha 0
--run_name uap_eps4
```

Examples:

```bash
python exp_flf2v/run_flf2v_experiment.py --attack none --defense none --run_name clean
python exp_flf2v/run_flf2v_experiment.py --attack uap --epsilon 4 --run_name uap_eps4
python exp_flf2v/run_flf2v_experiment.py --attack ftuap --epsilon 4 --run_name ftuap_eps4
python exp_flf2v/run_flf2v_experiment.py --attack uap --defense jpeg --epsilon 4 --jpeg_quality 85 --run_name uap_jpeg_eps4
```

Attack and defense utilities are organized under:

```text
exp_flf2v/attacks.py
exp_flf2v/defenses.py
```

Note: `pgd-d`, `pgd-r`, and `pgd-rd` are currently image-space proxy attacks. Full white-box GVCC-gradient PGD requires enabling gradients conditionally and defining a differentiable attack objective.

# GVCC

Official code release for **[GVCC: Zero-Shot Video Compression via Codebook-Driven Stochastic Rectified Flow](https://arxiv.org/abs/2603.26571)** (arXiv:2603.26571).

A short qualitative example is included under [`demo/`](demo/).
Algorithmic details of the encode/decode pipeline are in [`PIPELINE.md`](PIPELINE.md).

## Install

Tested with Python 3.10 and CUDA 13.0; the dependency lower bounds match upstream Wan2.1, so any environment that runs Wan2.1 should run GVCC.

```bash
# 1. PyTorch (CUDA 13.0 wheel - adjust the index URL for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 2. Remaining dependencies
pip install -r requirements.txt
```

`flash_attn` is optional. `wan/modules/attention.py` falls back to PyTorch SDPA if it is not installed.

## Model Weights

Download the three Wan2.1-14B checkpoints from HuggingFace and place each one **alongside its corresponding `run_*.sh`**. The run scripts locate the model via `${SCRIPT_DIR}/Wan2.1-*`.

| Method | HuggingFace repo | Required local path |
| ------ | ---------------- | ------------------- |
| T2V | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | `exp_t2v/Wan2.1-T2V-14B-Diffusers/` |
| I2V | `Wan-AI/Wan2.1-I2V-14B-720P` | `exp_i2v/Wan2.1-I2V-14B-720P/` |
| FLF2V | `Wan-AI/Wan2.1-FLF2V-14B-720P` | `exp_flf2v/Wan2.1-FLF2V-14B-720P/` |

Convenience download scripts:

```bash
bash exp_t2v/download_t2v_14b.sh
bash exp_i2v/download_i2v_14b.sh
bash exp_flf2v/download_flf2v_14b.sh
```

Each path may be a real directory or a symlink to a shared cache.

**Smaller backbones.** GVCC is backbone-agnostic. Any Wan2.1 variant works as a drop-in replacement. For low-VRAM experimentation with the T2V configuration, use `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`.

## Data

Download the seven [UVG-1080p](https://ultravideo.fi/) YUV sequences (Beauty, Bosphorus, HoneyBee, Jockey, ReadySetGo, ShakeNDry, YachtRide) and place them anywhere under `data/uvg/`. The loader (`uvg_data.py`) recursively scans for `*.yuv` and matches sequences by filename.

## Run

```bash
# T2V - codebook only
bash exp_t2v/run_t2v.sh
bash exp_t2v/run_t2v_1080p.sh

# I2V - autoregressive with tail-residual correction
bash exp_i2v/run.sh
bash exp_i2v/run_1080p.sh

# FLF2V - first/last-frame conditioning
bash exp_flf2v/run_flf2v.sh
bash exp_flf2v/run_flf2v_1080p.sh
```

**VRAM** scales with the chosen backbone. With the 14B Wan2.1 used in the paper, expect around 48 GB at 720p and around 70 GB at 1080p. Swapping in the 1.3B T2V variant brings the requirement down to roughly a single consumer GPU.

Pass `--help` to any of the `run_*_experiment.py` files for the full parameter list (`M`, `K`, `steps`, `g_scale`, `ddim_tail`, etc.).

### Reproducing Paper Figures

Rate-distortion and ablation sweeps used in the paper:

```bash
bash exp_param_sweep/run_sweep.sh
bash exp_flf2v/run_rd_sweep.sh
bash exp_appendix_rd_sweep/run.sh
```

## Output Layout

```text
exp_{method}/results_{resolution}/
  summary.json
  {sequence}/
    original.mp4
    reconstructed_full.mp4
    gop{N}/
      metrics.json
      reconstructed.mp4
      codebook.tdcm
```

Attack/defense FLF2V runs additionally save:

```text
experiment_config.json
{sequence}/
  preprocess_config.json
  attacked_input.mp4
  codec_input.mp4
```

## Citation

```bibtex
@article{zeng2026gvcc,
  title   = {GVCC: Zero-Shot Video Compression via Codebook-Driven Stochastic Rectified Flow},
  author  = {Zeng, Ziyue and Su, Xun and Liu, Haoyuan and Lu, Bingyu and Tatsumi, Yui and Watanabe, Hiroshi},
  journal = {arXiv preprint arXiv:2603.26571},
  year    = {2026}
}
```

## License

Apache-2.0 (see [LICENSE](LICENSE)). The `wan/` subpackage is vendored from [Wan2.1](https://github.com/Wan-Video/Wan2.1) (Apache-2.0); upstream copyright headers are preserved. See [NOTICE](NOTICE) for the full attribution list.
