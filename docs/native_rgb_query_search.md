# Native RGB query-search pilot

Run on the laboratory machine, not the development laptop:

```powershell
& '.\exp_flf2v\run_native_rgb_search.ps1'
```

This is an offline, reconstruction-feedback attack search on Jockey GOP0, not
an end-to-end gradient attack or proof of access to a deployed service. It
assumes access to the original video and the reconstructed output for scoring.

## Fixed protocol

- Initial input: saved old-VAE perturbation, no new gradients.
- Native codec: 33 frames, 1280x720, M64/K16384, 20 steps/3 deterministic tail,
  g=3, CompressAI q4, model seed42. Every query re-encodes and reselects atoms.
- Four rounds, each with a seeded smooth low-dimensional RGB direction;
  evaluate both signs from the SAME current incumbent, then select by mean
  frame PSNR relative to the clean source (lower is better). LPIPS is reported,
  not used for selection. PSNR is averaged from per-frame values rather than
  the two-decimal display field. Minimum accepted improvement is 0.001 dB,
  an optimization tolerance, NOT a significance threshold.
- Project each candidate to <=4/255 and quantized input SSE in [99%,100%] of
  the initial perturbation's SSE. Preserve both initial submitted boundary
  frames exactly. RMS matching alone is not a perceptual invisibility claim.
- One fresh baseline query and eight search candidates. If the best search
  improvement is >=0.1 dB, evaluate initialization and frozen best input with
  model seed123. Held-out output is NEVER used to change the input.
- At most eleven distinct intended evaluations. Failed attempts/retries are
  counted separately in `codec_launches`; duplicate input/seed pairs use verified
  cached metrics. A crash can require repeating an unfinished evaluation.
- Codec models currently load per candidate; the coordinator uses no GPU model.

## Outputs and interpretation

`results_720p/native_query_jockey_gop0_energy_matched_v1_search/` contains
`search.json`, `summary.csv`, assets, and per-query logs. `best` in search.json
identifies the selected asset and metrics. Source hashes and a configuration
fingerprint prevent resuming under changed inputs/settings; metrics hashes guard
cache reuse. Run the same command to resume after a failure.

The score is the result of an adaptive search on ONE GOP and model seed. It is
optimistically selected, not generalization evidence. The additional seed check
only tests model-seed transfer on the same content. Report negative results too,
the full query budget, input MSE, LPIPS, submitted-reference PSNR and BPP. This
pilot does not establish novelty over established derivative-free search methods.
