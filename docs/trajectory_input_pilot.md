# Trajectory-aware input pilot v1

## Question and limits

Can a bounded RGB perturbation, optimized through a short *updated* Wan/SDE
trajectory with codebook reselection, damage a fresh GVCC reconstruction more
than a matched random perturbation? This is a hypothesis, not an established
attack or a claim of paper novelty.

The default experiment uses Jockey GOP0, seed 42, epsilon 2/255, 33 frames,
128x224 surrogate resolution, two SDE updates on the native 20-step/shift-5
schedule, and four PGD iterations. Each update recomputes velocity, residual,
full-K=16384 hard top-M=64 selection, correction and the next velocity. The
backward path approximates discrete selection with a freshly selected top-80
soft bank. The hard-forward equivalence test applies at the surrogate geometry,
not at native 720p. The objective averages predicted clean-latent error after
each correction against the clean source latent. It is not final RGB distortion.

Initialization is the existing full-positive old-VAE perturbation. First/last
RGB perturbations are frozen. Those submitted boundary frames are compressed
with CompressAI quality 4 once, then resized to construct fixed surrogate
conditions. The full-GOP VAE target remains differentiable. VAE and recurrent
DiT backward passes use a separated latent VJP to save memory; frozen weights
do not mean detached input gradients. Two-GPU DiT dispatch and block checkpointing
are used without the normal inference-only memory interceptors.

## One-command workflow (lab PowerShell, repository root)

```powershell
& .\exp_flf2v\run_trajectory_input_pilot.ps1
```

Use `-PlanOnly` to check paths without loading models. `-Iterations 4
-RolloutSteps 2` are the defaults. The generator checks actual quantized hard
rollout scores at full and half update sizes and accepts only an improvement.
It checkpoints every completed optimization iteration. Repeating the command
resumes after interruption; changed settings/source hashes require another
output directory. If the objective never improves, the runner stops before
expensive full-codec acceptance and records the negative outcome.

When improved, initial/candidate/matched-random inputs each undergo fresh normal
720p encoding and decoding: no clean codebook reuse, step splicing or stream
modification. Matching is per-pixel perturbation magnitude after quantization,
with identical boundary perturbations; the random control randomizes interior
spatial signs shared over time. Consequently it is a conditional interior
randomization control, not an entirely random boundary attack. Initial-vs-new
candidate comparisons need their input MSE reported because these need not match.

Generation loads the model once. Acceptance currently launches three separate
codec processes (and thus loads three times). Results with matching receipts
and file hashes are skipped on restart. No full single-load codec batch claim.

## Read the result

`trajectory_v1_jockey_gop0_eps2_seed42_r2_p4_assets/generation.json` records
hard objective values, accepted/backtracked updates, gradients, time, bounds,
source hashes, limitations and completion status. `optimizer_checkpoint.npz`
stores resumable controls, not model weights.

`trajectory_v1_jockey_gop0_eps2_seed42_r2_p4_summary.csv` records final PSNR,
LPIPS, quality relative to submitted video, actual input MSE/L-infinity and BPP.
Lower candidate PSNR and higher LPIPS than matched random are the initial
acceptance evidence. Proxy improvement alone is not success. Report disagreement
between metrics and all three cases; do not cherry-pick. Further GOP/seed testing
is needed before any generalization claim.

CPU unit tests cover the actual hard-forward atom selection, surrogate gradient,
recurrent update/re-prediction and invalid schedules. Full 14B GPU memory use,
runtime and attack efficacy must be tested on the laboratory machine.
