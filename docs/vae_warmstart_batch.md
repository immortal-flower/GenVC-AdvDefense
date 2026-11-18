# VAE warm-start refinement pilot

This batch tests whether the previously effective VAE reconstruction-deviation
objective can improve the accepted input perturbation. It does not optimize
the full GVCC trajectory or claim a successful generalizable attack.

## Protocol

- Jockey GOP0, 33 native 720p frames, model seed 42.
- Start both branches independently from the realized old VAE input saved in
  `input_direction_oldvae_eps2_seed42_assets/full_positive.npz`.
- Bounds: 2/255 and 4/255. Save after 1 and 4 additional complete crop sweeps.
- Frozen VAE; maximize MSE between perturbed and clean VAE crop reconstructions.
  Five-frame 192x192 crops, stride 144, three interpolated temporal controls,
  step size 0.5/255, momentum 0.8. Optimization is unquantized; exported inputs
  are realized through the native RGB quantization rule.
- This is a realized-input warm start with **reset momentum**, not resumption
  of the original optimizer. Initial perturbation stays as a fixed base; three
  anchor controls supply the additional bounded update. First/last frames can
  change, including their ordinary codec conditioning paths.
- Each of four candidates has a paired random-sign reference with identical
  absolute integer perturbation at every pixel. Saturation is handled without
  changing magnitudes. Spatial signs are shared across frames. This matches
  energy and magnitudes, not spatial frequency or all temporal statistics.
- Initial + four candidates + four random references = nine native evaluations.
  All undergo normal fresh encoding, atom reselection and decoding. No clean
  codebooks are reused. Native codec parameters: M64/K16384, 20 steps, DDIM tail3,
  guidance3, CompressAI quality4. Generation loads one VAE; each native case
  currently reloads the complete model.

## Run on laboratory machine

```powershell
& '.\exp_flf2v\run_vae_warmstart_batch.ps1'
```

Use `-PlanOnly` to check paths without using the GPU. Rerun the same command
after interruption. Generation checkpoints are saved after each complete
sweep; a partial sweep repeats. Evaluation receipts hash inputs and metrics.
Changing generation settings/source fails closed; choose a new `-Tag`.

Results: `exp_flf2v/results_720p/vae_warmstart_jockey_gop0_v1_evaluation/`
contains `summary.csv`, `paired.csv`, and `completed.json`.

## Interpretation

Positive extra PSNR drop and positive extra LPIPS indicate greater degradation.
Inspect both the gain over the original initialization and over matched random.
Input energy may grow between checkpoints and bounds: do not attribute an
epsilon-4 gain over epsilon-2 solely to a better direction. One GOP/seed is
screening only. Do not expand this batch automatically if there is no useful
improvement; inspect original/perturbed frames before claiming imperceptibility.

## Frozen candidate model-seed validation

After selecting `eps4_r4_candidate` on seed 42, run:

```powershell
& '.\exp_flf2v\run_vae_warmstart_seed_validation.ps1'
```

This runs clean/candidate/magnitude-matched random at model seeds 123 and 2026
(six fresh native evaluations). Both saved perturbations remain frozen; there
is no additional generation or tuning on these seeds. Each seed gets its own
clean baseline. Source data and asset hashes are checked before launch;
completed metrics are hash-checked for resumption. All inputs use normal
encoding, atom selection, conditioning and decoding, without codebook reuse.

Outputs: `vae_warmstart_eps4_r4_heldout_v1_evaluation/summary.csv`, `paired.csv`,
and `completed.json` under the results root. Positive extra PSNR drop and
positive extra LPIPS mean greater degradation than the matched random input.
Report both seeds, including negative results. These are held-out model seeds,
not held-out videos: source-content generalization remains untested.

## Continue the effective attack at the same bound

```powershell
& '.\exp_flf2v\run_vae_attack_continuation.ps1'
```

Continues the completed epsilon4 optimizer checkpoint, retaining its controls,
momentum, fixed initial input and crop schedule. The source batch stays intact;
the new tag is `vae_continue_eps4_r12_jockey_gop0_v1`. Save total warm-start
rounds8 and12 (4 and8 further sweeps beyond the already completed round4).
The bound remains 4/255, alpha0.5/255 and momentum0.8. The source four-round
checkpoint must reproduce its saved candidate and random pixels exactly.

Two new candidates and their matched random references require four native
720p codec evaluations. Verified round4 metrics are reused as the baseline.
In this batch `ExtraDropVsInitial` and `ExtraLPIPSVsInitial` refer to the existing
**epsilon4 round4 candidate**, not the old epsilon2 initializer or clean video.
Do not assume later iterations improve the actual codec. Input energy can
change within the same Linf bound; report it with paired random controls.
Existing files are preserved even if later candidates are weaker. This remains
VAE-surrogate optimization accepted by a real codec, not end-to-end GVCC PGD.

To continue the completed round12 state to rounds20 and28, without increasing
the 4/255 bound or resetting momentum:

```powershell
& '.\exp_flf2v\run_vae_attack_continuation.ps1' `
    -SourceTag 'vae_continue_eps4_r12_jockey_gop0_v1' `
    -Tag 'vae_continue_eps4_r28_jockey_gop0_v1' `
    -TargetRounds @(20,28)
```

The source checkpoint's final round is read from its completed generation
report and checked against the optimizer. In this run the comparison baseline
is **round12**, and only four new native codec evaluations are needed. Source
round12 assets and results are preserved. An interrupted new run resumes with
the same command. The 20/28 labels count total warm-start refinement sweeps,
not an additional 20/28 sweeps beyond round12.
