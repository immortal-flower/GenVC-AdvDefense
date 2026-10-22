# GVCC codebook sensitivity grid

This stage replaces the earlier separate step scan and latent-position scan
with one controlled `17 x 9` experiment. Every grid cell uses the same physical
bit budget and targets exactly one SDE step and one latent time position.

## What the sweep controls

- SDE steps use the command-line convention `1..17`.
- Latent positions use `0..8`.
- `Count` is the number of physical bits modified in each cell.
- `Field` selects sign bits, index bits, or both fields.
- Every job reuses the same clean serialized `.tdcm`, so encoding is skipped.
- Existing jobs with `metrics.json` are skipped and the sweep can resume.

`AttackSeeds` only changes which eligible bits are selected inside a fixed
clean codebook. It does **not** create a new Gaussian codebook or a new clean
generation trajectory. Cross-codebook-seed validation requires separately
encoded clean codebooks and belongs to the next validation stage.

## Inspect the job plan

Run the script directly from PowerShell so that list-valued parameters such as
`Steps` and `Frames` are bound as arrays. If local policy blocks `.ps1` files,
enable them only for the current terminal first:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
```

The following command creates a 153-row manifest without starting inference:

```powershell
& ".\exp_flf2v\run_codebook_sensitivity_grid.ps1" `
  -Field sign -Count 16 -AttackSeeds 42 -PlanOnly
```

The manifest is saved under `exp_flf2v/results_720p` and marks each job as
`pending` or `complete`.

## Run the uniform sign-bit grid

```powershell
& ".\exp_flf2v\run_codebook_sensitivity_grid.ps1" `
  -Field sign -Count 16 -AttackSeeds 42
```

To test both fields under the same budget:

```powershell
& ".\exp_flf2v\run_codebook_sensitivity_grid.ps1" `
  -Field both -Count 16 -AttackSeeds 42
```

For a short pipeline check before the full sweep:

```powershell
& ".\exp_flf2v\run_codebook_sensitivity_grid.ps1" `
  -Field sign -Count 16 -Steps @(1,17) -Frames @(0,3,4,8) -AttackSeeds @(42)
```

## Outputs

After the run, `summarize_codebook_sensitivity_grid.py` produces:

- `sensitivity_grid_jockey_tidy.csv`: one row per completed experiment;
- `sensitivity_grid_jockey_aggregate.csv`: mean and standard deviation over
  available attack-selection seeds;
- one `17 x 9` PSNR-drop matrix for each field, bit budget, and attack seed;
- `sensitivity_grid_jockey_summary.json`: clean reference, run metadata, and
  aggregate results.

Important columns include the field type, SDE step, latent position, requested
and actual changed bits, payload BER, PSNR drop, LPIPS change, and metrics path.

The summarizer can also be rerun without model inference:

```powershell
& "C:\Users\Admin\.conda\envs\GVCC-5090\python.exe" `
  ".\exp_flf2v\summarize_codebook_sensitivity_grid.py" `
  --results_root ".\exp_flf2v\results_720p" `
  --sequence Jockey `
  --clean_metrics ".\exp_flf2v\results_720p\sensitivity_clean_redecode\Jockey\gop0\metrics.json"
```

## Interpretation boundary

The sign and index grids provide an equal-physical-bit comparison. They do not
yet provide an equal-semantic-change comparison because one index identifies a
14-bit atom number while one sign contains one bit. A later experiment should
also compare the fields under an equal number of affected atoms.

## Persistent targeted validation

After the full grid identifies candidate hotspots, use the persistent batch
runner instead of launching one Python process per cell. It loads Wan2.1 once
and reuses the reference compression, boundary condition, empty-prompt
embedding and clean codebook for all validation jobs.

Inspect the default 8-cell x 3-seed plan without loading the model:

```powershell
& ".\exp_flf2v\run_codebook_validation_batch.ps1" -PlanOnly
```

Run the 16-bit repeatability validation. Existing seed-42 grid results are
recognized by their run names and skipped:

```powershell
& ".\exp_flf2v\run_codebook_validation_batch.ps1" `
  -Counts @(16) -AttackSeeds @(42,123,2026)
```

Once repeatability is established, add the 8-bit and 32-bit budgets:

```powershell
& ".\exp_flf2v\run_codebook_validation_batch.ps1" `
  -Counts @(8,16,32) -AttackSeeds @(42,123,2026)
```

The default run omits reconstructed MP4 files because metrics are computed
from in-memory frames. Add `-SaveVideo` only when qualitative examples are
needed. Progress is written to
`validation_batch_manifest_jockey.csv`, and rerunning the same command resumes
only missing jobs.
