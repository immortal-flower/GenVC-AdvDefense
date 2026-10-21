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

The following command creates a 153-row manifest without starting inference:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File ".\exp_flf2v\run_codebook_sensitivity_grid.ps1" `
  -Field sign -Count 16 -AttackSeeds 42 -PlanOnly
```

The manifest is saved under `exp_flf2v/results_720p` and marks each job as
`pending` or `complete`.

## Run the uniform sign-bit grid

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File ".\exp_flf2v\run_codebook_sensitivity_grid.ps1" `
  -Field sign -Count 16 -AttackSeeds 42
```

To test both fields under the same budget:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File ".\exp_flf2v\run_codebook_sensitivity_grid.ps1" `
  -Field both -Count 16 -AttackSeeds 42
```

For a short pipeline check before the full sweep:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File ".\exp_flf2v\run_codebook_sensitivity_grid.ps1" `
  -Field sign -Count 16 -Steps 1,17 -Frames 0,3,4,8 -AttackSeeds 42
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
