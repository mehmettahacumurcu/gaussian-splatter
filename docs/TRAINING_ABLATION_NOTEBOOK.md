# A100 training ablation notebook

`colab/learned_quality_training_ablation.ipynb` diagnoses why the learned-quality
training path can produce a white, exploded, or otherwise corrupted splat. It is
an isolated experiment: it does not replace the legacy notebook, publish a
production splat, or modify the normal learned-quality result folder.

## What it compares

The notebook holds the selected frames, camera solution, RGB images, random
seed, camera order, 720p training resolution, and checkpoint views constant. It
then runs six primary 5,000-iteration experiments:

1. `legacy_control`
2. `dense_seeds_only`
3. `masks_only`
4. `depth_only`
5. `adaptive_density_only`
6. `full_learned`

If all isolated features pass but the full combination fails, it automatically
runs the six pairwise feature combinations for 2,500 iterations each. Every
experiment records the same checkpoints at iterations 500, 1,000, 2,500, and
5,000 where applicable.

The gates inspect fixed-view masked and unmasked PSNR/SSIM/L1, Gaussian count,
non-finite values, visible white coverage, scale outliers, and anisotropy. A
failing experiment stops early so a clearly broken variant does not continue
burning A100 time.

## Drive and local-disk behavior

Before using the A100 notebook, the CPU cache-audit notebook must have completed
successfully for the same input folder. The diagnostic notebook mounts Drive
once, verifies the audit and learned-model manifest, and restores the selected
frames and pre-training evidence once into `/content/4dgs-ablation/`.

All experiment children read that session-local copy. They do not restore the
same large Drive artifacts before each training run. Generated training PLY and
checkpoint files are removed after their metrics and contact sheets have been
captured.

## Run it in Colab

1. Confirm `colab/learned_quality_cache_audit.ipynb` previously printed
   `TRACK AUDIT PASSED` for the input.
2. Open `colab/learned_quality_training_ablation.ipynb` from the
   `feature/learned-quality-a100` branch.
3. Select an A100 High-RAM runtime.
4. Set `INPUT_FOLDER` to the MyDrive-relative folder, for example
   `myroom_test`.
5. Choose **Runtime -> Run all** and grant Drive access once.
6. Leave the notebook running. It prints stage transitions, experiment IDs,
   checkpoints, gate decisions, and completion receipts live.
7. The final cell flushes Drive and releases the Colab runtime on success or
   failure.

## Output

The Drive output folder is always named `<input>_training_ablation`.
For input `myroom_test`, reports are published to:

```text
/content/drive/MyDrive/myroom_test_training_ablation
```

The folder includes:

- `ablation_report.json`: complete machine-readable results and diagnosis.
- `ablation_summary.md`: compact human-readable comparison and conclusion.
- `metrics.csv`: checkpoint metrics for all executed variants.
- `psnr_plot.png`: matched checkpoint PSNR curves.
- per-experiment receipts and fixed-view contact sheets.
- `_SUCCESS.json` only after the required diagnostic matrix is completely
  published; `_PARTIAL.json` if infrastructure interrupts the matrix.

`_SUCCESS.json` means the diagnostic matrix was published successfully. It does
not mean every tested quality variant passed. The notebook intentionally does
not publish `splat.ply` or a web viewer; its job is to identify which training
feature or interaction caused the corrupted result before the production
pipeline is changed.
