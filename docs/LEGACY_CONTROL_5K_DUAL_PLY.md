# Legacy-control 5K dual PLY comparison

This bounded notebook answers one question: how does the exact raw output of the
passing legacy-control 5K diagnostic differ from the output of the production
PLY polish stage?

## Run it

1. Complete the matching CPU track audit and A100 structural diagnostic matrix.
2. Open `colab/learned_quality_legacy_control_5k.ipynb`.
3. Select an A100 80 GB High-RAM runtime.
4. Set `INPUT_FOLDER` to the original MyDrive-relative input, such as
   `myroom_test`.
5. Choose **Runtime -> Run all**.

The notebook restores the verified selection and pre-training lineage once. It
then trains only the deterministic 5K `legacy_control` variant. It does not run
the diagnostic matrix again and it never starts a full 120K training.

## Outputs

The owned result is:

```text
MyDrive/<input>_legacy_control_5k_result/
```

Important files:

- `raw_legacy_control_5k.ply`: byte-identical trainer export.
- `polished_legacy_control_5k.ply`: candidate created by the production
  polisher.
- `polish_report.json`: counts, opacity-mass change, render metrics, acceptance,
  and ordered rejection reasons.
- `metrics.jsonl`: the 5K training metrics.
- `contact_005000*.png`: fixed and perturbed diagnostic views.
- `provenance.json`: source, lineage, and file hashes.

The result publisher verifies that both PLY files exist inside the owned result
folder and that their hashes differ. A missing or aliased PLY is a hard failure.

## What polish changes

The production polisher conservatively filters low-opacity, excessive-scale,
high-anisotropy, non-finite, and sparse-bound outliers. It retains a crop margin
with a smooth opacity fade, validates the candidate PLY, and renders matched
camera views for regression checks.

The report can reject the candidate for excessive point removal, opacity-mass
loss, incomplete or non-finite render metrics, mean PSNR/SSIM loss, or a
single-view PSNR regression. This diagnostic still preserves a valid rejected
candidate so the raw and polished files can be compared visually. That does not
weaken the production acceptance gate.
