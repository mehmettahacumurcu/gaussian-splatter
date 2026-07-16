# Learned-quality A100 room test

This is an isolated acceptance experiment for the highest-quality static Gaussian
splat path. It does not change the generated production notebook or its
`<input_folder>_result` output.

## Run

1. Open `colab/learned_quality_a100_experiment.ipynb` in Google Colab.
2. Select an **A100 GPU** with **High-RAM**.
3. Set `INPUT_FOLDER` to the existing MyDrive-relative capture folder. Do not include
   `/content/drive/MyDrive`, and do not select a result folder.
4. Choose **Runtime -> Run all**, authorize Drive, and leave the tab running.

The notebook preflights A100 VRAM and local disk before mounting Drive. It then checks
out one immutable source commit, creates a protected learned-model environment, verifies
the pinned source/checkpoint manifest, and runs the experiment through an argv-only CLI.

## Output boundary

For input `captures/room`, success is published only to:

```text
MyDrive/captures/room_learned_test_result/
```

The legacy `MyDrive/captures/room_result/` is not an ownership or replacement target.
Failures publish allowlisted diagnostics under
`room_learned_test_diagnostics/<run_id>/` and never publish a partial splat.

The success folder includes:

- `splat.ply`
- `experiment_report.json` and `geometry_candidates.json`
- `quality_report.json` and `run_manifest.json`
- the verified `model_manifest.json`
- `diagnostics/masks_contact_sheet.png`
- `diagnostics/depth_contact_sheet.png`
- `diagnostics/geometry_contact_sheet.png`
- `diagnostics/final_render_contact_sheet.png`
- density and photometric diagnostic reports

There is no bundled web viewer. Open `splat.ply` in SuperSplat or the existing project
viewer.

## Acceptance boundary

The local suite validates deterministic artifact joins, mask polarity, photometric
fallbacks, depth support, adaptive density safety, training isolation, publishing,
and notebook structure. The real A100 room run remains the quality acceptance test.
Do not claim Luma parity or a visual improvement until the learned output is compared
with the untouched legacy output and the four contact sheets show no systematic
corruption.
