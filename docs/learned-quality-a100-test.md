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

## Resolved: COLMAP startup failure (2026-07-17)

The first corrected A100 room run passed the iPhone MOV compatibility probe, then
failed before training while validating the installed COLMAP executable:

```text
Command '('/usr/local/bin/colmap', '--version')' returned non-zero exit status 1.
```

- Input: `MyDrive/myroom_test/IMG_5624.MOV`
- Diagnostics:
  `MyDrive/myroom_test_learned_test_diagnostics/1929fd7d0f2c46d495588b95a9448998/`
- Receipt status: `failed`
- Error type: `CalledProcessError`
- Runtime was flushed, unmounted, and released correctly after the failure.

The pinned CUDA build supports `colmap -h` but rejects `colmap --version` with
exit status 1. Runtime commit `d130d92531be5f805acaaef4c4681f10ca287cd2`
now probes the supported help command and extracts the version banner from its
stdout or stderr. Every later notebook pin includes that repair.

The subsequent A100 run confirmed COLMAP advanced past the version probe and
completed feature extraction, matching, and mapping.

## Resolved: duplicate static-track observation (2026-07-17)

Run `a2ff823246ad40949a357be7fe1324d5` then failed while converting the
classical COLMAP model into learned static-track evidence:

```text
a static track cannot observe one frame more than once
```

The COLMAP reconstruction itself was usable, but one point track referenced the
same canonical frame more than once. Runtime commit
`30e7cd2` now discards only such ambiguous evidence tracks before flow, mask,
and photometric validation; it does not discard or weaken the accepted geometry.

The notebook launch is also unbuffered as of `04c81d5`, so preprocessing events
and training iteration logs stream live during the next A100 run. The checked-in
notebook is pinned to the full `04c81d5480a6772fd7054ca25954808963fea627`
runtime commit, which includes both repairs.
