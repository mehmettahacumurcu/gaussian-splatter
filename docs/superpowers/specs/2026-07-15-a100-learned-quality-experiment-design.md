# Isolated A100 Learned-Quality Notebook Experiment Design

**Date:** 2026-07-15
**Status:** Approved for implementation
**Target:** Maximum-quality static room reconstruction on an NVIDIA A100
**Output:** `<input_folder_name>_learned_test_result`

## 1. Summary

This experiment adds one standalone Colab notebook that tests learned geometry,
depth, segmentation, and motion cues around the existing static Gaussian-splat
trainer. It does not add a frontend choice, API route, notebook-generator preset,
or production runner behavior.

The notebook runs multiple geometry candidates on the same Smart-selected frame
manifest, rejects candidates that fail the existing reconstruction gates, and
trains only the strongest passing candidate. Learned depth and masks may improve
the training inputs when their validation gates pass, while the existing Ultra
trainer, guarded polish, metadata, and portable PLY export remain the final
reconstruction path.

The experiment is quality-first. Runtime, download size, and A100 cost are not
optimization goals. The prior `<input_folder_name>_result` folder is read only
and is never renamed, merged, or replaced.

## 2. Locked Product Boundaries

1. Create one static notebook at
   `colab/learned_quality_a100_experiment.ipynb`.
2. Put experiment orchestration and adapters under
   `experiments/learned_quality/`.
3. Do not add the experiment to the frontend, FastAPI routes, notebook
   generator, product presets, or `_production_services()`.
4. Do not edit the behavior of `colab/video_to_world.ipynb`,
   `colab/sota_verify.ipynb`, `colab/phase2_verify.ipynb`, or the generated
   static notebook workflow.
5. Reuse production modules through the existing `RunnerServices` injection
   boundary. Production callers continue to receive the unchanged service set.
6. The Drive input remains read only.
7. The only successful destination is the exact sibling
   `<input_folder_name>_learned_test_result`.
8. The existing `<input_folder_name>_result` is never considered a replacement
   target, even when `replace_owned_result` is enabled for the experiment.
9. The result contains a standard INRIA/3DGS `splat.ply` and no bundled viewer.
10. This first experiment does not create a public product feature. Promotion
    into the production pipeline requires a later decision based on the room
    result and its reports.

## 3. Notebook Experience

The notebook has one Colab form field:

```python
INPUT_FOLDER = ""  # MyDrive-relative path; prompt when blank
```

When `INPUT_FOLDER` is blank, the first execution prompts once after Drive is
mounted. The notebook validates the same relative MyDrive path rules as the
production generator and shows the three immutable paths before compute starts:

```text
Input:       MyDrive/<input_folder>
Old result:  MyDrive/<input_folder>_result          (read only)
Experiment:  MyDrive/<input_folder>_learned_test_result
```

The remaining cells are Run-All-safe orchestration:

1. GPU, VRAM, disk, CUDA, and Python runtime preflight.
2. Mount Drive, then prompt for, normalize, validate, display, and copy
   `INPUT_FOLDER` to a unique local workspace.
3. Checkout one pinned repository commit.
4. Run the normal Colab bootstrap, then install pinned experiment dependencies
   without upgrading Torch, NumPy, or gsplat.
5. Verify Torch/CUDA, gsplat, COLMAP, PyCOLMAP, FFmpeg, and every learned model
   import in fresh subprocesses.
6. Run one checked argv command:
   `python scripts/learned_quality_run.py --spec /content/learned_spec.json`.
7. Validate `_SUCCESS`, print the result paths, and display the mask, geometry,
   depth, and final-render contact sheets.

User strings are JSON data and argv elements. They are never interpolated into
Python source or shell commands.

The experiment accepts A100 runtimes with at least 39 GiB usable VRAM. It does
not inherit the product Ultra preset's 75 GiB preflight because this notebook's
density budget is measured dynamically. An 80 GB A100 receives the larger DA3
anchor budget; a 40 GB A100 keeps the same 120k/native-resolution quality policy
but may stop Gaussian growth earlier to preserve the mandatory VRAM reserve.

## 4. Reproducible Model Stack

The first experiment deliberately uses checkpoints that can also remain viable
if the result later justifies a public product path.

| Job | Model | Pinned source | License |
| --- | --- | --- | --- |
| Multi-view cameras, confidence, and relative depth | `depth-anything/DA3-BASE` | HF revision `f4a6c9b3c95e41c82048423d3493a81ec3fa810e`; DA3 code `3fe327a6abe2e5db95b54444ea95463dbfef5610` | Apache-2.0 |
| Metric-depth diagnostic and sky proposal | `depth-anything/DA3METRIC-LARGE` | HF revision `4010e39f3634a45bc60553321fb49fb760bd594e` | Apache-2.0 |
| Open-vocabulary transient detection | `IDEA-Research/grounding-dino-tiny` | HF revision `a2bb814dd30d776dcf7e30523b00659f4f141c71` | Apache-2.0 |
| Precise image/video masks | `facebook/sam2.1-hiera-large` | HF revision `665f8e2ad61cf5f53d65644ff27c8ee525124610`; SAM 2 code `2b90b9f5ceec907a1c18123530e92e794ad901a4` | Apache-2.0 |
| Dense flow and uncertainty | `MemorySlices/Tartan-C-T-TSKH-spring540x960-M` | HF revision `eb97ef34ba5d856c3fa2cdcd073150c057ac8b69`; SEA-RAFT code `9137517ba24e628442aec097d3afe71d03503b75` | BSD-3-Clause |

`VGGT-Omega`, DA3 Giant/Large/Nested, CoTracker3, and UniDepth are not part of
this notebook because their released checkpoints are non-commercial. The gated
`VGGT-1B-Commercial` checkpoint is also excluded so Run All does not require a
Hugging Face approval/token flow.

Models run sequentially. Each stage writes CPU artifacts, deletes the model,
runs garbage collection, empties the CUDA cache, and records before/after VRAM.
No learned model remains resident during Gaussian training.

The dependency lock records exact package versions and direct-source commits.
Installation uses that lock only after the normal Colab bootstrap and is
rejected if its resolved plan would replace Torch, NumPy, or gsplat. The final
model manifest records the resolved package versions, checkpoint revisions, and
repository commit used by the run.

## 5. Component Boundaries

```text
experiments/learned_quality/
  contracts.py       experiment spec, artifacts, branch and report schemas
  dependencies.py    pinned install/validation manifest
  requirements-lock.txt  exact experiment-only Python dependency versions
  da3.py             anchor cameras, pose-conditioned depth, sky diagnostic
  segmentation.py    Grounding DINO detections and SAM 2 refinement/propagation
  flow.py            SEA-RAFT forward/backward flow and rigid-flow residuals
  masks.py           evidence fusion, polarity conversion, statistics/previews
  photometric.py     robust cross-frame RGB gain/bias normalization
  geometry.py        classical and DA3-seeded COLMAP candidates and selection
  depth.py           final-pose depth alignment, consistency, dense seed fusion
  training.py        experiment training adapter and validity-mask wiring
  density.py         no-overshoot, VRAM-aware adaptive density controller
  reports.py         experiment evidence and production bundle augmentation
  publish.py         transactional learned-test suffix publishing/diagnostics
  runner.py          isolated RunnerServices composition and stage orchestration

scripts/learned_quality_run.py
  validated CLI and machine-readable receipt

colab/learned_quality_a100_experiment.ipynb
  thin pinned Run-All orchestration
```

The permitted shared-code edits are narrow, default-off trainer seams:

1. a `validity_mask` argument whose absent/`None` path executes the exact legacy
   loss and logging expressions; and
2. an optional trainer-extension boundary plus feature-detected density methods
   that provide iteration/view context to an injected experiment controller.

No production runner or configuration enables either seam, and regression tests
must demonstrate unchanged legacy outputs and logs. All mask loading, adaptive
density policy, model lifecycle, and experiment configuration remain
experiment-local. The experiment must not reuse `lambda_mask_motion`: that field
upweights masked pixels and has the wrong semantics for transient/sky exclusion.

## 6. End-to-End Data Flow

### 6.1 Source and frame selection

The experiment reuses production source discovery, immutable source hashing,
Smart selection, native extraction, and the Ultra frame budget. Smart selection
remains deterministic and preserves timestamp coverage. Fixed FPS is not exposed
in this standalone notebook because the existing result is the external
baseline and this run is explicitly maximum quality.

All geometry candidates use the same final selected-frame manifest and exact
frame bytes. Both must attempt every manifest frame; naturally unregistered
frames remain in each candidate's metrics and report rather than being
prefiltered from that candidate's denominator.

### 6.2 DA3 anchor reconstruction

DA3 does not receive all 800 possible Ultra frames in one tensor. It receives
chronological, coverage-aware anchors:

- 96 anchors on an A100 with less than 70 GiB VRAM;
- 120 anchors on an A100 with at least 70 GiB VRAM;
- fewer anchors only when the selected manifest contains fewer images.

Those anchor counts and the 504-pixel process resolution are quality locks. A
DA3 anchor-camera OOM aborts with diagnostics; it does not silently lower anchor
coverage or resolution. The smaller-chunk OOM retry described later applies
only to stages whose outputs are mathematically batch/chunk independent.

Inference uses `process_res=504`, `upper_bound_resize`, `use_ray_pose=True`, and
`ref_view_strategy="middle"`. Extrinsics are normalized defensively from either
3x4 or 4x4 W2C form. Intrinsics are converted to one robust shared PINHOLE camera
because the input is one iPhone video; per-frame focal variation is retained
only when diagnostics prove real variation rather than model noise.

The built-in DA3 COLMAP exporter is forbidden. It creates one track-length-one
point per accepted pixel and can materialize tens of millions of points that
bundle adjustment cannot use. The experiment creates a cameras-and-images-only
known-pose reconstruction, extracts/matches real features, triangulates shared
tracks with PyCOLMAP, and then bundle-adjusts them.

Bundle adjustment uses two guarded passes:

1. poses and points with principal point and focal fixed;
2. optional focal-only refinement, accepted only when reconstruction metrics
   improve, no gate worsens, and focal drift remains within 5% of the robust
   shared initialization.

The refined anchor model is then used as mapper input to register the remaining
selected frames.

### 6.3 Semantic, sky, and motion evidence

Grounding DINO runs on every selected frame with these transient prompts:

```text
person. child. hand. dog. cat. animal. bicycle. motorcycle. scooter.
car. truck. bus.
```

Furniture, plants, curtains, mirrors, screens, and windows are not semantic
transients. SAM 2.1 Large refines every direct box. Short chronological clips
are seeded near their midpoint and propagated forward and backward only to fill
or stabilize direct detections. A propagated mask cannot become hard evidence
without nearby direct detection or motion support.

DA3Metric-Large proposes sky. An indoor sky proposal becomes hard evidence only
when it is also far/low-support under final geometry and is not explained as a
stable ceiling or wall. Suspicious ceiling-shaped proposals remain uncertain and
are not excluded.

SEA-RAFT runs forward and backward. Raw flow magnitude is never interpreted as
object motion. DA3 depth/pose predicts rigid camera flow, and motion evidence is
the residual after camera-motion compensation. A pixel is eligible only when:

- projected depth is positive and in bounds;
- target-depth/z-buffer agreement passes;
- it is outside dilated depth/occlusion boundaries;
- forward/backward cycle consistency passes;
- SEA uncertainty is below the robust per-pair threshold; and
- the residual exceeds a threshold calibrated from static, long-track,
  low-reprojection COLMAP keypoints.

Interior frames require agreement from incoming and outgoing pairs, or one
flow residual plus a direct semantic mask. High flow uncertainty means unknown,
never dynamic.

### 6.4 Mask fusion and polarity

The experiment preserves four independent maps:

- `semantic_confirmed`;
- `sky_confirmed`;
- `motion_confirmed`;
- `uncertain`.

`hard_exclude` is the union of only the three confirmed maps. Boundary dilation,
opening, component filtering, temporal area checks, and propagation-collapse
checks are resolution-scaled. If hard exclusion exceeds 45% of a non-sky frame,
the weakest motion-only components are removed first. If it still exceeds 55%,
that frame's learned mask is rejected and the unmasked frame is used with a
warning.

Two explicitly named polarities prevent silent inversion:

- COLMAP keep mask: white means extract features, black means ignore;
- training exclusion mask: one means exclude from loss.

COLMAP masks follow the exact relative image filename plus `.png` convention and
are generated at exact image dimensions. Any mask change creates a fresh feature
database.

### 6.5 Photometric normalization

Using only confirmed-static overlap pixels, the experiment solves one robust RGB
gain and bias per frame with temporal smoothness. Gains are clamped to
`[0.75, 1.33]` and biases to `[-0.10, 0.10]` in normalized RGB. The corrected
sequence is accepted only when held-out static correspondences improve and no
channel clips more than 0.5% additional pixels. Otherwise the original frames
remain the training truth.

This correction is deterministic and does not use an image-enhancement model,
which could invent view-inconsistent details.

Photometric correction is a training-only input. Classical and Learned Hybrid
feature extraction always see the same original selected-frame bytes; their
only intended input difference is the Learned Hybrid COLMAP keep mask. After a
geometry winner is selected, an accepted photometric correction becomes the RGB
training truth.

### 6.6 Geometry candidates and winner selection

Two candidates run from fresh directories and databases:

1. **Classical:** the existing unmasked COLMAP path and reconstruction policy.
2. **Learned hybrid:** learned COLMAP keep masks plus DA3 known-pose anchors,
   true shared-track triangulation, guarded bundle adjustment, and registration
   of the remaining frames.

Both candidates are evaluated against the same selected manifest with the exact
production gates. A failed candidate cannot reach training. If neither passes
and Smart has not backfilled, the failed candidate closest to passing is ranked
by `(failed_gate_count, normalized_gate_deficit, candidate_id)`, ascending. For
a minimum gate, deficit is `max(0, (threshold - value) / max(abs(threshold),
1e-9))`; for a maximum gate it is `max(0, (value - threshold) /
max(abs(threshold), 1e-9))`; a failed Boolean gate contributes one. The sum is
rounded to `1e-9` before comparison. Candidate IDs are `classical` and
`learned_hybrid`.

Exactly one canonical production Smart backfill is then planned from that
candidate's uncovered intervals with `max_per_interval=2`. It replaces weaker
interior choices, never exceeds the original Ultra frame budget of 800, and
marks the manifest so another backfill is impossible. Both candidates rerun
from fresh databases on this identical backfilled manifest. If neither final
candidate passes, the experiment stops.

Among passing candidates, selection compares this exact ordered tuple. Floating
metrics are rounded to `1e-9`, and `-` means higher is better. A candidate
missing any required selection metric is schema-invalid and cannot win:

```text
(-registered_ratio,
 -registered_count,
 -dominant_component_share,
 -covered_endpoint_count,
  maximum_interior_gap,
  median_reprojection_error,
  p95_reprojection_error,
 -median_track_length,
 -usable_sparse_point_count,
  candidate_id)
```

`covered_endpoint_count` is 0, 1, or 2 under the existing production endpoint
tolerance; `maximum_interior_gap` uses the manifest's canonical time/photo-order
coverage unit. Lexicographically smallest wins, so the final candidate ID is
only a stable tie-breaker.

The report includes both candidates, even when the classical candidate wins.
Learned initialization is never forced merely because this is a learned test.
The independently validated training mask, final-pose depth, photometric RGB,
and dense seeds are applied to whichever geometry candidate wins, including the
Classical candidate. A learned artifact is omitted individually when its own
gate fails; the geometry branch name does not accept or reject it implicitly.

### 6.7 Final-pose depth and dense initialization

After a winner exists, DA3-Base reruns pose-conditioned depth in overlapping
24-48-frame chunks using the final COLMAP W2C poses and intrinsics with
`align_to_input_ext_scale=True`. This ties every chunk to one COLMAP coordinate
system. DA3Metric depth is a diagnostic/cross-check, not an independently mixed
scale.

Depth values are zeroed on excluded/uncertain boundaries, rejected on low DA3
confidence, and required to agree after reprojection in at least two neighboring
views. Multi-view-consistent points are fused, voxel-filtered, and colorized by
the median of valid observations. Existing sparse COLMAP points are always
preserved. Dense learned seeds are bounded to one million as a corruption guard,
not a target; confidence and view support decide the actual count.

The production reconstruction model remains untouched for gate revalidation.
The experiment training adapter supplies the additional dense seeds only during
Gaussian initialization.

## 7. Maximum-Quality Training Policy

The baseline is the existing Ultra static trainer:

- 120,000 iterations;
- native source resolution at the final multiresolution stage;
- SH degree 3;
- VGG LPIPS `0.15`;
- depth loss `0.15`;
- SSIM weight `0.20`;
- density events from iteration 500 through 79,999 every 100 iterations;
- existing anisotropy, opacity-reset, scale, and polish controls.

The experiment changes only what is required to consume validated learned
evidence:

1. confirmed invalid pixels contribute zero gradient to L1, SSIM, LPIPS, and
   depth;
2. low-confidence motion may be softly downweighted but cannot become hard zero;
3. final-pose DA3 depth replaces Metric3D depth only after alignment and
   multi-view consistency gates pass;
4. validated dense seeds augment sparse initialization;
5. adaptive density growth replaces the fixed 3M target.

Pixelwise L1 and depth terms multiply by the valid mask and normalize by valid
weight, not full image area. Before SSIM or LPIPS, invalid prediction pixels are
composited with detached target pixels so excluded predictions have no gradient.
Their spatial reductions use conservative validity maps eroded/pooled to each
loss's receptive field; a view with no valid support skips that term and records
the reason. Logged L1, SSIM, LPIPS, depth, and PSNR use the same support as their
optimized term. `validity_mask=None` bypasses all of this and executes the exact
legacy expressions.

### 7.1 Adaptive Gaussian growth

Six million Gaussians is an emergency ceiling, not a target. The Ultra density
schedule remains iterations 500 through 79,999 at interval 100; an early-stop
decision disables later growth events but never shortens the 120k training run.
Before every density step, pruning runs first. Eligible clone/split candidates
are ranked by visibility-qualified accumulated gradient. Growth is limited by
all of:

- remaining slots below 6,000,000;
- live CUDA free memory with a reserve of `max(10 GiB, 15% of total VRAM)`;
- a conservative per-Gaussian memory estimate including parameters, gradients,
  Adam state, and a 1.5 safety factor; and
- the number of genuinely eligible high-gradient candidates.

At each event, CUDA is synchronized after pruning and temporary render tensors
are released before free memory is measured. The controller selects only the
highest-ranked candidates that fit the budget and cannot overshoot the hard
ceiling in one step. The first eligible growth is limited to 10,000 Gaussians to
calibrate the analytical memory estimate against its synchronized allocated-
memory delta. Later events use the larger per-Gaussian estimate with the 1.5
safety factor; if a safe estimate cannot be established, growth stops instead
of guessing.

For the low-candidate stop, the numerator is the number of finite, above-
threshold clone/split candidates and the denominator is the number of
non-pruned Gaussians visible in at least two training views during that density
window. It must remain below 0.5% for three consecutive windows. The quality
plateau uses eight deterministic coverage-spaced cameras (or all cameras when
fewer than eight), fixed when training starts and rendered at the active
resolution with the accepted validity masks. Growth stops only after three
consecutive 5,000-iteration evaluations each improve aggregate valid-pixel PSNR
by less than 0.02 dB. Training continues after growth stops so appearance and
opacity can stabilize.

The report records pre/post-prune and post-growth Gaussian count, synchronized
free/allocated/peak VRAM, candidate numerator/denominator, clone/split/prune
counts, memory-estimate inputs, growth-stop reason, and quality trend at every
density event.

## 8. Failure, Fallback, and Artifact Safety

- Model download/import failure stops before geometry and records the exact
  pinned model/revision.
- A learned-stage numerical/shape failure marks that evidence unavailable; it
  never fabricates an all-valid or all-invalid success artifact.
- The unmasked classical reconstruction is always retained as an independent
  fallback candidate.
- A mask cannot destroy the only usable reconstruction because the learned
  branch has its own fresh DB/model directory.
- A weak DA3 branch cannot train because the normal production gate evaluates it.
- A failed learned depth stage falls back to the original Ultra depth source and
  records the fallback.
- A failed photometric validation uses original RGB frames.
- Adaptive density stops growth before the VRAM reserve; it does not lower image
  resolution silently.
- CUDA OOM during a batch/chunk-independent learned stage releases that model
  and retries once with a smaller batch/chunk recorded in the report. The DA3
  anchor-camera stage does not reduce its locked anchors or resolution. Training
  OOM stops and publishes diagnostics rather than changing the approved quality
  policy invisibly.
- Local stages publish atomically inside the workspace and include content
  digests.
- Failure may publish only to
  `<input_folder_name>_learned_test_diagnostics/<run_id>`.
- The last good training checkpoint may be copied into diagnostics for salvage,
  but cross-runtime optimizer resume is not claimed in this first experiment.
- An existing learned-test result is replaceable only when its manifest and
  `_SUCCESS` authenticate it as this experiment. Unowned folders are never
  removed.

## 9. Result Contract

A successful learned-test result contains:

```text
<input_folder_name>_learned_test_result/
  splat.ply
  preview.png
  scene_metadata.json
  quality_report.json
  run_manifest.json
  selection_manifest.json
  experiment_report.json
  geometry_candidates.json
  model_manifest.json
  diagnostics/
    masks_contact_sheet.png
    depth_contact_sheet.png
    geometry_contact_sheet.png
    final_render_contact_sheet.png
    density_history.json
    photometric_report.json
  logs/
  _SUCCESS
```

`experiment_report.json` states which candidate won, which learned stages were
accepted/rejected, the final Gaussian count, peak VRAM by stage, and every
fallback. Full per-frame float depth maps are not copied to Drive; their hashes,
statistics, and representative previews are retained. This keeps the result
auditable without adding many gigabytes of intermediate arrays.

## 10. Verification Strategy

### 10.1 Non-GPU automated tests

Tests must cover:

- experiment path normalization and exact `_learned_test_result` derivation;
- notebook order: runtime preflight precedes Drive mount, while source-path
  validation/copy occurs only after the mount;
- isolation: production `RunnerServices`, routes, presets, frontend, and legacy
  notebooks remain unchanged;
- pinned model/dependency manifest and no Torch/NumPy/gsplat upgrade command;
- exact dependency-lock resolution and source/checkpoint revision recording;
- DA3 3x4/4x4 W2C normalization, camera convention, shared-intrinsics creation,
  database image-ID mapping, triangulation, and BA acceptance guards;
- locked DA3 anchor budgets and no quality-reducing anchor OOM fallback;
- same-manifest geometry candidates, exact failed-gate ranking, at most two
  backfill replacements per uncovered interval, deterministic winner tuple, and
  classical fallback;
- rigid-flow projection, forward/backward consistency, uncertainty handling, and
  the rule that unknown is never motion;
- mask fusion, area/runaway rejection, COLMAP/training polarity, and exact mask
  filenames/dimensions;
- robust photometric solve and rejection when held-out error does not improve;
- original RGB for both geometry branches and independently gated learned
  training inputs regardless of which branch wins;
- exact filename joins for learned depth/masks and zero depth on invalid pixels;
- multi-view dense-point consistency and one-million corruption ceiling;
- separate zero-gradient validity tests for L1, SSIM, LPIPS, and depth, plus
  exact legacy outputs/logs when `validity_mask=None`;
- no density-step overshoot, the exact 500/80k/100 Ultra schedule, synchronized
  mocked VRAM reserve, calibrated memory estimate, deterministic top-gradient
  selection, pruning-first order, candidate denominator, and plateau stop;
- custom publisher ownership/rollback while an existing `_result` remains byte
  identical;
- report inventory/checksums and notebook structural Run-All safety.

### 10.2 GPU integration tests

GPU-marked tests use a tiny 3-6-frame fixture to verify real output shapes and
finite values from DA3, DA3Metric, Grounding DINO, SAM 2, and SEA-RAFT. A small
known-camera PyCOLMAP fixture must triangulate shared tracks and pass convention
checks. A 100-iteration masked-training smoke must show zero gradient in excluded
pixels and a portable finite PLY.

These tests are not claimed locally without appropriate CUDA hardware.

### 10.3 Real A100 room acceptance

The user runs the notebook on the same room capture and an A100. Acceptance
requires:

1. a verified `_learned_test_result` while the old `_result` hash inventory is
   unchanged;
2. at least one geometry candidate passes all production gates;
3. the selected candidate and both candidate metric sets are recorded;
4. learned-stage acceptance/fallback decisions are explicit;
5. no density event exceeds the VRAM reserve or 6M ceiling;
6. the final PLY opens in SuperSplat and the project viewer;
7. the mask/depth/final-render contact sheets are visually sane; and
8. the new room result is compared visually with the prior best A100 result.

The notebook does not claim Luma parity or guaranteed improvement before this
real run. The purpose of the isolated experiment is to produce the evidence
needed for the later pipeline decision.

## 11. Explicit Non-Goals

- No frontend or notebook-generator integration.
- No phone capture, ARKit, VIO, or recording-app work.
- No direct feed-forward Gaussian output from DA3.
- No non-commercial model checkpoint.
- No web viewer bundle.
- No silent threshold weakening when geometry fails.
- No full training of both geometry branches; only the winning passing branch
  receives the 120k Ultra optimization.
- No cross-runtime exact optimizer resume in the first experiment.
- No automatic promotion into the production pipeline after a successful run.
