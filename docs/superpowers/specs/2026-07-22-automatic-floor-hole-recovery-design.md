# Automatic Floor-Hole Recovery Design

**Date:** 2026-07-22

**Status:** Approved for specification; implementation requires written-spec review

**Scope:** A bounded A100 experiment that attempts to recover only automatically
detected missing floor regions. It never starts a 120K run and never overwrites an
existing result.

## Objective

Recover the missing floor in the `myroom` reconstruction without reintroducing the
global dense-seed, semantic-mask, or adaptive-density failures found by the structural
diagnostic matrix.

The experiment must preserve the recognizable `legacy_control` room structure, add
geometry only where an automatically detected floor plane is genuinely unsupported,
and produce enough fixed-view and perturbed-view evidence to decide whether a later
full run is justified.

## Diagnostic Basis

The completed 5K A100 matrix isolated three independent regressions:

- `legacy_control` passed at 28.11 dB with 3.35M Gaussians and 0.80% visible white;
- `depth_only` passed at 27.78 dB and did not repair empty floor topology;
- `dense_seeds_only` reached only 24.27 dB after globally adding 4.03M Gaussians;
- `adaptive_density_only` reached 20.78 dB;
- `masks_only` and `full_learned` collapsed to 6.37 and 7.16 dB, with roughly 20%
  visible white.

The missing floor is therefore not evidence that more global learned supervision is
needed. It is a localized topology problem: sparse reconstruction did not provide
enough floor support, while depth-only training cannot create Gaussians in empty space.

## Non-Goals

- Do not fill walls, ceilings, furniture, or arbitrary missing static surfaces.
- Do not use semantic training-validity masks.
- Do not enable adaptive density control.
- Do not inject the existing global dense-seed artifact into training.
- Do not run more than 5,000 optimization iterations.
- Do not publish a production PLY or authorize a 120K run.
- Do not fall back to global seeding when floor detection is uncertain.

## Approaches Considered

### Reuse global dense seeds with a lower cap

This is inexpensive to implement, but the diagnostic showed that global dense seeds
damage quality independently. A smaller global cap reduces the magnitude of the same
failure without restricting it to the missing floor.

### Use semantic floor segmentation

A semantic model could propose floor pixels directly, but the mask-only diagnostic was
the most destructive isolated feature. It also makes recovery depend on category
classification when the required property is geometric support.

### Geometry-first floor-hole seeding

Fit a reliable walkable plane from the accepted sparse reconstruction, locate holes in
that plane inside a multi-view-observed footprint, and fuse only depth candidates that
land near the plane and pass independent view checks. This directly targets the failure
while keeping the successful legacy training behavior. Selected.

## Experiment Shape

The notebook stages the verified selection, accepted geometry, final pretraining
evidence, and prior diagnostic report from Drive once. All reconstruction, seed
generation, training, rendering, and evaluation then use local disk.

Only one new training arm runs:

| Setting | Value |
|---|---|
| Initialization | accepted sparse model plus verified floor-hole seeds |
| RGB loss | legacy control behavior |
| Depth loss | validated safe depth behavior from `depth_only` |
| Training validity masks | disabled |
| Adaptive density | disabled |
| Density controller | standard legacy controller |
| Iterations | 5,000 |
| Historical reference | preserved `legacy_control` 5K diagnostic row |

The historical reference is accepted only when its selection fingerprint, camera-model
fingerprint, resolution, random seed, checkpoint schedule, and training configuration
match the new experiment. A mismatch stops before training instead of producing a
misleading comparison.

## Floor-Plane Estimation

A new learned-quality floor module consumes the exact accepted sparse points and
registered cameras used for training. It must not infer the floor from the learned PLY.

1. Estimate the raw-frame up direction using the existing deterministic walkable-plane
   orientation logic.
2. Generate RANSAC plane candidates from the finite, robust-core sparse points.
3. Reject planes whose normal differs from the estimated up axis by more than 15
   degrees.
4. Orient every candidate so registered camera centres lie on its positive side.
5. Require at least 100 inliers, at least 0.5% sparse-point support, an above/below content
   ratio of at least 4, and a positive median camera height greater than four plane
   inlier tolerances.
6. Refine the best candidate by weighted least squares and record its normal, offset,
   inlier residual distribution, camera-height distribution, support, and confidence.

The inlier tolerance is 2% of the robust scene radius. The estimator is deterministic
under the experiment seed. If no candidate satisfies every condition, floor recovery
is reported as unavailable and no training starts.

## Observed Floor Footprint and Hole Map

The plane receives a deterministic orthonormal two-dimensional basis. Validated
pose-conditioned depth pixels are unprojected into world space and retained as floor
candidates only when all of the following hold:

- depth, confidence, camera, and RGB values are finite and in bounds;
- confidence is at least 0.20;
- the world point lies within the larger of one plane tolerance or 3% of its depth from
  the floor plane;
- the source pixel is not confirmed sky and not confirmed motion;
- reprojection into at least two additional registered cameras lands in bounds and
  agrees with their validated depth within 5% relative error.

Semantic class, semantic uncertainty, and general training-validity masks are not used
for this decision.

Project the surviving candidates and sparse floor inliers into the plane basis. The grid
cell width is 1% of the robust scene radius. A cell belongs to the observed floor
footprint only if it has supported depth evidence and is inside the eroded multi-view
footprint; one-cell erosion prevents edge extrapolation. A footprint cell is a hole only
when no accepted sparse point lies within 1.5 cells.

Disconnected hole components smaller than four cells are discarded. Candidates
outside accepted hole components are never emitted as seeds.

## Floor-Seed Fusion

Within accepted hole cells:

- fuse candidates in deterministic voxels of one-half grid-cell width;
- use the support-weighted median position projected back onto the refined floor plane;
- use a robust median source RGB color;
- record confidence, total view support, source-frame IDs, plane residual, and hole-cell
  ID;
- rank by view support, confidence, and distance from existing sparse support;
- cap the artifact at 150,000 seeds.

At least 1,000 fused seeds and at least one accepted hole component are required to run
training. Fewer seeds are reported as `no_recoverable_floor_hole`, not padded or
expanded. The cap is a hard ceiling; exceeding it triggers deterministic ranking rather
than random sampling.

The typed artifact contains the seed arrays, plane and grid metadata, input digests,
policy values, candidate/rejection counts, hole statistics, and a content fingerprint.
Every path must belong to one artifact root.

## Safe Gaussian Initialization

The floor seeds are appended after the original accepted sparse points in a fresh local
COLMAP text model. The point parser preserves file order. The experiment disables
initial point subsampling and records the original and appended counts.

When the trainer is constructed, its customizer must assert:

- initialized Gaussian count equals original sparse count plus floor-seed count;
- the trailing slice has exactly the floor-seed count;
- the initialized trailing means and colors match the artifact within float tolerance;
- no non-finite model tensor exists.

Only after those assertions pass, the trailing floor-seed slice is initialized with:

- opacity logit `-4.0` (approximately 1.8% opacity);
- scale clamped to at most one-quarter of the floor grid-cell width;
- identity rotation and zero higher-order SH coefficients.

The original sparse Gaussian slice is untouched. If point order, count, or values do not
match, training aborts before the first iteration. There is no implicit fallback to the
default high-opacity initialization.

## Training and Evaluation

The experiment uses the same deterministic frame order, seed, 720-pixel resolution,
evaluation cameras, and checkpoint schedule as the accepted 5K diagnostic. It records
checkpoints at iterations 0, 100, 499, 500, 600, 1,000, 2,500, and 5,000.

Global measurements remain identical to the diagnostic matrix: unmasked PSNR, SSIM,
L1, edge correlation, alpha and depth coverage, perturbed-view consistency, Gaussian
parameter distributions, visible-white fraction, oversized fraction, out-of-bounds
fraction, and non-finite count.

Floor-specific measurements are computed only inside the automatically detected plane
footprint:

- rendered floor alpha coverage;
- residual floor-hole area;
- depth agreement with the fitted plane;
- cross-view floor depth consistency;
- seeded-cell survival and contribution;
- coverage split between originally supported and recovered cells.

Contact sheets show, for the same cameras, ground truth, legacy reference, floor
recovery, alpha, depth, floor footprint, original holes, added seeds, and remaining
holes. Perturbed-view sheets are required so a flat camera-facing patch cannot pass as
recovered 3D floor.

## Pass and Stop Rules

The floor-recovery arm passes only when all conditions hold at iteration 5,000:

- floor-hole area is reduced by at least 25% relative to initialization;
- floor alpha coverage improves by at least 10 percentage points;
- unmasked PSNR is no more than 1.0 dB below matched `legacy_control`;
- visible-white fraction is at most 1.5%;
- oversized fraction is at most 1.0%;
- out-of-bounds fraction is at most 0.1%;
- non-finite count is zero;
- perturbed-view floor depth disagreement does not exceed the legacy reference by more
  than 10%.

Training stops immediately for non-finite tensors, CUDA out-of-memory, seed-slice
contract failure, or inability to render diagnostic evidence. Other quality failures
are recorded through 5K so the report can distinguish insufficient recovery from a new
structural regression.

A pass means only that a separate full-run proposal may be considered. The notebook
does not contain a 120K execution path.

## Cache and Publication

Floor-plane, hole-map, and seed artifacts use new schema versions and fingerprints that
include the exact source revision, selection, accepted model, camera model, depth,
motion, sky, and policy digests. Existing global dense-seed or pretraining milestones
cannot satisfy this artifact contract.

The result publishes atomically to:

```text
<input>_floor_recovery_diagnostic/
  _SUCCESS.json
  decision.json
  floor_plane.json
  floor_holes.json
  floor_seeds.npz
  metrics.csv
  comparison.json
  diagnostic_summary.md
  plots/
  contact_sheets/
  checkpoints/
  candidate.ply
```

`_SUCCESS.json` means the requested diagnostic completed and its evidence was published;
the recovery decision is the separate `decision.json.pass` value. `candidate.ply` is
present only when training reaches a renderable checkpoint and is an inspection
artifact, not a production result. The notebook cannot write to `<input>_result`,
`<input>_learned_test_result`, or the structural diagnostic folder.

## Failure Handling

Every failure publishes a small receipt with the completed stage, reason, input
fingerprints, and any already-produced diagnostic evidence. The notebook flushes Drive
and releases the runtime on success, rejection, or failure.

No condition broadens the target from floor-only to global completion. A rejected floor
plane, empty hole map, insufficient multi-view support, or failed pass gate ends the
experiment safely.

## Verification

CPU tests cover deterministic plane fitting, camera-side orientation, fail-closed plane
criteria, floor projection, footprint erosion, sparse-hole classification, motion/sky
exclusion, three-view support, deterministic fusion and capping, typed artifact
round-trips, seed-slice alignment, low-opacity/small-scale initialization, and cache
fingerprints.

Integration tests use synthetic rooms with a known missing floor patch. They prove that
seeds are emitted only inside the patch, semantic masks do not influence selection, a
weak plane produces no training, original sparse initialization is unchanged, and a
point-order mismatch fails before iteration zero.

Notebook contract tests enforce A100 High-RAM preflight, one-time Drive staging, pinned
source revision, bounded 5K training, diagnostic continuation and publication,
automatic runtime release, no production-path overwrite, and absence of any 120K path.
