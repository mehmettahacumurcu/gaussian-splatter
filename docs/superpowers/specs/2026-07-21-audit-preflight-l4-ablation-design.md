# Audit-First L4 Training Ablation Design

## Goal

Make the learned-training ablation fail cheaply when its CPU track audit is
incompatible, select an audited cached lineage when one exists, and support the
same diagnostic matrix on an NVIDIA L4 without weakening reconstruction or
training inputs.

## Context

The first ablation run restored more than 12 GiB from Drive and only then
rejected the restored selection because no track-audit receipt matched it. The
cache contains more than one selection address: the immutable CPU-audit
selection and later A100-produced selections. The graph restore currently
chooses the first restorable selection before checking whether its exact image
set has a compatible audit.

The successful 120K room training used materially less than 24 GiB VRAM. The
diagnostic matrix runs variants sequentially in isolated child processes for
only 5K iterations, so an L4 is an appropriate cheaper diagnostic target. A T4
is excluded because its 16 GiB budget does not leave adequate safety margin for
densification spikes.

## Considered Approaches

1. Re-run the full CPU audit before every ablation. This is safe but repeats
   work and does not solve selection-address ambiguity.
2. Relax or bypass exact audit validation. This is fast but unsafe because an
   audit for different frames or COLMAP geometry could authorize training.
3. Prefer an audited selection lineage and validate it before large restores.
   This preserves exact binding, fails early, and reuses the existing verified
   cache. This is the selected approach.

## Design

### Audited lineage selection

Graph restore will evaluate candidate selection milestone references in order.
For each candidate it will:

1. restore and verify only the selection milestone;
2. require a compatible CPU track-audit receipt for the exact input digest,
   image-set digest, qualification policy, producer digest, and a still-present
   COLMAP generation;
3. require a complete Round-0 masks lineage rooted at that selection;
4. only then restore base evidence, semantic maps, motion, masks, COLMAP,
   geometry, and final pretraining artifacts.

An unaudited candidate will be removed locally and the next candidate will be
tried. If no candidate is both audited and complete, the run fails before the
large evidence restore. The post-restore audit check remains as defense in
depth.

### Runtime profiles

`AblationRunSpec` gains a required, explicit runtime profile with two values:

- `l4_diagnostic`: accepts an NVIDIA L4 or A100 with at least 22 GiB VRAM;
- `a100_reference`: accepts an A100 with at least 75 GiB VRAM.

Both profiles run the same experiment variants, iteration counts, images,
masks, depth data, density policy, and evaluation thresholds. The profile only
changes hardware admission; it does not reduce quality settings and therefore
keeps results comparable. Local disk must still have at least 80 GiB free at
notebook preflight and 35 GiB at runner staging.

The generated notebook exposes `RUNTIME_PROFILE` as a dropdown, defaulting to
`l4_diagnostic`, prints the admitted hardware, and writes the selected profile
into the strict run specification and final environment report.

### Failure behavior

- Audit incompatibility occurs before large Drive restoration.
- A failed experiment is recorded and subsequent variants continue in their
  isolated child processes.
- Infrastructure or staging failures stop the matrix and release the runtime.
- Durable cache data is never modified merely to make an incompatible audit
  pass.

## Testing

- Reproduce the bug with two restorable selections where the first is
  unaudited and the second is audited; prove the audited lineage is selected
  before evidence restore.
- Prove no large artifact restore runs when every candidate fails audit.
- Cover L4 admission, undersized/non-L4 rejection, A100 reference admission,
  and profile serialization.
- Regenerate the notebook and smoke-test its parameters, hardware checks,
  immutable commit pin, and execution contract.
- Run the focused ablation suite, integration notebook smoke tests, Ruff, and
  the wider learned-quality cache/milestone/runtime regression suite.
