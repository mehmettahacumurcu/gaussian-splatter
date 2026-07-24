# Legacy-Control Native-1080p 30K Local Run Implementation Plan

> Execute this plan test-first on `feature/learned-quality-a100`.

**Goal:** Generate an A100 Colab notebook that restores the verified
pre-training lineage once, runs the successful legacy-control configuration
continuously at native 1920x1080 through 30K, and keeps raw local PLY and
resumable checkpoint snapshots every 5K without reports, polish, Drive result
publication, or runtime release.

**Architecture:** Add a focused long-run module beside the existing 5K export
module. Reuse the verified cache/audit staging and legacy-control scene
preparation, but install a lightweight trainer diagnostic callback that writes
atomic raw PLY and full training checkpoints at six boundaries. A lazy CLI
executes the run without durable receipts, and a generated pinned notebook
performs A100 preflight, mounts Drive for input staging, launches the CLI, and
leaves the session connected.

**Stack:** Python 3.12, Pydantic, PyTorch, nbformat, pytest.

---

## Task 1: Lock the long-run configuration contract

**Files:**

- Create: `tests/experiments/learned_quality/test_legacy_control_long_run.py`
- Create: `experiments/learned_quality/legacy_control_long_run.py`

**Test first:**

- validate and normalize the input-folder-only run spec;
- require A100 80 GB, High-RAM, and sufficient local disk;
- derive the long run from the passing 5K legacy-control spec;
- prove the only resolved experiment deltas are 30K duration, native 1920x1080
  resolution, and an empty multi-resolution schedule;
- prove density remains 500-4500, opacity reset remains 3000, six-million cap
  remains, and learned-quality variables remain disabled;
- reject any accepted camera set that is not exactly 1920x1080.

Run the focused test and confirm it fails because the module is absent. Then
implement only the spec, hardware, variant, and resolution helpers required to
make it pass.

## Task 2: Add atomic 5K PLY and resumable checkpoint snapshots

**Files:**

- Modify: `tests/experiments/learned_quality/test_legacy_control_long_run.py`
- Modify: `experiments/learned_quality/legacy_control_long_run.py`

**Test first:**

- use a fake trainer/exporter/saver to assert exact filenames at
  5K/10K/15K/20K/25K/30K;
- assert the callback rejects any other iteration or resolution;
- assert PLY and checkpoint writes use temporary sibling paths and become
  visible only after success;
- assert a later failure preserves all earlier snapshots;
- assert checkpoint payload includes Gaussian state, optimizer state, iteration,
  camera generator state, global RNG state, scene extent, and SH degree.

Implement a `LocalSnapshotWriter` callback using `backend.export.to_splat` and
atomic `os.replace`. Do not write JSON reports or markers.

## Task 3: Run one continuous trainer from one local staging tree

**Files:**

- Modify: `tests/experiments/learned_quality/test_legacy_control_long_run.py`
- Modify: `experiments/learned_quality/legacy_control_long_run.py`

**Test first:**

- assert verified staging is called once;
- assert no historical reference, report, polish, or publisher is called;
- assert one pipeline/trainer invocation owns all six callbacks;
- assert training paths are outside Drive;
- assert the returned result contains all six local PLY and checkpoint paths;
- assert the local run tree remains on callback or training failure.

Implement:

- an exact legacy-control pipeline adapter using
  `prepare_ablation_scene`;
- one deterministic CPU camera generator seeded with 1701;
- `run_legacy_control_long_from_staged`;
- `run_legacy_control_long`, reusing exact CPU audit and final pre-training
  lineage verification from the existing experiment stack;
- local-only work root `/content/legacy_control_1080p_30k`.

## Task 4: Add the lazy local-only CLI

**Files:**

- Create:
  `tests/experiments/learned_quality/test_legacy_control_long_run_cli.py`
- Create: `scripts/learned_quality_legacy_control_long_run.py`

**Test first:**

- `--help` must not import Torch;
- invalid specs return 2 without creating a result receipt;
- success returns 0 and prints the local run root plus six PLY/checkpoint paths;
- failure returns 1, prints a traceback, writes no Drive failure artifact, and
  preserves the local root.

Implement a lazy loader and plain stdout progress. Do not create a receipt JSON.

## Task 5: Generate the pinned A100 notebook

**Files:**

- Create:
  `tests/integration/test_legacy_control_long_run_notebook_smoke.py`
- Create:
  `experiments/learned_quality/legacy_control_long_run_notebook.py`
- Create: `colab/learned_quality_legacy_control_1080p_30k.ipynb`

**Test first:**

- assert a single `INPUT_FOLDER` form field and fixed runtime profile;
- assert A100/High-RAM/local-disk preflight;
- assert one Drive mount and no Drive result path;
- assert pinned immutable checkout and verified learned environment;
- assert execution invokes the new CLI;
- assert sources contain no `polish`, `report`, `receipt`,
  `drive.flush_and_unmount`, `runtime.unassign`, or automatic cleanup;
- assert a monitoring cell lists snapshot PLYs/checkpoints, tails trainer
  metrics, and displays GPU usage;
- assert the checked-in notebook exactly matches its generator.

Generate the notebook pinned to the implementation commit.

## Task 6: Verify and publish the implementation branch

Run:

```text
pytest -q tests/experiments/learned_quality/test_legacy_control_long_run.py
pytest -q tests/experiments/learned_quality/test_legacy_control_long_run_cli.py
pytest -q tests/integration/test_legacy_control_long_run_notebook_smoke.py
pytest -q tests/experiments/learned_quality/test_ablation_training.py
pytest -q tests/experiments/learned_quality/test_legacy_control_export.py
pytest -q tests/integration/test_legacy_control_export_notebook_smoke.py
```

Then run `git diff --check`, review staged scope, commit with a conventional
message explaining why the bounded local-only run exists, push the feature
branch, regenerate the notebook at the committed SHA, rerun notebook
determinism, and commit/push the pin update.
