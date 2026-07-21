# A100 Structural Diagnostic Matrix Implementation Plan

> Execute this plan test-first. The notebook's terminal state is a published diagnostic matrix; no task adds a 120K launch path.

**Goal:** Diagnose both Gaussian instability and unrecognizable room structure across a bounded A100 training matrix, using one local cache restore and the preserved 120K run as historical evidence.

**Architecture:** Extend the standalone ablation harness rather than the production learned runner. Add an iteration-zero trainer hook with unchanged defaults, a seven-row fixed 5K matrix, phase-aware checkpoint decisions, fixed/perturbed camera diagnostics, historical-result analysis, and a versioned report/notebook contract.

**Tech stack:** Python 3.12, PyTorch, gsplat renderer, NumPy, Pillow, nbformat, pytest, Google Colab/Drive.

---

## Task 1: Version the forensic matrix contract

**Files:**
- Modify: `experiments/learned_quality/ablation.py`
- Modify: `tests/experiments/learned_quality/test_ablation.py`

**Steps:**
1. Add failing tests for the `fixed_topology_control` row, 0/100/499/500/600/1000/2500/5000 checkpoints, and removal of automatic pairwise execution.
2. Add phase-aware structural fields and persistence-based decisions. Non-finite tensors remain the only ordinary immediate stop; severe emergency thresholds remain versioned.
3. Add diagnosis inputs for initialization failure, density-transition failure, feature failures, weak 3D consistency, and inconclusive results.
4. Run the focused contract tests.

## Task 2: Observe initialization without changing production training

**Files:**
- Modify: `backend/model/trainer.py`
- Modify: `tests/model/test_trainer_diagnostics.py`

**Steps:**
1. Add a failing test proving diagnostic iteration 0 is called before optimizer or density work and that existing callers without diagnostics are unchanged.
2. Permit zero in the diagnostic schedule and invoke the callback once after scene/camera preparation and before the training loop.
3. Preserve the existing post-density callback ordering for iterations 1..N.
4. Run the trainer diagnostic tests.

## Task 3: Add structural-fidelity measurements

**Files:**
- Modify: `experiments/learned_quality/ablation_training.py`
- Modify: `tests/experiments/learned_quality/test_ablation_training.py`

**Steps:**
1. Add failing CPU tests for edge metrics, spatial extent/eigenvalue metrics, out-of-envelope fractions, phase-aware checkpoints, and the fixed-topology training spec.
2. Extend fixed-view rendering to save ground truth, RGB, error, alpha/depth diagnostics and aggregate PSNR/SSIM/L1/LPIPS availability/edge metrics.
3. Render deterministic small pose translations and record perturbed-view alpha/depth coverage plus contact sheets.
4. Capture density-event deltas from the trainer/run logger where available without changing production logging.
5. Keep LPIPS optional and explicit: unavailable is `null`, never a fabricated zero.
6. Run the focused training tests.

## Task 4: Analyze the preserved 120K reference

**Files:**
- Create: `experiments/learned_quality/ablation_history.py`
- Create: `tests/experiments/learned_quality/test_ablation_history.py`
- Modify: `experiments/learned_quality/ablation_staging.py`
- Modify: `tests/experiments/learned_quality/test_ablation_staging.py`

**Steps:**
1. Add fixtures for ASCII and binary little-endian Gaussian PLY headers/properties, reports, and density history.
2. Implement streaming header/property analysis with bounded memory and graceful unavailable fields.
3. Summarize density/reset cycles, final counts, scene bounds, scale/opacity/SH distributions, geometry failures, and preserved report metadata.
4. Stage the historical learned-result folder once when present; absence is non-fatal and recorded.
5. Verify that experiment subprocesses read only local staged files.

## Task 5: Run every primary row and publish a diagnostic matrix

**Files:**
- Modify: `experiments/learned_quality/ablation_runner.py`
- Modify: `tests/experiments/learned_quality/test_ablation_runner.py`

**Steps:**
1. Add failing tests that all seven rows are attempted, one row error does not stop later rows, pairwise rows never launch, and the report ends after the matrix.
2. Publish per-row durable receipts and consolidate checkpoint metrics, structural diagnoses, historical reference data, plots, and contact sheets.
3. Rename/version the report contract while retaining compatibility aliases for existing partial diagnostic folders where safe.
4. Require `_SUCCESS.json` only when all seven requested rows produced a result or recorded diagnostic failure; infrastructure staging failure remains fatal.
5. Run the runner tests.

## Task 6: Generate the A100-only stop-after-matrix notebook

**Files:**
- Modify: `experiments/learned_quality/ablation_notebook.py`
- Modify: `tests/experiments/learned_quality/test_ablation_cli.py`
- Modify: `tests/experiments/learned_quality/test_ablation_runner.py`
- Regenerate: `colab/learned_quality_training_ablation.ipynb`

**Steps:**
1. Add failing notebook tests for A100 >=75 GiB, one Drive restore boundary, stage/checkpoint output, required matrix artifacts, runtime release, and absence of `120000`/production-run launch code.
2. Make `a100_reference` the only accepted notebook profile and explain the seven rows/checkpoints in the title cell.
3. Print live row/checkpoint progress and the final evidence-backed diagnosis.
4. Regenerate the notebook from the post-implementation commit pin.
5. Run notebook/CLI tests.

## Task 7: Full verification and publication

**Files:**
- Verify all modified files and generated notebook.

**Steps:**
1. Run focused ablation, trainer, staging, CLI, and notebook tests.
2. Run the broader learned-quality test suite.
3. Run formatting/static checks already configured by the repository.
4. Inspect `git status`, staged diff, and generated notebook pin.
5. Commit with a conventional message and push `feature/learned-quality-a100`.
6. Report the notebook link, exact source pin, expected artifacts, and step-by-step A100 usage.
