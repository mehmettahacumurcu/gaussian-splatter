# Training Ablation Notebook Implementation Plan

> **For agentic workers:** Execute this plan in order. Keep the existing legacy and learned-quality notebooks behavior-compatible, write the failing test before every production change, and commit each verified milestone to `feature/learned-quality-a100`.

**Goal:** Build a standalone A100 Colab notebook that restores the verified learned-quality inputs from Drive once, runs deterministic short Gaussian-training ablations from the same local scene, diagnoses the quality regression, and publishes a compact report without replacing the normal splat result.

**Architecture:** Add a pure ablation policy layer, a one-time staging layer, an experiment runner that parameterizes the four learned training inputs, and a parent orchestrator that invokes every experiment in a fresh subprocess. Extend the trainer through optional, default-off diagnostic seams so fixed-camera checkpoints and structural metrics can be collected without changing existing callers. Generate the notebook from Python and preserve the existing notebook generators unchanged.

**Tech stack:** Python 3.12, PyTorch, NumPy, PIL, pytest, Google Colab, existing 4DGS pipeline and learned-quality checkpoint store.

## Global constraints

- Do not alter the legacy notebook pipeline or the result contract of `learned_quality_a100_experiment.ipynb`.
- Restore the selection and final pre-training lineage once into `/content/4dgs-ablation/<run_id>/inputs`; experiments may not read their training payload through `/content/drive`.
- Use the same cameras, image order, fixed evaluation views, seed, resolution, and checkpoints for every experiment.
- Keep large PLY/checkpoint artifacts local. Publish receipts, reports, CSV/Markdown, plots, and contact sheets only.
- Treat `_SUCCESS.json` as “diagnostic matrix published,” never as a quality claim.
- Stop an experiment on non-finite values or a confirmed structural/PSNR gate failure; continue the matrix and record the failure.
- Flush Drive and request runtime release on success and failure.

---

### Task 1: Define the ablation matrix, gates, and diagnosis classifier

**Files:**
- Create: `experiments/learned_quality/ablation.py`
- Create: `tests/experiments/learned_quality/test_ablation.py`

- [ ] Write tests for the exact primary matrix: `legacy_control`, four one-feature variants, and `full_learned`, all at 5,000 iterations and checkpoints 500/1,000/2,500/5,000.
- [ ] Write tests for the six deterministic two-feature variants at 2,500 iterations.
- [ ] Write tests for the `ablation-v1` gates: non-finite abort, visible-white threshold, anisotropy threshold, scene-relative scale threshold, two consecutive PSNR deficits, and unusable control below 15 dB.
- [ ] Write classification tests for `isolated_cause`, `interaction_cause`, `multiple_independent_causes`, and `inconclusive`.
- [ ] Run `pytest -q tests/experiments/learned_quality/test_ablation.py` and confirm it fails because the module does not exist.
- [ ] Implement immutable dataclasses `AblationVariant`, `StructuralMetrics`, `AblationCheckpoint`, `AblationGatePolicy`, `GateDecision`, and `AblationDiagnosis` plus pure functions `primary_variants()`, `pairwise_variants()`, `evaluate_checkpoint()`, and `classify_experiments()`.
- [ ] Run the focused test until green, then run `pytest -q tests/experiments/learned_quality/test_ablation.py tests/experiments/learned_quality/test_training.py`.
- [ ] Commit and push: `feat: define learned training ablations`.

### Task 2: Add deterministic, default-off training diagnostics

**Files:**
- Modify: `backend/model/trainer.py`
- Modify: `backend/pipeline.py`
- Create: `tests/model/test_training_diagnostics.py`

- [ ] Write tests proving an explicitly supplied `torch.Generator` controls camera sampling, callbacks fire exactly at requested iterations, and the default call path remains unchanged.
- [ ] Write a pipeline validation test proving only the new diagnostic kwargs are accepted and arbitrary kwargs remain rejected.
- [ ] Run `pytest -q tests/model/test_training_diagnostics.py tests/model/test_density_extension_seam.py` and confirm the new tests fail.
- [ ] Add optional `camera_generator`, `diagnostic_iterations`, and `diagnostic_callback` keyword arguments to `Trainer4DGS.train`; use the generator only for camera-index draws and invoke the callback after a completed training step at exact checkpoints.
- [ ] Extend the pipeline train-kwargs allowlist for those three keys without changing defaults.
- [ ] Run the focused tests and existing model seam tests.
- [ ] Commit and push: `feat: add deterministic training diagnostics`.

### Task 3: Stage verified inputs once and forbid Drive-backed experiment payloads

**Files:**
- Create: `experiments/learned_quality/ablation_staging.py`
- Create: `tests/experiments/learned_quality/test_ablation_staging.py`

- [ ] Write fake-store tests that count restore calls and prove selection plus final pre-training evidence are restored exactly once.
- [ ] Write tests for cache ownership, CPU-audit receipt, source-pin, free-disk, and manifest/hash failures.
- [ ] Write a test that materializes two experiment workspaces and proves both reference the same immutable local input tree while mutable output/COLMAP text is private.
- [ ] Write a test that rejects any staged artifact whose resolved path is under `/content/drive`.
- [ ] Run the focused test and confirm failure.
- [ ] Implement `StagedAblationInputs`, `stage_ablation_inputs()`, `validate_staged_inputs()`, and `materialize_experiment_workspace()` using the existing `LearnedCheckpointStore.restore_pretraining` contract.
- [ ] Make staged files read-only after validation and expose a local-only manifest containing resolved paths and SHA-256 identities.
- [ ] Run the focused test and cache/milestone tests.
- [ ] Commit and push: `feat: stage ablation inputs once`.

### Task 4: Build a parameterized short-training experiment runner

**Files:**
- Create: `experiments/learned_quality/ablation_training.py`
- Create: `tests/experiments/learned_quality/test_ablation_training.py`

- [ ] Write scene-construction tests proving `legacy_control` uses no learned dense seeds, masks, depth, or adaptive density and each isolated variant changes only its named feature.
- [ ] Write tests proving all variants share the exact camera join/order, RGB inputs, seed, 720p resolution, fixed evaluation frames, and initial sparse geometry.
- [ ] Write structural-metric tests for opacity, SH/DC, scale, anisotropy, pure-white fraction, non-finite counts, and robust scene extent.
- [ ] Write checkpoint-output tests for masked/unmasked PSNR, SSIM, L1, loss components, point counts, and contact sheets.
- [ ] Run the focused test and confirm failure.
- [ ] Implement `AblationTrainingSpec`, `PreparedAblationScene`, `prepare_ablation_scene()`, `collect_structural_metrics()`, `render_fixed_views()`, and `run_ablation_experiment()`.
- [ ] Reuse the accepted local reconstruction and existing training helpers; use fresh process/output directories and a dedicated seeded camera generator.
- [ ] Enable dense-seed augmentation, learned validity masks, learned depth supervision, and adaptive density independently. Use legacy density control when adaptive density is disabled.
- [ ] Stop only the current experiment on a gate failure and atomically write its receipt/metrics/contact sheets locally.
- [ ] Run focused tests plus `tests/experiments/learned_quality/test_training.py` and the static pipeline training tests.
- [ ] Commit and push: `feat: run controlled training ablations`.

### Task 5: Orchestrate primary/pairwise runs and publish a compact report

**Files:**
- Create: `experiments/learned_quality/ablation_runner.py`
- Create: `tests/experiments/learned_quality/test_ablation_runner.py`

- [ ] Write orchestration tests proving primary variants run sequentially in fresh subprocesses, a tiny Drive progress receipt is published after each result, and staged inputs are not restored again.
- [ ] Write tests proving pairwise runs are scheduled only when isolated results cannot explain a failing full-learned variant.
- [ ] Write atomic-publication tests for `ablation_report.json`, `ablation_summary.md`, `metrics.csv`, plots/contact sheets, per-experiment receipts, environment/staging diagnostics, and `_SUCCESS.json`.
- [ ] Write failure-path tests proving partial receipts survive and the output never claims a diagnostic success before the complete required matrix is present.
- [ ] Run the focused test and confirm failure.
- [ ] Implement `AblationRunSpec`, `run_training_ablation()`, subprocess invocation, diagnosis aggregation, plot/CSV/Markdown generation, ownership markers, and atomic Drive publication.
- [ ] Ensure large experiment PLY/checkpoint files are removed after measurements and never copied to Drive.
- [ ] Run focused tests and learned-quality runner tests.
- [ ] Commit and push: `feat: orchestrate training ablation report`.

### Task 6: Add the CLI and resilient run receipt

**Files:**
- Create: `scripts/learned_quality_ablation_run.py`
- Create: `tests/scripts/test_learned_quality_ablation_run.py`

- [ ] Write CLI tests for a valid spec, invalid input path, child-experiment dispatch, success receipt, and exception receipt.
- [ ] Run the focused test and confirm failure.
- [ ] Implement parent mode reading `/content/learned_ablation_spec.json`, child mode for one isolated experiment, unbuffered progress output, and `/content/learned_ablation_result.json` receipt writing.
- [ ] Preserve diagnostics/progress output on exceptions and return non-zero without swallowing the cause.
- [ ] Run the focused tests.
- [ ] Commit and push: `feat: add training ablation CLI`.

### Task 7: Generate the standalone A100 Colab notebook

**Files:**
- Create: `experiments/learned_quality/ablation_notebook.py`
- Create: `tests/integration/test_learned_quality_ablation_notebook_smoke.py`
- Generate: `colab/learned_quality_training_ablation.ipynb`

- [ ] Write a notebook smoke test asserting the full immutable Git SHA, A100/VRAM/disk preflight, one Drive mount, input-folder validation, output suffix `_training_ablation`, CLI invocation, result verification, Drive flush, and runtime unassignment.
- [ ] Assert the notebook contains no normal splat-result publication and no web viewer.
- [ ] Run the smoke test and confirm failure.
- [ ] Implement a Python notebook generator following the existing learned-quality notebook style without modifying the existing generator.
- [ ] Generate the notebook mechanically and rerun the smoke test.
- [ ] Run both existing learned notebook smoke tests to prove compatibility.
- [ ] Commit and push: `feat: add A100 training ablation notebook`.

### Task 8: Document usage and verify the complete feature

**Files:**
- Modify: `README.md`
- Modify: `experiments/learned_quality/README.md`

- [ ] Document that the notebook is diagnostic, requires the verified CPU audit/cache, stages Drive data once, runs primary then conditional pairwise experiments, and publishes `<input>_training_ablation`.
- [ ] Document exact Colab steps and explain `_SUCCESS.json` semantics.
- [ ] Run formatting/lint commands used by the repository for touched Python files.
- [ ] Run the focused suite:
  `pytest -q tests/experiments/learned_quality/test_ablation.py tests/experiments/learned_quality/test_ablation_staging.py tests/experiments/learned_quality/test_ablation_training.py tests/experiments/learned_quality/test_ablation_runner.py tests/scripts/test_learned_quality_ablation_run.py tests/model/test_training_diagnostics.py tests/integration/test_learned_quality_ablation_notebook_smoke.py`.
- [ ] Run regression suites:
  `pytest -q tests/experiments/learned_quality/test_training.py tests/model/test_density_extension_seam.py tests/integration/test_learned_quality_notebook_smoke.py tests/integration/test_learned_quality_audit_notebook_smoke.py`.
- [ ] Run the repository’s broader relevant test command and record any unrelated pre-existing failure separately.
- [ ] Inspect `git status` and `git diff --staged`, then commit and push: `docs: explain training ablation workflow`.
- [ ] Verify the remote branch contains every milestone and report the notebook path plus the exact user run sequence.
