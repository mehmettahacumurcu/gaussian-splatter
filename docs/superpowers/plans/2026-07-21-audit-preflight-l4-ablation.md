# Audit-First L4 Training Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select an exactly audited cached lineage before large Drive restoration and allow the unchanged diagnostic matrix to run on an NVIDIA L4.

**Architecture:** Move exact audit admission into candidate selection during graph restoration, while retaining the existing post-restore validation. Add an explicit runtime profile to the strict ablation spec and use one shared hardware-policy function from both the runner tests and generated notebook contract.

**Tech Stack:** Python 3.12, Pydantic, pytest, nbformat, Google Colab, NVIDIA L4/A100.

## Global Constraints

- Never bypass or weaken exact CPU track-audit validation.
- Validate selection compatibility before restoring base evidence, motion, masks, geometry, or final pretraining data.
- `l4_diagnostic` and `a100_reference` use the same experiment matrix and quality inputs.
- T4 and GPUs below 22 GiB VRAM remain unsupported.
- Preserve automatic Drive flush and Colab runtime release.

---

### Task 1: Select an audited graph lineage before evidence restore

**Files:**
- Modify: `experiments/learned_quality/ablation_staging.py`
- Modify: `experiments/learned_quality/ablation_runner.py`
- Test: `tests/experiments/learned_quality/test_ablation_staging.py`
- Test: `tests/experiments/learned_quality/test_ablation_runner.py`

**Interfaces:**
- Consumes: `selection_refs: tuple[MilestoneRef, ...]` and an exact selection validator.
- Produces: `restore_output_first_pretraining(..., selection_validator: Callable[[object], None] | None)` that tries candidates until one is both audited and graph-complete.

- [ ] **Step 1: Write the failing audited-candidate regression test**

Add a test with two selection references. Make the first selection validator raise the production audit error and make the second pass. Assert that no non-selection milestone is restored for the rejected candidate and that the returned reconstruction uses the second selection.

- [ ] **Step 2: Run the regression test and verify RED**

Run:

```bash
pytest -q tests/experiments/learned_quality/test_ablation_staging.py -k audited
```

Expected: failure because `selection_validator` is not accepted and the first selection is chosen unconditionally.

- [ ] **Step 3: Implement candidate admission in graph restoration**

Extend `restore_output_first_pretraining` with the optional validator. For each selection reference, restore selection into a clean candidate directory, invoke the validator, and require `find_latest_complete_lineage`. Remove a rejected candidate directory before trying the next reference. Raise the collected audit error when candidates exist but none are audited; otherwise return `None` when no complete graph exists.

- [ ] **Step 4: Wire exact audit validation before large restoration**

Pass a closure from `run_training_ablation` into `restore_output_first_pretraining`:

```python
def validate_selection(selection: object) -> None:
    audit_exact(cache_root, selection, inventory)
```

Keep `restored_validator=validate_restored` as defense in depth.

- [ ] **Step 5: Run focused staging and runner tests**

Run:

```bash
pytest -q tests/experiments/learned_quality/test_ablation_staging.py tests/experiments/learned_quality/test_ablation_runner.py
```

Expected: all tests pass.

### Task 2: Add explicit L4 and A100 runtime profiles

**Files:**
- Modify: `experiments/learned_quality/ablation_runner.py`
- Modify: `scripts/learned_quality_ablation_run.py`
- Test: `tests/experiments/learned_quality/test_ablation_runner.py`
- Test: `tests/experiments/learned_quality/test_ablation_cli.py`

**Interfaces:**
- Consumes: `AblationRunSpec.runtime_profile`.
- Produces: `validate_ablation_hardware(profile, hardware)` and environment metadata containing `runtime_profile`.

- [ ] **Step 1: Write failing runtime-profile tests**

Cover strict parsing and these hardware cases:

```python
L4 24 GiB + l4_diagnostic -> accepted
A100 80 GiB + l4_diagnostic -> accepted
T4 16 GiB + l4_diagnostic -> rejected
L4 24 GiB + a100_reference -> rejected
A100 80 GiB + a100_reference -> accepted
```

- [ ] **Step 2: Run profile tests and verify RED**

Run:

```bash
pytest -q tests/experiments/learned_quality/test_ablation_runner.py tests/experiments/learned_quality/test_ablation_cli.py -k 'profile or hardware'
```

Expected: failure because `runtime_profile` and the shared policy do not exist.

- [ ] **Step 3: Implement the minimal strict profile policy**

Add:

```python
RuntimeProfile = Literal["l4_diagnostic", "a100_reference"]
runtime_profile: RuntimeProfile = "l4_diagnostic"
```

Implement hardware admission with 22 GiB for L4/A100 diagnostic mode and 75 GiB plus A100 identity for reference mode. Include the profile in the report environment.

- [ ] **Step 4: Run profile and CLI tests**

Run the command from Step 2. Expected: all selected tests pass.

### Task 3: Regenerate an L4-capable, audit-first notebook

**Files:**
- Modify: `experiments/learned_quality/ablation_notebook.py`
- Regenerate: `colab/learned_quality_training_ablation.ipynb`
- Modify: `docs/TRAINING_ABLATION_NOTEBOOK.md`
- Modify: `experiments/learned_quality/README.md`
- Test: `tests/integration/test_learned_quality_ablation_notebook_smoke.py`

**Interfaces:**
- Consumes: `INPUT_FOLDER` and `RUNTIME_PROFILE` Colab form parameters.
- Produces: strict JSON with `runtime_profile`, early hardware admission, and the existing safe execute/finally cell.

- [ ] **Step 1: Write failing notebook smoke assertions**

Require a dropdown cell containing:

```python
RUNTIME_PROFILE = "l4_diagnostic"  # @param ["l4_diagnostic", "a100_reference"]
```

Assert the preflight accepts L4 for diagnostic mode, retains the A100 reference check, serializes the profile, and still pins a full commit SHA.

- [ ] **Step 2: Run notebook smoke tests and verify RED**

Run:

```bash
pytest -q tests/integration/test_learned_quality_ablation_notebook_smoke.py
```

Expected: failure because the notebook is A100-only and has no profile input.

- [ ] **Step 3: Update the generator and documentation**

Generate the form parameter, profile-specific hardware assertions, and strict run spec. Explain that audit mismatch now fails after selection restore and before large evidence restore, and recommend L4 for diagnostics.

- [ ] **Step 4: Regenerate the checked-in notebook using the implementation commit SHA**

Run the notebook generator after committing Tasks 1-3, then confirm the generated notebook equals the generator output.

### Task 4: Verify, commit, and publish

**Files:**
- Verify all files changed above.

- [ ] **Step 1: Run formatting and focused tests**

```bash
ruff check experiments/learned_quality/ablation_staging.py experiments/learned_quality/ablation_runner.py experiments/learned_quality/ablation_notebook.py scripts/learned_quality_ablation_run.py tests/experiments/learned_quality/test_ablation_staging.py tests/experiments/learned_quality/test_ablation_runner.py tests/experiments/learned_quality/test_ablation_cli.py tests/integration/test_learned_quality_ablation_notebook_smoke.py
pytest -q tests/experiments/learned_quality/test_ablation_staging.py tests/experiments/learned_quality/test_ablation_runner.py tests/experiments/learned_quality/test_ablation_cli.py tests/integration/test_learned_quality_ablation_notebook_smoke.py
```

- [ ] **Step 2: Run the learned-quality regression suite**

```bash
pytest -q tests/experiments/learned_quality tests/integration/test_learned_quality_ablation_notebook_smoke.py
```

- [ ] **Step 3: Review and commit implementation**

```bash
git status --short
git diff --check
git add <changed files>
git diff --staged
git commit -m "fix: validate ablation audit before restore"
```

- [ ] **Step 4: Regenerate, verify, and commit the immutable notebook pin**

```bash
python -m experiments.learned_quality.ablation_notebook --commit <implementation-sha> --output colab/learned_quality_training_ablation.ipynb
pytest -q tests/integration/test_learned_quality_ablation_notebook_smoke.py
git add colab/learned_quality_training_ablation.ipynb tests/integration/test_learned_quality_ablation_notebook_smoke.py
git commit -m "fix: pin L4 ablation notebook"
git push
```
