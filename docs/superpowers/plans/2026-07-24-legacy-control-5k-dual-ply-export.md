# Legacy-Control 5K Dual PLY Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one pinned A100 notebook that reruns only the passing
`legacy_control` 5K arm and atomically publishes both its untouched raw PLY and
its production-algorithm polish candidate.

**Architecture:** A focused exporter reuses the existing output-first staging,
historical legacy receipt validation, `primary_variants()` selection, and
`run_ablation_experiment()` training path. After training, an isolated helper
creates an in-memory acceptance proxy solely for `polish_static_ply`; the
original reconstruction is preserved and its failures are written into the
comparison report. A CLI and generated Colab notebook provide the same pinned,
flush-on-exit lifecycle as the existing diagnostic notebooks.

**Tech Stack:** Python 3.12, dataclasses, Pydantic contracts, PyTorch training,
existing static PLY polish/render evaluator, pytest, nbformat, Google Colab.

## Global Constraints

- Run exactly one `legacy_control` variant for exactly 5,000 iterations.
- Use seed `1701`, deterministic camera sampling, and a fixed 720-pixel long
  edge through the existing ablation implementation.
- Restore large Drive artifacts once into session-local storage.
- Preserve the trainer raw PLY byte-for-byte.
- Invoke the unchanged production polish algorithm through a diagnostic-only
  acceptance proxy; do not weaken the production reconstruction gate.
- Publish the valid polish candidate even when the polish regression gate
  rejects it and selects the raw PLY.
- Never start the full diagnostic matrix or a 120K run.
- Never overwrite production, learned-test, ablation, or floor-recovery
  results.

---

### Task 1: Dual-PLY exporter contract, polish adapter, and publisher

**Files:**
- Create: `experiments/learned_quality/legacy_control_export.py`
- Test: `tests/experiments/learned_quality/test_legacy_control_export.py`

**Interfaces:**
- Consumes: `StagedAblationInputs`, `AblationExperimentResult`,
  `primary_variants()`, `run_ablation_experiment()`,
  `polish_static_ply(...) -> PolishReport`.
- Produces:
  `LegacyControlExportRunSpec`,
  `LegacyControlExportResult`,
  `diagnostic_polish_bundle(reconstruction) -> ReconstructionBundle`,
  `run_legacy_control_export_from_staged(...) -> LegacyControlExportResult`,
  `run_legacy_control_export(...) -> LegacyControlExportResult`.

- [ ] **Step 1: Write failing tests for exact variant selection and the
  diagnostic polish boundary**

```python
def test_legacy_export_selects_only_exact_5k_control() -> None:
    variant = select_legacy_control_variant()
    assert variant.experiment_id == "legacy_control"
    assert variant.features == frozenset()
    assert variant.n_iterations == 5_000
    assert variant.density_events is True


def test_diagnostic_polish_proxy_does_not_mutate_original(
    learned_reconstruction,
) -> None:
    original = learned_reconstruction.bundle
    proxy = diagnostic_polish_bundle(learned_reconstruction)
    assert proxy is not original
    assert proxy.decision.passed is True
    assert proxy.decision.failures == ()
    assert original.decision.passed is False
    assert original.decision.failures
    assert proxy.decision.dominant == original.decision.dominant
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```powershell
python -m pytest tests/experiments/learned_quality/test_legacy_control_export.py -q
```

Expected: collection fails because
`experiments.learned_quality.legacy_control_export` does not exist.

- [ ] **Step 3: Implement the minimal contracts and isolated proxy**

```python
RESULT_SUFFIX = "_legacy_control_5k_result"
GENERATOR_ID = "4dgs-studio.legacy-control-5k-dual-ply-v1"


class LegacyControlExportPublishSpec(StrictModel):
    replace_owned_result: bool = True


class LegacyControlExportRunSpec(StrictModel):
    input_folder: str
    runtime_profile: Literal["a100_legacy_control_5k"] = (
        "a100_legacy_control_5k"
    )
    publish: LegacyControlExportPublishSpec = Field(
        default_factory=LegacyControlExportPublishSpec
    )


def select_legacy_control_variant() -> AblationVariant:
    matches = tuple(
        variant
        for variant in primary_variants()
        if variant.experiment_id == "legacy_control"
    )
    if len(matches) != 1:
        raise RuntimeError("exactly one legacy_control variant is required")
    variant = matches[0]
    if (
        variant.features
        or variant.n_iterations != 5_000
        or not variant.density_events
    ):
        raise RuntimeError("legacy_control contract changed")
    return variant


def diagnostic_polish_bundle(
    reconstruction: LearnedReconstructionOutput,
) -> ReconstructionBundle:
    original = reconstruction.bundle
    if original.decision.passed or not original.decision.failures:
        raise ValueError("diagnostic proxy requires rejected best-effort geometry")
    decision = replace(
        original.decision,
        passed=True,
        failures=(),
        retry_recommended=False,
    )
    return replace(original, decision=decision)
```

- [ ] **Step 4: Add failing tests for raw preservation, rejected-candidate
  preservation, report fields, and atomic ownership**

The test creates distinct raw/candidate byte payloads, injects a
`PolishReport(accepted=False, candidate_path=candidate, selected_path=raw, ...)`,
and asserts:

```python
assert (published / "raw_legacy_control_5k.ply").read_bytes() == raw_bytes
assert (published / "polished_legacy_control_5k.ply").read_bytes() == candidate_bytes
payload = json.loads((published / "polish_report.json").read_text())
assert payload["accepted"] is False
assert payload["original_geometry_failures"] == ["registered_ratio"]
assert payload["original_count"] == 100
assert payload["kept_count"] == 90
assert payload["reasons"] == ["mean_psnr_drop"]
assert json.loads((published / "_OWNERSHIP.json").read_text())[
    "generator_id"
] == GENERATOR_ID
```

Also assert an existing folder without this exact ownership marker is never
replaced and that a missing/empty polish candidate prevents publication.

- [ ] **Step 5: Run the focused tests and confirm RED**

Run:

```powershell
python -m pytest tests/experiments/learned_quality/test_legacy_control_export.py -q
```

Expected: proxy tests pass; publication and execution tests fail because the
functions are not implemented.

- [ ] **Step 6: Implement training, polish, evidence preparation, and atomic
  publication**

The staged runner must:

```python
variant = select_legacy_control_variant()
workspace = materialize_experiment_workspace(
    staged,
    experiments_root=local_root / "experiments",
    experiment_id=variant.experiment_id,
)
prepared = PreparedTrainingInput(
    run_id=variant.experiment_id,
    data_root=workspace.scene_root,
    scene_name="scene",
    frames_dir=staged.reconstruction.frames_dir,
    reconstruction=staged.reconstruction.bundle,
    source_digest=staged.source_digest,
    selection_digest=staged.reconstruction.selected_manifest.image_set_digest,
)
training = run_ablation_experiment(
    prepared,
    base_spec,
    staged.reconstruction,
    variant,
    diagnostic_root=report_root / "diagnostics",
    seed=1701,
    control_checkpoints={row.iteration: row for row in historical_checkpoints},
)
```

It then parses cameras, exact-joins registered frames, constructs the proxy,
and calls:

```python
polish = polish_static_ply(
    training.raw_ply_path,
    local_root / "polish" / "candidate.ply",
    diagnostic_polish_bundle(staged.reconstruction),
    registered_frames,
)
```

Success requires `training.passed`, eight exact checkpoints, a valid raw PLY,
and a non-empty valid `polish.candidate_path`. The report records the untouched
original geometry decision and acceptance contract. Publication copies both
PLYs, metrics, manifest, checkpoint contact sheets, receipt, provenance, and
polish report into a sibling staging directory, writes `_SUCCESS.json` last,
then swaps only an owned destination.

- [ ] **Step 7: Run focused tests and confirm GREEN**

Run:

```powershell
python -m pytest tests/experiments/learned_quality/test_legacy_control_export.py -q
```

Expected: all exporter tests pass.

- [ ] **Step 8: Commit the exporter**

```powershell
git add experiments/learned_quality/legacy_control_export.py `
  tests/experiments/learned_quality/test_legacy_control_export.py
git commit -m "feat: export raw and polished legacy 5k PLYs"
```

### Task 2: Fail-closed CLI and pinned A100 notebook

**Files:**
- Create: `scripts/learned_quality_legacy_control_export_run.py`
- Create: `experiments/learned_quality/legacy_control_export_notebook.py`
- Create: `tests/experiments/learned_quality/test_legacy_control_export_cli.py`
- Create: `tests/integration/test_legacy_control_export_notebook_smoke.py`
- Generate: `colab/learned_quality_legacy_control_5k.ipynb`

**Interfaces:**
- Consumes:
  `run_legacy_control_export(spec, model_manifest_path, expected_source_revision)`.
- Produces:
  `/content/legacy_control_export_result.json` and a checked-in pinned notebook.

- [ ] **Step 1: Write failing CLI tests**

Tests assert the CLI:

```python
assert payload["status"] == "success"
assert payload["raw_ply_path"].endswith("raw_legacy_control_5k.ply")
assert payload["polished_ply_path"].endswith("polished_legacy_control_5k.ply")
```

On failure it must write:

```python
{
    "status": "failed",
    "error_type": type(error).__name__,
    "error_message": str(error),
    "durable_failure_path": str(path),
    "full_120k_training_started": False,
}
```

- [ ] **Step 2: Run CLI tests and confirm RED**

Run:

```powershell
python -m pytest tests/experiments/learned_quality/test_legacy_control_export_cli.py -q
```

Expected: import fails because the CLI module does not exist.

- [ ] **Step 3: Implement CLI and durable failure receipt**

The CLI accepts `--spec`, `--source-revision`, and `--model-manifest`, invokes
the runner once, writes `/content/legacy_control_export_result.json`
atomically, and never reports success unless both published PLY paths exist.

- [ ] **Step 4: Write failing notebook smoke tests**

Tests require one parameter cell, A100/75-GiB VRAM and 80-GiB disk preflight,
source pin verification, isolated environment setup, unbuffered CLI execution,
both exact output filenames, Drive flush, and `runtime.unassign()` in `finally`.
They also require the checked-in notebook to equal generator output for its
pinned commit.

- [ ] **Step 5: Run notebook smoke test and confirm RED**

Run:

```powershell
python -m pytest tests/integration/test_legacy_control_export_notebook_smoke.py -q
```

Expected: import or generated-notebook assertion fails because the generator
and notebook do not exist.

- [ ] **Step 6: Implement and generate the notebook**

Generate with:

```powershell
python -m experiments.learned_quality.legacy_control_export_notebook `
  --commit-sha HEAD `
  --output colab/learned_quality_legacy_control_5k.ipynb
```

The final cell prints both Drive paths and the polish acceptance/rejection
summary before flushing and releasing the runtime.

- [ ] **Step 7: Run CLI and notebook tests and confirm GREEN**

Run:

```powershell
python -m pytest `
  tests/experiments/learned_quality/test_legacy_control_export_cli.py `
  tests/integration/test_legacy_control_export_notebook_smoke.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit CLI and notebook**

```powershell
git add scripts/learned_quality_legacy_control_export_run.py `
  experiments/learned_quality/legacy_control_export_notebook.py `
  tests/experiments/learned_quality/test_legacy_control_export_cli.py `
  tests/integration/test_legacy_control_export_notebook_smoke.py `
  colab/learned_quality_legacy_control_5k.ipynb
git commit -m "feat: add legacy 5k dual PLY notebook"
```

### Task 3: Documentation, regression suite, and source repin

**Files:**
- Modify: `colab/README.md`
- Modify: `experiments/learned_quality/README.md`
- Modify: `colab/learned_quality_legacy_control_5k.ipynb`

**Interfaces:**
- Consumes the completed exporter, CLI, and notebook generator.
- Produces user-facing run instructions and a notebook pinned to the final
  reviewed implementation commit.

- [ ] **Step 1: Document the exact use and outputs**

Document that the notebook runs only the legacy-control 5K arm, publishes both
PLYs, labels polish as diagnostic when source geometry is best-effort, and
never starts 120K.

- [ ] **Step 2: Run focused and neighboring regression tests**

Run:

```powershell
python -m pytest `
  tests/experiments/learned_quality/test_legacy_control_export.py `
  tests/experiments/learned_quality/test_legacy_control_export_cli.py `
  tests/integration/test_legacy_control_export_notebook_smoke.py `
  tests/experiments/learned_quality/test_ablation_training.py `
  tests/static_pipeline/test_polish.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Run formatting and compile checks**

Run:

```powershell
python -m ruff check `
  experiments/learned_quality/legacy_control_export.py `
  experiments/learned_quality/legacy_control_export_notebook.py `
  scripts/learned_quality_legacy_control_export_run.py `
  tests/experiments/learned_quality/test_legacy_control_export.py `
  tests/experiments/learned_quality/test_legacy_control_export_cli.py `
  tests/integration/test_legacy_control_export_notebook_smoke.py
python -m compileall -q experiments/learned_quality scripts
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 4: Commit implementation and documentation**

```powershell
git add colab/README.md experiments/learned_quality/README.md
git commit -m "docs: explain legacy 5k PLY comparison"
```

- [ ] **Step 5: Repin generated notebook to the final reviewed commit**

Generate the notebook with the current commit, run the smoke test again, then:

```powershell
git add colab/learned_quality_legacy_control_5k.ipynb
git commit -m "chore: repin legacy 5k export notebook"
```

- [ ] **Step 6: Verify the final clean branch and push**

Run:

```powershell
git status --short --branch
git log -5 --oneline
git push
```

Expected: the feature branch is clean and synchronized with its remote.
