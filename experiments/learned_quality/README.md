# Learned-quality experiments

The production learned-quality experiment remains
`colab/learned_quality_a100_experiment.ipynb`. Its input, cache, result, and PLY
publication contracts are unchanged.

Use `colab/learned_quality_training_ablation.ipynb` when a completed learned run
trains successfully but renders as a white, exploded, or structurally corrupted
splat. The diagnostic notebook:

- requires a compatible passing CPU cache audit;
- restores a candidate selection first and rejects an audit mismatch before
  restoring large evidence artifacts;
- restores verified selection and pre-training evidence from Drive once;
- defaults to the recommended `l4_diagnostic` profile for an eligible L4, while
  retaining `a100_reference` for a 75+ GiB A100 reference run;
- runs every experiment from the same immutable session-local inputs;
- compares legacy control, dense seeds, masks, depth, adaptive density, and the
  complete learned combination at matched checkpoints;
- schedules pairwise experiments only when isolated variants do not explain a
  failing full combination; and
- publishes compact reports to `<input>_training_ablation` without publishing a
  production PLY or modifying `<input>_learned_test_result`.

The two runtime profiles change hardware admission only. They keep the same
experiment variants, iteration counts, reconstruction inputs, and quality
thresholds so diagnostic and reference results remain comparable.

The implementation is split into:

- `ablation.py`: matrix definitions, quality gates, and diagnosis classifier.
- `ablation_staging.py`: one-time verified local staging.
- `ablation_training.py`: independently controlled training features and
  deterministic checkpoint measurements.
- `ablation_runner.py`: process isolation, orchestration, reports, and atomic
  Drive publication.
- `ablation_notebook.py`: mechanically generated, immutable-SHA Colab notebook.

See [the operator guide](../../docs/TRAINING_ABLATION_NOTEBOOK.md) for the exact
Colab sequence and output semantics.

Use `colab/learned_quality_legacy_control_5k.ipynb` after the diagnostic matrix
when the goal is to inspect the proven legacy-control arm itself. It runs one
deterministic 5K training, preserves the raw trainer PLY byte-for-byte, invokes
the unchanged production PLY polisher, and publishes both the raw and polished
candidate with the complete polish report under
`<input>_legacy_control_5k_result`. A polish rejection does not discard the
candidate in this isolated comparison, and no full 120K training is started.
