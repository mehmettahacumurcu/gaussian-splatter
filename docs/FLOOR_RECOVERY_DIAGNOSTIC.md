# Floor-recovery diagnostic

This notebook is the final bounded test for the missing-room-floor failure. It does not
change the accepted legacy pipeline and it never launches a full 120k training run.
Instead, it compares one automatic floor-hole candidate against the preserved 5k A100
legacy control.

## Prerequisites

Use the exact same input folder for every prerequisite:

1. The CPU cache-audit notebook must have printed `TRACK AUDIT PASSED`.
2. The A100 structural diagnostic matrix must have completed and published
   `<input>_training_ablation/_SUCCESS.json` with its legacy control. The runner pins
   and verifies the exact accepted A100 matrix producer separately from its newer
   floor-recovery source revision.
3. Start a fresh **A100 80 GB High-RAM** Colab runtime. L4, T4, and standard-RAM
   runtimes are deliberately rejected.
4. Keep at least 80 GiB of free local Colab disk.

## Run it

1. Open `colab/learned_quality_floor_recovery.ipynb` from the
   `feature/learned-quality-a100` branch.
2. Choose **Runtime -> Change runtime type -> A100 GPU**, then select High-RAM.
3. Set `INPUT_FOLDER` to the MyDrive-relative capture folder, for example
   `myroom_test`. Do not select a cache, result, diagnostic, or ablation folder.
4. Choose **Runtime -> Run all** and leave the final cell running.

The notebook restores the already-verified input lineage once into local Colab storage.
For the current room this is roughly 19 GiB and many small files, so Google Drive FUSE
can take 30-90 minutes or longer even when the byte-rate display looks low. The restored
data stays local for the plane fit, hole detection, candidate construction, and the one
5k training arm; it is not downloaded again between those steps.

The bounded arm uses legacy RGB training and the proven legacy density schedule, plus
safe depth and at most 150,000 low-opacity seeds only in empty cells of an automatically
verified floor plane. Confirmed motion and sky are excluded. No semantic mask, global
dense-seed, adaptive-density, or full learned arm is enabled.

## Monitor it

Run this in the Colab terminal:

```bash
RUN=$(find /content/4dgs-floor-recovery -mindepth 1 -maxdepth 1 -type d \
  -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
PID=$(pgrep -f '[s]cripts.learned_quality_floor_recovery_run' | head -1)

date
echo "Run: ${RUN:-not created}"
ps -p "$PID" -o pid,etime,%cpu,%mem,rss,stat,cmd 2>/dev/null || \
  echo "No active floor diagnostic process."

METRICS=$(find "$RUN" -type f -name metrics.jsonl \
  -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
if [ -n "$METRICS" ]; then tail -n 3 "$METRICS"; else echo "5K training has not started."; fi

nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader
cat /content/learned_floor_recovery_result.json 2>/dev/null || \
  echo "No final receipt yet."
```

## Result and decision

The notebook publishes only:

```text
MyDrive/<input>_floor_recovery_diagnostic/
```

It never overwrites `<input>_result`, `<input>_learned_test_result`, or the ablation
report. `_SUCCESS.json` means the bounded diagnostic completed, not that the candidate
passed. Read `decision.json` for the verdict:

- `passed`: every floor-recovery, image-quality, structural, and perturbed-view gate
  passed. `candidate.ply` is included for visual inspection.
- `quality_gates_failed`: the candidate was safely rejected. The report still includes
  its measurements and failure reasons; no full training is started.
- an infrastructure failure produces no successful diagnostic marker and the CLI
  receipt has `status: failed`.

Passing requires at least 25% hole reduction, at least 10 percentage points of floor
alpha gain, no more than 1 dB PSNR loss against legacy, visible-white fraction at most
1.5%, oversized fraction at most 1%, out-of-bounds fraction at most 0.1%, no non-finite
values, and perturbed-view depth disagreement at most 1.10.
