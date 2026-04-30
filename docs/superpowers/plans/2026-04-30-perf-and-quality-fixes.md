# 4DGS Studio — Performance & Quality Fix Plan

**Goal:** Cut training wall-time from 10+ hours to ~3-4 hours on a 3060 Ti 8GB and improve 4D + 3D reconstruction quality, by addressing the root causes identified in the 2026-04-30 audit.

**Architecture:** Surgical fixes across `backend/model/` (trainer, gaussian_model, renderer, deformation) and a one-line patch to gsplat's CUDA build flags. No structural rewrites — every change preserves existing behavior except where the existing behavior is the root cause of slowness or quality loss.

**Tech Stack:** PyTorch 2.x, gsplat (CUDA rasterizer), Adam optimizer, HexPlane + MLP deformation field.

**Validation:** No test framework exists. Validation is: (a) `python scripts/smoke_phase_1_2.py` after each change to catch import/AST regressions; (b) careful semantic-equivalence review against the original code; (c) user-driven full training run after the batch.

**No git repo** — changes are saved in-place; user can `git init` afterwards if desired.

---

## Symptom → root-cause mapping

The user reports three symptoms; each maps to a known root cause:

| Symptom | Primary root cause | Plan task |
|---|---|---|
| Training takes 10+ hours | gsplat fused Adam build failed, `with_depth` does 2 rasterizations, host-device sync cascade, optimizer rebuild every density step | T1, T2, T3, T4 |
| Training "stucks at some point" | `torch.cuda.empty_cache()` + 4× param realloc inside density step → memory thrash on 8GB; per-iter NaN guards force GPU stalls; mid-run loss-skip path | T4, T5, T9 |
| 4D animation quality poor | Adam state wiped ~215× during densify → momentum never accumulates; deformation MLP runs duplicated per iter (wastes budget on noise) | T4, T6 |
| 3D environment quality "alright" | Same: lost Adam state on density steps + over-large deformation field for size of dataset | T4, T7 |

---

## File map

| File | Touched in |
|---|---|
| `E:\anaconda3\envs\gs4d\lib\site-packages\gsplat\cuda\csrc\Adam.cpp` | T1 (build flag) |
| `4dgs-studio/backend/model/renderer.py` | T2 |
| `4dgs-studio/backend/model/trainer.py` | T3, T4, T6, T8, T9 |
| `4dgs-studio/backend/model/gaussian_model.py` | T4, T7 |
| `4dgs-studio/backend/model/density_control.py` | T4 (read-only context) |
| `4dgs-studio/backend/model/deformation.py` | T7 |
| `4dgs-studio/backend/model/losses_perceptual.py` | T8 |

---

## Task ordering

Tasks are ordered so each builds on a working state from the prior. After every task, run the smoke test to confirm imports + AST still parse.

### T1 — Patch gsplat CUDA build flag (environmental fix)

**Problem:** `Adam.cpp` fails to compile on Windows MSVC because the build script passes `-Wno-attributes` (a GCC flag MSVC rejects with D8021). Fused Adam silently unavailable; we fall back to PyTorch Adam, which is multiple-x slower per Adam step.

**File:** `E:\anaconda3\envs\gs4d\lib\site-packages\gsplat\cuda\_backend.py` (or wherever `extra_cflags` lives — find via grep).

**Action:**
1. Grep gsplat's source for `-Wno-attributes` and replace with a Windows-safe equivalent (`/wd4838` or just remove the flag entirely on `os.name == 'nt'`).
2. Force a rebuild: delete `~/.cache/torch_extensions/.../gsplat_cuda/` and re-import gsplat once.
3. Confirm `from gsplat.optimizers import SelectiveAdam` works (or whatever fused-Adam class gsplat exposes).

**Validation:** Run smoke script. If gsplat refuses to load after the patch, revert and skip — fused Adam is a nice-to-have but not required for the other fixes to land. Document the revert in the task notes.

**Risk:** Medium — the patch is to a third-party install. If the build still fails, training continues with the slow Adam (no regression).

---

### T2 — Single rasterization for RGB+Depth

**Problem:** `render_view(with_depth=True)` does two full `rasterization()` calls. With default config `lambda_depth=0.1`, this fires every iter.

**File:** `4dgs-studio/backend/model/renderer.py`

**Action:** Replace the two-call path with a single `render_mode="RGB+ED"` call. gsplat returns a 4-channel tensor (RGB + expected depth) in one pass.

```python
def render_view(..., with_depth: bool = False):
    ...
    mode = "RGB+ED" if with_depth else "RGB"
    out, alpha, info = rasterization(
        means=means, quats=quats, scales=scales,
        opacities=opacities, colors=colors,
        viewmats=viewmat, Ks=Ks,
        width=width, height=height,
        sh_degree=sh_degree if colors.dim() == 3 else None,
        backgrounds=bg if not with_depth else torch.cat([bg, torch.zeros(1, 1, device=device)], dim=-1),
        render_mode=mode,
        packed=False,
    )
    return out[0], alpha[0], info
```

Caller in `trainer.py:1170-1184` already splits into `rgb` + `rendered_depth` based on shape — that logic stays. The 2-call branch in renderer is removed.

**Validation:** Confirm `out[..., 3]` matches the depth from the old 2-call path on a synthetic test. Run the renderer's `__main__` smoke (line 121-128) which tests both modes.

**Risk:** Low. gsplat has supported `RGB+ED` for ~2 years. If it's not in the installed version, the comment on renderer.py:5 implied a quirk — fall back to the 2-call path under a try/except.

---

### T3 — Defer per-iter `.item()` calls until after backward

**Problem:** Every iter performs ~5+ `.item()` / `bool()` on CUDA tensors before backward, forcing host-device sync and breaking GPU pipelining. Lines 1194, 1577, 1617, 1649-1651 in trainer.py are the worst offenders.

**File:** `4dgs-studio/backend/model/trainer.py`

**Action:**
1. **Stop populating `comp[...]` with `.item()` during loss accumulation.** Build `comp` with raw tensors (or a list of (name, tensor) pairs), and only call `.item()` once per `log_interval` when actually logging. Lines: 1194, 1211, 1242, 1301, 1358, 1379, 1429, 1444, 1451, 1463, 1474, 1488, 1507, 1535.
2. **Remove the per-iter pre-backward `not torch.isfinite(loss)` check** (line 1577). Replace with a post-backward check on a flag tensor that only triggers a sync once every ~100 iters or only when `loss.detach()` is finite-AND-numeric (use `loss.isfinite()` accumulated into a counter; sync the counter every 100 iters).
3. **Remove the per-iter grad-NaN scan** (lines 1617-1620). Move it inside the existing `if it % log_interval == 0` block, or tie it to the same 100-iter cadence.
4. **Move the post-step param NaN/Inf check** (lines 1649-1666) to the same low-cadence schedule. Three `.all()` syncs per iter on means/scales/quats is unjustifiable.

**Acceptance:** After the change, only one forced host-device sync per iter (the implicit one when `optimizer.step()` returns control — and even that overlaps under proper streaming).

**Validation:** Smoke script. Then run a quick benchmark: `python -c "from backend.model.trainer import Trainer4DGS; ..."` — measure iter/s on a tiny synthetic scene before/after. Expect ~30-50% speedup just from this change.

**Risk:** Medium. The NaN guards exist for real reasons (the in-code comments cite past divergence incidents). Keep the guards, just reduce their frequency. If divergence happens, the next checkpoint catches it.

---

### T4 — Stop wiping Adam state on density steps + fix `empty_cache` thrash

**Problem:** `_build_optimizer()` is called after every density step (trainer.py:1692), creating a fresh Adam. All `exp_avg`/`exp_avg_sq` buffers for every parameter are discarded. This happens ~215× in a 30k run during densify (500-22000, every 100). Combined with `torch.cuda.empty_cache()` at line 1709 and 4× full-tensor realloc inside `_apply_mask`/`append_gaussians`, this is the single biggest cause of (a) quality loss (Adam momentum never accumulates) and (b) "stuck" stalls (allocator thrash on 8GB).

**Files:**
- `4dgs-studio/backend/model/gaussian_model.py` — `_apply_mask`, `append_gaussians`
- `4dgs-studio/backend/model/trainer.py` — `_build_optimizer` and the `_density.step()` callsite

**Action:** Migrate optimizer state alongside parameter mutations. Standard INRIA-3DGS approach:

1. Add `_optimizer_aware_apply_mask(mask)` and `_optimizer_aware_append(...)` methods to `GaussianModel` that take the optimizer as an argument and update its `state[p]` dicts in lockstep with the parameter tensors. For each parameter `p`:
   - Pull `state = optimizer.state[p]` (dict with `exp_avg`, `exp_avg_sq`, `step`).
   - Apply the mask / concat to `exp_avg` and `exp_avg_sq` the same way the parameter is updated.
   - Re-key the state dict to the new `nn.Parameter` object: `optimizer.state[new_p] = state; del optimizer.state[old_p]`.
   - Update the param-group's `params` list to reference `new_p`.
2. Replace `self._build_optimizer()` at trainer.py:1692 with the in-place state-aware calls.
3. **Delete `torch.cuda.empty_cache()` at trainer.py:1709.** Once the realloc churn is gone, the cached allocator handles the residual fine. Keep it only if a leak is observed during smoke.

**Note on split-then-prune ordering:** density_control.py currently does `clone → split (which calls _apply_mask + append) → prune (which calls _apply_mask)`. Make sure all four touch points use the optimizer-aware path.

**Validation:**
- Smoke script (parse + import).
- Manual: instrument a single iter pre/post density step to confirm `optimizer.state` dict has correct keys and same total state size pre-density (param count) vs post-density (new param count).
- Quality: this is what the user sees as "alright but not great" — should improve.

**Risk:** High. This is the most invasive change. If state migration is buggy, training silently degrades. Add an assertion in `_optimizer_aware_*` that `optimizer.state[new_p]['exp_avg'].shape[0] == new_p.shape[0]`.

---

### T5 — Remove `torch.cuda.empty_cache()` from inner loop

Already covered as a sub-step of T4. Listed separately in case T4 turns out to be too risky to land in this batch — then this isolated change still helps stalls.

**File:** `4dgs-studio/backend/model/trainer.py:1709`

**Action:** Delete the line.

**Risk:** Very low.

---

### T6 — Reuse deformation MLP output (eliminate duplicate forward)

**Problem:** `_apply_deformation` (trainer.py:414) calls `self.deform(self.gs.means, t_norm, ...)` and returns `(d_means, d_quats, d_scales)`. The motion-regularizer block at line 1438 then calls `self.deform(self.gs.means, t_norm, ...)` AGAIN with the same inputs to get `(dpos, dquat, dscale)`. The first call's raw deltas (pre-clamp) aren't surfaced.

**File:** `4dgs-studio/backend/model/trainer.py`

**Action:** Refactor `_apply_deformation` to return both the post-clamp `(d_means, d_quats, d_scales)` AND the raw `(dpos, dquat, dscale)`. Update the regularizer block at 1431-1474 to consume the raw deltas instead of re-running the MLP.

```python
def _apply_deformation(self, t):
    if static_mode or deform is None:
        return (means, normalize(quats), get_scales), (None, None, None)
    dpos_mlp, dquat, dscale = self.deform(means, t, scene_extent)
    # ... existing clamps ...
    return (deformed_means, deformed_quats, deformed_scales), (dpos_total, dquat, dscale)
```

For the smoothness/accel regularizers, those legitimately need extra deform calls at `t±dt` — keep those. The fix is only for the duplicate at `t_norm`.

**Validation:** Smoke script + spot-check that `dpos_total.pow(2).mean()` numerically matches what the prior code computed.

**Risk:** Low — pure refactor. The semantic identity is straightforward.

---

### T7 — Cache `get_colors` + reduce default deformation field size

**Problem (a):** `gaussian_model.py:140-142` allocates `torch.cat([sh_dc, sh_rest], dim=1)` every render — ~5M floats per call.

**Problem (b):** `deformation.py:140-150` defaults are 4× the original (`mlp_width=512`, `mlp_depth=4`, `HexPlane resolution=96 feat=48`). The `__main__` block (line 227) shows the previous defaults were `(256, 2, 64, 32)`. With per-iter MLP forward over 100k+ Gaussians, the bigger net is a meaningful cost and (per the user's quality complaint) is more likely overfitting than learning useful motion.

**Files:**
- `4dgs-studio/backend/model/gaussian_model.py`
- `4dgs-studio/backend/model/deformation.py`

**Action:**
1. **`get_colors` caching:** add a private buffer `_colors_concat` that is rebuilt only when `_apply_mask` / `append_gaussians` mutates the count. The render path reads the buffer directly. (Caveat: `torch.cat` is the simplest correct version; if the cache adds bug surface, leave the cat alone and instead change `sh_dc` from `(N,1,3)` to a slice of a single `(N,K_total,3)` tensor.)
2. **Deformation field defaults:** Lower defaults to `mlp_width=384, mlp_depth=3, HexPlane resolution=64, feat_dim=32`. Mid-point between current bloat and previous lean. Keep all CLI/config knobs so the user can override.

**Validation:** Smoke script. Quality is the open question — needs a full training run by the user to confirm.

**Risk:** Medium for (b) — changing model capacity changes the optimum. The smaller field trains faster and may even *improve* quality on small datasets (less overfit). Document the change so the user can revert if quality regresses.

---

### T8 — Pin frame caches + LPIPS `.eval()` correctness

**Problem (a):** `frames_cached_mv[cam_id][idx]` is a CPU tensor; trainer does `.to(self.device)` synchronously every iter. Pinning + `non_blocking=True` lets the H2D copy overlap with compute.

**Problem (b):** `losses_perceptual.py:44` doesn't put the LPIPS model in `.eval()` mode despite the comment claiming so.

**Files:**
- `4dgs-studio/backend/model/trainer.py` — frame cache loaders (lines 829, 831, 1096-1099, 1119-1120, 1224-1227)
- `4dgs-studio/backend/model/losses_perceptual.py:44`

**Action:**
1. After `load_frame_tensor(...)`, call `.pin_memory()` on the resulting tensor before storing in the cache.
2. Change every `.to(self.device)` on cached tensors to `.to(self.device, non_blocking=True)`.
3. In `losses_perceptual.py:44`, append `.eval()`: `self._model = lpips.LPIPS(...).to(device).eval()`.

**Validation:** Smoke script. Speedup is small (single H2D copy of ~700KB at 360p) but free correctness wins.

**Risk:** Very low.

---

### T9 — Stop-gap: defaults for 8GB VRAM safety

**Problem:** On 8GB VRAM, density step OOM is the real "stuck" symptom. Without lowering peak memory, no software change makes 8GB enough at default 30k iters with all the Stage-2 losses on.

**Files:** `4dgs-studio/backend/config.py`

**Action:** Adjust defaults for the documented "RTX 3060 Ti 8GB" config to safer values:
- `max_gaussians`: change from `0` (unlimited) to `400_000` (hard cap — split is skipped beyond this; prune still runs).
- `prune_max_scale`: keep `0.02` (already conservative).
- Keep `image_resolution = (640, 360)`, `n_iters = 30_000`.
- Add a comment explaining the `max_gaussians` rationale.

**Note:** This is a config tweak, not a code change. The bigger memory wins come from T4 (no realloc spikes) and T2 (single rasterization buffer).

**Validation:** None required — config defaults only.

**Risk:** Low. User can override via `cloud_config()` for 24GB GPUs.

---

## Out of scope (deliberately deferred)

These were in the audit but are **not** in this plan:

- **Tier 3 preprocessing fixes** (depth/flow batching). User said training is the bottleneck. Preprocessing is cached after the first run anyway.
- **Mip-Splatting Python loop optimization** (audit #10) — only fires when `mip_scale_floor_frac > 0`, which is `0.0` in default config.
- **Track loss `cdist`** (audit #12) — could be replaced with a KNN index, but the size (256 × N) is bounded and only fires when track loss is on; not worth the complexity in this batch.
- **`int(torch.randint(...).item())`** (audit #21) — micro-optimization, ignored.
- **Dead `if False and ...` block** (audit #20) — leave alone; not a perf issue.

---

## Execution & validation flow

For each task:
1. Apply the change.
2. Run `python scripts/smoke_phase_1_2.py` from `4dgs-studio/`. Must print `PASS`.
3. Move to next task.

After all tasks:
1. Final smoke run.
2. User runs an actual training (5k iters) on a small scene to confirm: (a) no stalls, (b) iter/s noticeably higher, (c) no NaN aborts, (d) checkpoint saves.
3. If green, run full 30k.

## Self-review

- **Spec coverage:** All four user-reported symptoms map to at least one task (table at top).
- **Risk distribution:** T1 (gsplat patch) and T4 (state migration) carry most of the risk; T2/T3/T5/T6/T8/T9 are low-risk.
- **Reversibility:** Every change is a localized edit. T1 is reversible by reinstalling gsplat. T4 is reversible by reverting two methods + one trainer line. No data migrations, no schema changes.
- **Consistency:** Method names are consistent (`_optimizer_aware_apply_mask`, `_optimizer_aware_append`); referenced files exist; line numbers cross-checked against the audit.
