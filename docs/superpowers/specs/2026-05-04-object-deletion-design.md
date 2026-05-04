# Object Deletion (Static 3DGS) — Design Spec

**Status:** Draft, ready for plan
**Date:** 2026-05-04
**Owner:** TAHA
**Phase:** Phase 2 feature on top of static 3DGS pipeline

## Summary

Add a single-shot object-deletion edit operation to the static 3DGS pipeline. User clicks an object in any training frame; backend uses SAM 2 to mask it, propagates the mask to all training views, identifies and removes the corresponding Gaussians, AI-inpaints the void across views, and runs a brief local refit to integrate the inpainted regions. Result is a new versioned checkpoint + PLY; the source is preserved. The "killer combo" path from `docs/FUTURE_DIRECTIONS.md`.

## Goals

- One-click selection from any training frame produces a clean delete on the entire object across all views.
- Result looks like the object was never there for the user's typical capture style (small-to-medium clutter in indoor scenes), not just a void.
- Two compute modes: a fast LaMa preview (~5 min total) and a quality SD inpaint pass (~15 min total).
- Original training output untouched; each edit produces a new `edits/edit_<n>/` artifact.
- Cancellable mid-job from the frontend.
- Frontend integrates with the existing JobsList / job-progress pattern; no new orchestration primitives.
- 8 GB VRAM budget respected via sequential model loading (SAM 2 → inpainter → main pipeline).

## Non-goals (v1)

- Multi-object delete in a single job (user submits two jobs)
- In-app undo (user loads `parent_ckpt` manually)
- Soft delete / opacity-zero (deleted Gaussians are removed)
- Inpaint quality knobs exposed to the UI (sane defaults baked in)
- Multi-click positive/negative SAM refinement (single positive click only — v1.5 hook)
- Side-by-side compare viewer (v1.5)
- Recolor, add, LLM editing, branching edit chains (separate specs)

## Decisions made during brainstorming

| Decision | Choice | Why |
|---|---|---|
| Tier 1 (delete only, void) vs Tier 2 (delete + inpaint + refit) | **Tier 2** | User explicitly requires the void to be filled; matches "killer combo" framing in `FUTURE_DIRECTIONS.md`. |
| Selection method | **SAM 2 click-on-frame, video-propagate, classify Gaussians** | Mature model, no training-pipeline changes, video propagation is a perfect fit for multi-view-of-static-scene captures. Click-in-3D-viewer (option A) deferred — would require retraining with per-Gaussian instance labels. |
| Inpaint backend | **SD inpainting (mode B, quality default) + LaMa (mode A, fast preview)** | SD for credible quality bar (matches GScream/SPin-NeRF baselines); LaMa as preview lets the user trial the selection before committing to a 15-min run. |
| Inpaint coverage | **Sparse-view (~95 anchors of ~750 frames) + depth-warp propagation** | Cross-view consistency is the critical quality factor; warping anchor inpaints into other frames produces consistency that per-frame inpaint cannot. |
| Refit scope | **Local refit (Gaussians within 1.5× max-deleted-Gaussian-extent), freeze rest** | Avoids perturbing distant good regions. Matches established Gaussian-editing literature. |
| Edit interaction model | **(A) Single-shot per session in v1, (B) chained edits in v2** | Reuses existing job submission flow; v2's chain is just UI on top of `parent_ckpt` metadata that v1 already writes. |
| Cancel UX | **In v1 scope** | Mode B's 15-min runs make cancel a real UX need; once shipped, benefits training jobs too. |

## Architecture overview

A new backend job type (`object_delete`), parallel to the existing `static_3d` training job. It consumes a source checkpoint and produces a new versioned checkpoint + PLY. Frontend submits exactly like a training job; the existing `JobsList` and progress widgets work unchanged.

### Components (all new)

- **`backend/edit/sam_service.py`** — Loads SAM 2 ViT-B (~600 MB) lazily, kept resident across edits. Provides `predict_from_click(frame, x, y) -> mask` and `propagate_video(frames, seed_mask) -> mask_stack`.
- **`backend/edit/inpainter.py`** — Two backends behind one interface: `LaMaInpainter` (~200 MB) and `SDInpainter` (~4 GB peak). Method: `inpaint(rgb, mask) -> rgb_filled`. Owners are responsible for load/unload of their model.
- **`backend/edit/warp.py`** — Depth-warp anchor inpaints into target frames using existing depth maps + COLMAP poses. Method: `warp_inpaint(target_frame_idx, anchor_inpaints, anchor_indices, depths, w2cs, K) -> warped_rgb`.
- **`backend/edit/gauss_classifier.py`** — Projects each Gaussian center into every view, accumulates "fraction of views where projection lands inside mask," thresholds. Method: `classify(gs, mask_stack, K, w2cs, threshold=0.6) -> bool[N]`.
- **`backend/edit/refit.py`** — Loads source ckpt, removes flagged Gaussians, freezes survivor gradients, runs local-region densify+refit for 3k (mode A) or 5k (mode B) iters. Reuses existing `Trainer4DGS` infrastructure.
- **`backend/edit/runner.py`** — `EditJobRunner` class wired into `JobManager`. Orchestrates the 5 phases, emits progress callbacks, handles cancel checks at phase boundaries and inside Phase 4's per-anchor loop.
- **API extensions to `backend/api.py`**: `POST /jobs/edit`, `POST /jobs/:id/cancel`.
- **Frontend `frontend/src/components/EditPanel.tsx`** — New tab/mode in viewer area. Frame picker → mask preview → mode toggle → submit → progress → "switch to result". Reuses Spark viewer for the resulting PLY.
- **Frontend `JobsList.tsx`** — Add "İptal" button on rows where `status === "running"`.

## Edit job pipeline (5 phases)

| Phase | What | Time mode A / B | Cancel checkpoint |
|---|---|---|---|
| 1 — Mask in clicked frame | SAM 2 single-positive-click → mask `[H, W]` | ~3 s | Before phase |
| 2 — Video propagation | SAM 2 video propagation across all training frames → `[T, H, W]` | ~30 s for 750 frames | Before phase |
| 3 — Gaussian classification | Project, vote, threshold (≥60% mask coverage = delete) | ~10 s | Before phase |
| 4 — Inpaint | Sparse-view (~95 anchors) inpaint + depth-warp to all views | A: ~1 min · B: ~5–10 min | Per anchor |
| 5 — Local refit | Freeze survivors, densify in affected region, train with inpainted GT | A: ~3 min · B: ~10 min | Every iter (existing trainer pattern) |

### Loss caveats during refit

- RGB + SSIM: full strength
- LPIPS-512 (existing): full strength
- Depth supervision: **masked off** inside the SAM mask region (no GT depth for the inpaint)
- Aniso: unchanged

### Memory choreography (8 GB)

- Phase 1–3: SAM 2 (600 MB) + source ckpt loaded → comfortable.
- Phase 4 mode A: SAM 2 unloaded → LaMa loaded (200 MB) → run → unload.
- Phase 4 mode B: SAM 2 unloaded → SD inpainter loaded (4 GB peak) → run → unload.
- Phase 5: full main pipeline state (~6 GB peak with LPIPS-512). SAM and inpainter MUST be unloaded by this point.

## App integration

### Backend

- `POST /jobs/edit` payload: `{ scene, source_ckpt, frame_idx, click_xy: [u, v] (normalized to [0,1] frame coords), quality_mode: "A"|"B" }`. Response: `{ job_id, status, status_url }`.
- `POST /jobs/:id/cancel` returns 200 if accepted, 404 if not found, 409 if already terminal.
- New `EditJobRunner` subclasses or composes with `TrainingJobRunner`. Phase callbacks emit standard progress events.
- Cancel mechanism: `JobManager.cancel(job_id)` sets a flag. Runners check between phases and at known long-loop boundaries (per-anchor in Phase 4, per-iter in Phase 5).

### Disk artifact layout

```
data/<scene>/output/
├── ckpt/                         # original training output, untouched
├── ply/
├── eval/
└── edits/                        # NEW
    ├── edit_001/
    │   ├── ckpt.pt               # contains parent_ckpt metadata
    │   ├── splat.ply             # exported PLY for viewer
    │   ├── mask_preview.png      # debugging / "what got selected"
    │   ├── inpaint_anchors_tmp/  # in-progress, atomic-renamed when complete
    │   ├── inpaint_anchors/      # 95 inpainted anchor RGBs (cached for re-refit)
    │   └── meta.json             # {parent_ckpt, edit_op, click_xy, frame_idx, mode, n_deleted, timestamps}
    ├── edit_002/
    └── ...
```

`edit_<n>` is sequence-numbered globally per scene (next number = `max(existing edit_NNN dirs) + 1`). `meta.parent_ckpt` enables v2's chain UI without schema changes. Atomic-rename pattern on `inpaint_anchors_tmp/` → `inpaint_anchors/` ensures cancel mid-Phase-4 leaves no corrupt cache.

### Frontend `EditPanel` state machine

1. **Idle** — Edit Mode button visible if a completed training job exists for current scene.
2. **Selecting** — Spark viewer replaced by a frame picker strip (8–12 thumbnails at trajectory-uniform intervals, computed on demand from `data/<scene>/frames/`). User selects a frame, frame fills viewer at full size.
3. **Click target** — User clicks the object on the displayed frame. Backend Phase 1 runs.
4. **Mask preview** — Returned mask overlays on frame in red+50% alpha. User confirms (proceed) or re-clicks (re-prompt SAM, replaces mask).
5. **Quality choice** — Dialog: "Quick preview (~5 min total, LaMa)" vs "Full quality (~15–20 min total, SD)". Times include phases 1–5, not just inpaint. Defaults to LaMa on first edit, remembers user's last choice within the session.
6. **Running** — Progress bar with phase indicator, "İptal" button. Reuses existing job-progress component; new behavior is the cancel.
7. **Done** — Notification offers "Switch to result" (replace Spark viewer's source). v1.5: "Compare side-by-side" loads both source and edit in two viewports.

Frame thumbnails computed on demand, not preprocessed, on local SSD.

## Edge cases & failure modes

| Case | Detection | v1 mitigation |
|---|---|---|
| SAM mask leaks to neighbors (clicks chair, grabs floor strip) | User sees mask preview that's clearly wrong | User re-clicks. v1.5 adds positive/negative refinement clicks on same code path. |
| Object behind has no captured surface — inpainter must hallucinate from scratch | Phase 5 refit produces affected-region LPIPS materially worse than scene-mean LPIPS | Surface as a metric in the result panel; user decides whether to keep. No auto-revert. |
| Selection lands on object visible in <5 frames | Phase 3 outputs `n_deleted < 50` | Backend returns early with structured "selection too sparse — try a more central object" status. |
| Cancel mid-Phase-4 | User clicks İptal while sparse-view inpaint is running | Per-anchor inpaints write to `inpaint_anchors_tmp/`. Atomic rename to `inpaint_anchors/` only at end of Phase 4. Cancel deletes tmp dir, no corruption. |
| Refit damages quality outside affected region | Regression test catches gradient leak | Freeze-survivors logic is unit-tested with synthetic fixture. |
| First-run model download (~600 MB SAM 2, ~4 GB SD) | Fresh install | Frontend shows "Downloading model (one-time): X / Y GB" before Phase 1. User confirms before metered download starts. |

## Testing strategy

- **Unit (synthetic)** — Mask projection given known intrinsics + 3D points → expected mask. Gaussian classifier given hand-crafted projections + masks → expected delete labels. Run in CI.
- **Integration fixture** — `tests/fixtures/cube_scene/` with 10 frames + ~5k Gaussians + a known target cube. Run full edit, assert: (a) cube Gaussians deleted, (b) PSNR on non-affected views unchanged within 0.5 dB of source, (c) `meta.json` has correct `parent_ckpt`. Run in CI.
- **Quality regression (manual)** — On `myroom_v2` post-training, delete a known small object, eyeball orbit mp4. Smoke test before each release.
- **Cancel test** — Submit edit, cancel at each phase boundary, assert no partial artifacts written, `nvidia-smi` returns to baseline within 30 s.

## v2 hooks (no rework required)

- **Chained edits** — `meta.parent_ckpt` is set in v1; v2 adds chain-tree UI in JobsList. No schema change.
- **Recolor** — Phases 1–3 unchanged; replaces inpaint+refit with SH coefficient operation. New file `backend/edit/recolor.py`. `EditPanel` becomes mode-aware: "Delete" vs "Recolor" radio.
- **Multi-click selection** — Phase 1 API already accepts arbitrary `(x, y, label)` lists; v1 sends a single positive point. v1.5 frontend lets user accumulate clicks with shift-click.
- **Side-by-side compare** — `EditPanel` already references `parent_ckpt` from meta; v1.5 adds a second viewport.
- **AI insertion (Tier 3)** — Different operation; needs text-to-3DGS or asset library. Selection placement infrastructure carries over but the operation is genuinely new. Separate spec when scheduled.

## Newly shared infrastructure

The cancel mechanism (`POST /jobs/:id/cancel`, `JobManager.cancel(job_id)`, JobsList "İptal" button) is implemented as a general primitive on `JobManager`, not edit-specific. Once shipped with v1 edit, it's available for training jobs without further work. Cancel checkpoints in `TrainingJobRunner` (between iterations) are added as part of this scope.

The Gaussian classifier (`backend/edit/gauss_classifier.py`) is reused as-is by v2 recolor.

## Open questions / risks

1. **SAM 2 video propagation quality on COLMAP-name-sorted "video"** — for an orbital capture, sorted-by-name is roughly trajectory-temporal but not strictly continuous (there are jumps when the camera transitions between loops). SAM 2's video propagation expects temporal continuity. Risk: propagated masks have artifacts at loop boundaries. Mitigation: if quality is poor in testing, sort frames by camera pose distance instead of by name. Test with `myroom_v2` where the user explicitly walked multiple loops.
2. **Depth-warp quality with MiDaS depth** — MiDaS produces relative depth. The pipeline aligns it to COLMAP scale, but warp accuracy depends on alignment quality. Risk: warped inpaints have ghosting at depth discontinuities. Mitigation: in failure regions, fall back to per-frame inpaint; the test fixture should include a depth-discontinuity case.
3. **8 GB peak with SD inpaint + Phase 5** — assuming sequential loading. Need to verify that PyTorch fully releases SD's VRAM between Phase 4 and Phase 5 on Windows (where `expandable_segments` is a no-op). Add an explicit `torch.cuda.empty_cache()` and a sanity-check assertion in `EditJobRunner` between phases.
4. **First-run download UX over metered connection** — listed in failure modes but worth highlighting: 4 GB SD download could surprise users. Mitigation already specified (confirm dialog before download).

## Out of scope (deliberately)

- Multi-object delete in one job
- Undo (load `parent_ckpt` manually for now)
- Soft delete (opacity → 0)
- Inpaint quality knobs exposed (steps, guidance scale)
- Inpaint model selection beyond LaMa/SD (no Flux, no other backends)
- Branching edit history
- Cloud offload of inpaint (do everything locally on 8 GB)
- Object replacement (would need text-to-3DGS — Tier 3)
