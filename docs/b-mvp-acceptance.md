# B MVP Acceptance Results — 2026-05-19

Source spec: `docs/superpowers/specs/2026-05-19-single-image-fullscene-splat-design.md`
Implementation plan: `docs/superpowers/plans/2026-05-19-single-image-fullscene-splat.md`
Branch: `feat/b-single-image-fullscene-splat`

All three fixtures were run on the user's local hardware (RTX 3060 Ti 8 GB) using the `fast` profile (n_views=15, train_iterations=1500). Pipeline invoked via `scripts\run_with_msvc.bat scripts\run_b_fixture.py --scene <slug> --profile fast`.

## End-to-end pipeline outcomes

| Fixture | Source dims | Outcome | Gaussians (final) | Views added / rejected | Wall clock | PLY size |
|---|---|---|---|---|---|---|
| **fixture-a-render** (downscaled clean classroom) | 768 × 576 | ✅ Pipeline completes; all outputs on disk | 851,559 | 2 added / 12 rejected | **181 s (3 min)** | 57.9 MB |
| **fixture-b-empty-photo** (full-res classroom) | 4160 × 3120 | ✅ Pipeline completes; all outputs on disk | 4,356,003 | 1 added / 13 rejected | **1901 s (31.7 min)** | ~250 MB (est.) |
| **fixture-c-photo-with-objects** (classroom with desks) | 2390 × 1660 | ✅ Pipeline completes; all outputs on disk | 4,160,116 | 3 added / 11 rejected | **397 s (6.6 min)** | ~240 MB (est.) |

All runs are well **under the spec §2 budget of 90 min per fixture**.

For each successful run the following files exist under `worlds/<slug>/output/world/`:
- `0-world.ply` — the 3DGS scene
- `0-world-collider.json` — ground plane + bounding-box collider (§8 minimal collider)
- `0-world-trajectory.json` — 15 generated camera poses + spawn pose
- `.0-world-request.json` — sidecar with config, stats, wall-clock

## Acceptance checklist (manual visual playthrough — pending)

The viewer-side acceptance (spec §10 "Per-fixture acceptance checklist") requires loading each `.ply` into the D+E `Interactive` tab via `SplatBackground` and walking around. Frontend integration is implemented (Phase 11 — branch has SplatBackground, WorldCollider, WorldSelector wired) but **the manual playthrough has not been run** at the time of writing. User-driven; will be filled in after the user returns.

## Pipeline-level acceptance gates (spec §2 MVP completeness gate)

| Gate | Target | fixture-a | fixture-b | fixture-c |
|---|---|---|---|---|
| End-to-end success | Pipeline finishes, all outputs present | ✅ | ✅ | ✅ |
| Wall clock | ≤ 90 min on default profile | ✅ 3 min (fast) | ✅ 32 min (fast) | ✅ 6.6 min (fast) |
| Peak VRAM | ≤ 7.5 GB | unmeasured | unmeasured | unmeasured |

VRAM was not instrumented during these runs. The `peak_vram_gb_target` is in the config but no measurement code reads it. Future enhancement.

## Known quality concerns

1. **Very high outpaint-view rejection rate (12-13 of 14).** The depth-alignment step is rejecting almost every generated view as having too-high residual after the least-squares scale+shift fit. This means most views contribute zero new Gaussians, so the splat is dominated by:
   - the original seed pixels (the input image directly deprojected at MiDaS-inferred depth), plus
   - a small number of accepted view contributions

   The pipeline still produces a working splat, but the back wall / side walls (which are exactly what the outpaint loop was supposed to hallucinate) are sparse. Visual playthrough will likely show holes there. Tightening / loosening `align_residual_reject_threshold` (currently 0.25) is the lever to tune.

2. **No image-resolution cap on the seed step.** `cfg.image_max_dim = 512` is defined but not actually consumed; the runner deprojects every pixel of the input. fixture-b at 4160×3120 produced a 13M-point seed cloud, subsampled to 1.5M; fixture-a at 768×576 was 442k. This drives the wide variance in PLY size and training cost. A follow-up should downscale to `image_max_dim` before seeding.

3. **runwayml SD inpainting model is fragile.** The HF mirror only ships `.bin` files, which new diffusers versions block under the torch.load CVE. We're pinned to torch 2.6+ as the workaround. If runwayml is deleted from HF entirely, the pipeline will need to switch to a different inpainting model.

4. **gsplat MSVC compile dependency.** `gsplat 1.5.3` has no prebuilt wheel for Python 3.12 on Windows and requires JIT compilation via MSVC. The `_gsplat_msvc_shim` strips a GCC-only flag and the `run_with_msvc.bat` wrapper sources `vcvars64.bat`. First-run JIT takes ~2 min; subsequent runs reuse the cache.

## Environment captured for reproducibility

- Python: 3.12.8 via `py -3`
- torch: 2.5.1+cu124 → upgraded to 2.6.0+cu124 mid-session (diffusers CVE workaround)
- torchvision: 0.21.0+cu124
- gsplat: 1.5.3 (JIT-compiled via VS 2022 BuildTools MSVC 14.44.35207)
- diffusers: 0.38.0
- transformers / accelerate / safetensors: latest
- plyfile: 1.1.3
- pytorch-msssim: 1.0.0
- GPU: RTX 3060 Ti (8 GB)
- CUDA Toolkit: 12.6

## Next steps (when user returns)

1. Open `fixture-a-render` in the D+E viewer and do a manual playthrough; document holes and back-wall quality.
2. Same for fixture-b and fixture-c.
3. Decide whether to address the quality concerns (especially #1 — view rejection rate) before merging B → main, or to merge B-as-MVP and iterate on quality in a follow-up.
4. Consider the clean-plate sibling module (deferred per spec §2) to handle photos with foreground objects properly.
