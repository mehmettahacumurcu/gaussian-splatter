# Splat morph v1

The Scene Editor previews a deterministic, diffusion-inspired A → particle cloud → B transition entirely in the browser. This is a procedural effect, not a diffusion model. Default settings are full dissolve, full target blend, a six-second duration and seed 42.

## Try it

1. From the repository root, start the existing backend with `scripts\start-backend.bat`. Alternatively, in the prepared `gs4d` environment, run `uvicorn backend.api:app --host 127.0.0.1 --port 8000` as described in the README.
2. In a second terminal, run `cd frontend`, `npm install` if needed, then `npm run dev`. Open **http://localhost:1420**. For the existing desktop launcher, double-click root `run.bat`: it starts the backend and `npm run tauri dev` in two windows.
3. Open **Sahne Editörü**. Open an existing scene with **Aç**, or upload/select a base `.ply` under **Yeni sahne** and click **Oluştur**. Under **Obje ekle**, upload/select another `.ply` and click **Sahneye ekle**. Start with about 200,000 splats per asset.
4. Position, scale, crop and colour the objects as desired. Select **A** on the source object's row and **B** on the target's row. These pair buttons leave the normal inspector selection intact. Keep both objects visible for a complete transition.
5. In the right-hand **Morph** panel, click **Morph hazırla**, wait for the particle count/preparation time, then **Oynat**. Orbit the camera normally; object gizmos are suspended during preview. **Duraklat** pauses, **Başa dön** resets, and the timeline scrubs deterministically.
6. **Önizlemeyi kapat** restores the normal scene. **A ↔ B** swaps the pair; prepare again. Changing seed or an endpoint's geometry/appearance prepares a new pairing. Leaving the Scene Editor pauses playback; deleting an endpoint or closing the scene ends the preview.

## Controls

| Control | Meaning |
| --- | --- |
| Duration / Süre | 0.5–120 seconds of active playback; large frame stalls are capped at 0.1 seconds so returning from a suspended tab does not skip the effect. |
| Dissolve | 0–1; controls bounded cloud displacement and shrinkage toward small isotropic particles. At 1 the full cloud appears at the midpoint; at 0 it is a direct shape interpolation. |
| Target blend | 0–1; maximum progress toward B. Effective progress is `timeline * targetBlend`: 0 holds A, 0.5 ends in the cloud, and 1 reaches exact B. |
| Seed | Unsigned 32-bit integer; changes stable particle trajectories. Scrubbing, pausing and replaying never generate fresh randomness. |

## Implementation and Spark API

- Confirmed against the installed **Spark 0.1.10** types and embedded sources/examples. Extraction uses `PackedSplats.getSplat`, applies local crop before the composer’s uniform scale/rotation/translation, composes splat quaternions and applies the existing DC colour matrix. Extraction yields every 4,096 splats; sorting/matching runs in a module worker with transferred buffers.
- Per-axis 2nd/98th percentiles produce robust bounds. Each side is centred and uniformly normalized by its longest robust dimension, then ordered by a 30-bit Morton code and matched by rank. This is equivalent to centre/scale alignment of B to A **for correspondence**. Output positions retain the original composer transforms: B ends where the user placed it, rather than permanently moving to A. There is no rotation estimation or semantic correspondence.
- `N = max(NA, NB)` after crop. The smaller side receives adjacent rank clones with zero opacity; every original is retained exactly once with its original opacity. A hashed seed is stored per pair.
- Two RGBA32F textures hold paired position/opacity, scale/seed, quaternion and DC colour. A custom dyno graph on **`SplatGenerator`** emits world-space `Gsplat`s. This avoids creating another packed source mesh and uses the generator API instead of a `SplatMesh` modifier. Each frame changes only uniforms and calls `updateVersion()`; there is no CPU particle loop or attribute upload.
- Motion uses eased position interpolation plus bounded seeded sinusoidal turbulence. Displacement is at most `1.5 * robustRadius * dissolve * sin²(pi * progress)`. Scales interpolate in log space and shrink to `0.0015 * robustRadius` at full dissolve; rotation uses shortest-path quaternion slerp. Colour and opacity blend deterministically.
- The original source objects render at progress 0 and 1, preserving their original SH, crop, colour, placement and visibility. During the transition only DC colour is morphed. Spark can evaluate source SH, but v1 does not pair or interpolate SH coefficients, so view-dependent colour can change at the endpoint handoff. “Exact” refers to the original Spark-rendered object, including its existing packed precision.
- Worker cancellation, source cache ownership and preview GPU resources are cleaned up on rebuild/close. Spark 0.1.10 lacks a public generator disposal API; `morphDispose.ts` confines its version-specific cache access. Heavy buffers/materials are released; a small amount of shader metadata remains in Spark's inaccessible material cache until page reload.

## Performance and limits

Measured on 2026-10-07:

- Pure pairing of synthetic **200k + 200k** records in Node v24.12.0: **371 / 413 / 419 ms**, three runs. This excludes loading, extraction, worker startup and GPU upload.
- A browser WebGL2 smoke test reported **NVIDIA GeForce RTX 3060 Ti / ANGLE D3D11**. At **960×640**, pixel ratio 1, with 200k synthetic sphere-to-cube particles, 90 animated frames had **13.9 ms median (~72 fps), 14.0 ms p95**. All three sampled progress values (0, 0.5, 1) produced distinct nonempty images without shader/WebGL errors. Fixture generation, pairing and texture setup took **1.93 seconds**. This was VSync-limited development rendering with `preserveDrawingBuffer`; real assets, SH, overdraw, other objects, resolution and browser load will differ.
- Estimated additional attribute storage at 200k pairs: **24.4 MiB GPU textures** plus their **24.4 MiB CPU backing arrays**. During preparation, input/output/sorting buffers add tens of MiB. Original assets/SH, Spark accumulation/sorting targets and framebuffers are additional; this is not a total VRAM measurement.

V1 allows up to **1,000,000 raw splats per asset**, also subject to the GPU texture limit. Empty crops fail with a panel error; hidden endpoints contribute zero opacity. Morton matching is approximate and can connect unrelated parts or create crossings. There is no learned deformation, collision handling, SH interpolation, animation persistence, video recording or animated export. **Kaydet/Export still operate on the original static scene**; morph settings are transient. Start at 200k per side and reduce asset size/resolution if needed.

Validation: `cd frontend`, `npm test` (**226 tests / 32 files passed**), `npm run build` (**passed**, existing bundle-size warning). Tests cover ordering/alignment, unequal-count padding, exact endpoints, deterministic/bounded interpolation, real Spark decoding/crop/transforms/colour, worker ownership/cancellation, playback controls and GPU cache cleanup. A separate browser fixture with two 5,000-splat binary PLYs verified the actual editor controls, module worker, play/pause, cloud/target scrubbing and restoration on close, with no browser errors.
