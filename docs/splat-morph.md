# Splat morph v1

The Scene Editor previews a deterministic, diffusion-inspired A → particle cloud → B transition entirely in the browser. This is a procedural effect, not a diffusion model. Default settings are full dissolve, full target blend, a six-second duration and seed 42.

## Try it

1. From the repository root, start the existing backend with `scripts\start-backend.bat`. Alternatively, in the prepared `gs4d` environment, run `uvicorn backend.api:app --host 127.0.0.1 --port 8000` as described in the README.
2. In a second terminal, run `cd frontend`, `npm install` if needed, then `npm run dev`. Open **http://localhost:1420**. For the existing desktop launcher, double-click root `run.bat`: it starts the backend and `npm run tauri dev` in two windows.
3. Open **Sahne Editörü**. Open an existing scene with **Aç**, or upload/select a base `.ply` under **Yeni sahne** and click **Oluştur**. Under **Obje ekle**, upload/select another `.ply` and click **Sahneye ekle**. Start with about 200,000 splats per asset.
4. Position, scale, crop and colour the objects as desired. Select **A** on the source object's row and **B** on the target's row. These pair buttons leave the normal inspector selection intact. Keep both objects visible for a complete transition.
5. In the right-hand **Morph** panel, click **Morph hazırla**, wait for the particle count/preparation time, then **Oynat**. Orbit the camera normally; object gizmos are suspended during preview. **Duraklat** pauses, **Başa dön** resets, and the timeline scrubs deterministically.
6. **Önizlemeyi kapat** restores the normal scene. **A ↔ B** swaps the pair; prepare again. Changing seed or an endpoint's geometry/appearance prepares a new pairing. Leaving the Scene Editor pauses playback; deleting an endpoint or closing the scene ends the preview.

## Record a video

1. Open a scene with two visible splat objects, choose their **A** and **B** buttons, and set **Süre (sn)**, **Dissolve**, **Target blend** and **Seed**.
2. Click **Morph hazırla** and wait for the particle count. Position the camera and set the viewport/window size before recording.
3. Click **Videoyu kaydet**. Recording always samples the complete timeline from 0 to 1, regardless of the current preview position. The button shows **Kaydediliyor… %NN**. Morph controls, pair/object editing, editor shortcuts and the camera are locked during recording.
4. Keep this tab visible and its viewport size unchanged. On success the browser downloads **`splat-morph-<seed>.webm`**. Its usual download settings determine the destination or prompt. There is no audio; the video contains the current viewport, without the editor panels or crop selection outline.
5. Click **İptal** to discard the recording without downloading. Completion, cancellation and errors restore the previous timeline position and play/pause state. Leaving the editor or hiding the browser tab cancels and preserves the existing inactive-editor pause behavior. Closing/unmounting the editor also cancels. A viewport resize stops the recording with an error; resize first and retry.

The browser must support `canvas.captureStream`, manual `CanvasCaptureMediaStreamTrack.requestFrame`, and WebM `MediaRecorder`. The encoder preference is VP9, then VP8, then plain WebM, checked with `MediaRecorder.isTypeSupported`. Unsupported browsers report a panel error. Recording uses the existing canvas pixel dimensions; v1 does not resize it or offer an FPS/resolution selector.

### Frame ordering and timing

- The fixed sampling rate defaults to **30 FPS**. For duration `d` and sampling rate `fps`, `intervals = round(d * fps)`, and the schedule contains **`intervals + 1`** samples at `t = i / intervals`, including exact 0 and 1. Six seconds therefore requests 181 frames. Normal wall-clock playback is paused; render stalls never skip or advance a timeline sample.
- Each sample updates the morph uniforms and endpoint visibility. At the first frame and at source/particles/target handoffs, two animation frames allow React to commit recording controls, crop-outline hiding and original-object suppression across the R3F root. Sorting does **not** rely on those waits: a dedicated Spark viewpoint explicitly updates the splats and awaits the public `SparkViewpoint.prepare({ update: false })` GPU readback/worker sort, then calls `gl.render(scene, camera)` and `requestFrame()` in the same turn. This works without `preserveDrawingBuffer`.
- The live R3F render loop and Spark auto-updates are suspended for this session; the recording owns every draw, avoiding intervening live renders and concurrent sorts on its viewpoint. Spark 0.1.10's manual `prepare()` retains an extra temporary accumulator reference; `morphCapture.ts` balances it through the public `releaseAccumulator()` API, separately from the display's own reference. Cancellation stops the encoder/tracks immediately, then lets any in-flight Spark sort settle before releasing its inputs and restoring the live renderer.
- `captureStream(0)` disables periodic capture. The stream and encoder are created only at the first prepared draw, so the stream's implicit initial frame also sees the start of the morph. Capture requests are paced with a minimum `1 / fps` interval: preparation/sorting overlaps that interval, then the final draw waits only for its remaining time. The last frame gets an interval plus a paint opportunity before recorder finalization. Only successful, nonempty recorder output triggers a download.
- **Timing limitation:** samples and seeded trajectories are deterministic; WebM bytes, encoded frame count and timestamps are controlled by the browser's real-time `MediaRecorder`. Rendering/sorting overhead and encoder load can lengthen the output beyond the configured duration or cause browser-level frame drops. This is not an offline constant-frame-rate exporter. Exact encoded duration/frame count would require a timestamp-controlled encoder/muxer beyond the requested MediaRecorder path. Large scenes and high canvas resolutions also increase recording time, memory and file size; chunks are held in memory until completion.

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

V1 allows up to **1,000,000 raw splats per asset**, also subject to the GPU texture limit. Empty crops fail with a panel error; hidden endpoints contribute zero opacity. Morton matching is approximate and can connect unrelated parts or create crossings. There is no learned deformation, collision handling, SH interpolation, animation persistence or animated scene export. **Kaydet/Export still operate on the original static scene**; **Videoyu kaydet** separately downloads the viewport video. Start at 200k per side and reduce asset size/resolution if needed.

Morph settings remain transient. Optional scene-document persistence was assessed and skipped: `backend/compose/models.py` forbids extra fields, and both save routes and disk loading validate `SceneDoc`. Adding morph settings would require changing that backend schema, outside this follow-up's optional frontend-only scope. Older documents continue loading unchanged.

Original preview validation: `cd frontend`, `npm test` (**226 tests / 32 files passed**), `npm run build` (**passed**, existing bundle-size warning). Tests cover ordering/alignment, unequal-count padding, exact endpoints, deterministic/bounded interpolation, real Spark decoding/crop/transforms/colour, worker ownership/cancellation, playback controls and GPU cache cleanup. A separate browser fixture with two 5,000-splat binary PLYs verified the actual editor controls, module worker, play/pause, cloud/target scrubbing and restoration on close, with no browser errors.

Recording follow-up validation (2026-10-07): `npx vitest run` **274 tests / 34 files passed**; `npm run build` **passed**, with the existing bundle-size warning. The final suite used process-local `VITEST_MAX_FORKS=4`, `VITEST_MIN_FORKS=1`, `VITEST_MAX_THREADS=4`, `VITEST_MIN_THREADS=1` after machine-load timeouts; repository test configuration is unchanged. New tests mock MediaRecorder/canvas streams and renderer boundaries to cover exact schedules, codec fallbacks, preparation/pacing/render/capture order, long-recording accumulator balance, errors/cancellation, UI and shortcut locks, download naming, and playback/resource restoration on success, cancellation and unmount. No real-browser GPU/WebM recording smoke test was run for this follow-up; the browser preview measurements above are from v1, not video-encoder benchmarks.
