# Future Directions — Brainstorm

> **Status:** exploratory brainstorm, not a committed roadmap. Captured 2026-05-01
> after first cloud SOTA verification work, to remember ideas worth coming back to.

Grouped by how much the existing pipeline already buys us.

---

## Tier 1 — Rides on what we already have (low-medium effort)

### Object removal
Pipeline already has SAM2 dynamic masks per frame and 3D-anchored Gaussians.
Recipe: SAM2 picks a region (text prompt or click), project each Gaussian
center into that region across views, delete or alpha-fade Gaussians whose
projections fall in masked pixels. Add an "edit" tab to the desktop app.

- **Effort:** medium. Mostly a rendering + filtering pass over an existing checkpoint.
- **Quality:** good for whole-object removal. Edge bleeding is the main artifact;
  smoothable with per-Gaussian opacity decay rather than hard delete.
- **Limitation:** no inpainting — the void shows the original far-side Gaussians.
  See "Object removal + AI inpainting" in Tier 2 for the killer combo.

### Camera path / cinematic playback
`backend/eval/orbit_render.py` already does Catmull-Rom paths through training cams.
Easy extensions:
- User-defined keyframe paths (drag camera in viewer to record)
- Speed ramps (slow-motion using the Fourier trajectory)
- Auto-cinematics ("orbit around the brightest Gaussian", "follow the dynamic
  mask centroid")

- **Effort:** small. Mostly UI work on the viewer.
- **Quality:** great. Renderer is smooth.

### Color grading / tone / SH editing
Each Gaussian stores SH coefficients. Per-channel global scaling, gamma, hue
rotation are 5 lines of GPU shader code each. Pipe a "look" panel into the
desktop app.

- **Effort:** small.
- **Quality:** professional-grade once wired through Spark's render pipeline.

### Compression + LOD
A trained 4D scene is 15-30 GB of PLY. Real-world deployment needs ~10-100×
compression:
- Quantize means/quaternions to 16-bit
- Strip SH degree by importance (per-Gaussian)
- Hierarchical octree with view-dependent loading

- **Effort:** medium-large.
- **Why interesting:** the biggest blocker for scaling beyond local viewer is
  file size.

---

## Tier 2 — AI integration (medium-large effort, big payoff)

### Object removal + AI inpainting (the killer combo)
After removing Gaussians for a masked object, you have a hole. Feed the
rendered hole into a 2D diffusion model (Stable Diffusion inpainting or LaMa),
then re-train new Gaussians from those inpainted views. Tools like
[GaussianEditor](https://buaacyw.github.io/gaussian-editor/) (CVPR 2024) do
exactly this for static scenes; doing it for *4D* with our dynamic masks would
be a real research contribution.

- **Effort:** large. Diffusion models, retraining a Gaussian subset, coordinate
  alignment.
- **Wow factor:** very high. Demos amazingly.

### Text-to-edit ("make the banana red", "add a candle to the table")
LLM picks the target object → SAM2 segments it → either modify SH coefficients
(recolor) or run inpainting (replace). Natural-language UI on top of the
editing primitives above.

- **Effort:** medium *if* object removal + recolor primitives exist.
- **What's hard:** spatial grounding ("on the left side of the banana") needs
  depth + view reasoning.

### AI render upscaling
Train at 720×405, render at the same, then run a 2D super-resolution model
(Real-ESRGAN, SwinIR) on output frames. Effectively gives 1440p/4K outputs
without the training cost.

- **Effort:** small (just an output post-processor).
- **Caveat:** will sharpen artifacts as well as detail. Better than nothing,
  worse than training at higher res.

### Diffusion-guided refinement (research-grade)
Use a 2D diffusion model as a prior during training: "renders should look like
real photos." Improves quality on sparse views especially.
See DreamGaussian, ProlificDreamer.

- **Effort:** very large (changes the training loop).
- **Quality lift:** ~1-2 dB PSNR plausible. Useful for thin / sparse scenes.

---

## Tier 3 — Generative (research-toy quality currently)

### Text-to-4D scene
Type "a dragon flying over a city" → model generates a full 4D Gaussian splat.
Methods exist (4DGen, AnimateGS) but the quality is *low* — recognizable but
visually crude. Probably not yet a feature, more like a tech-demo.

### Image / video-to-4D
Single image or short video → 4D scene. Marginally better than text-to-4D
thanks to having pixel constraints. Still toy quality.

### Honest take on generative
Don't build a generative pipeline as a primary feature *yet*. The quality gap
between "trained on real captures" (current pipeline) and "AI-generated"
(current research) is huge. Generative is a great future direction but isn't
a wow factor today.

---

## Wildcards (not directly suggested, but interesting)

### Mesh extraction → Blender / Unity / DCC tools
Convert Gaussians to a triangle mesh (or hybrid mesh+gauss) for use in standard
DCC tooling. Hot research area (SuGaR for example). Lets splats be used by
people who don't care about Gaussian rendering.

- **Effort:** medium. Published algorithms with reference code exist.
- **Reach:** opens the project to a 100× larger audience (game devs, VFX).

### WebGL / mobile playback
Port Spark renderer to vanilla three.js + browser. Then anyone can view a
trained scene from a URL — no Tauri install.

- **Effort:** medium. Browser memory + bandwidth are the constraints.
- **Reach:** biggest distribution unlock once anyone with a phone link can view.

### Online / streaming training
Incremental scene that updates as new frames arrive. Captures live events.

- **Effort:** very large.

### Relighting (separate albedo from lighting)
Open research area. If we crack it, lighting can be swapped in a captured scene.
Currently best-in-class methods produce mediocre results.

- **Effort:** very large + research bet.

---

## Personal recommendations

If picking *one* direction:

1. **Object removal + AI inpainting (Tier 2).** Nobody has nailed this for 4D
   yet (only static). Existing SAM2 + CoTracker + multi-view is exactly the
   substrate. Demos virally. Opens the door to a "scene editor" product, not
   just a viewer.

2. **Mesh extraction (Wildcard).** Makes scenes useful to a 100× larger
   audience without competing in the rendering arms race.

3. **WebGL / mobile playback (Wildcard).** Biggest distribution unlock — once
   anyone with a phone link can view, the project's reach fundamentally changes.

---

## Note for future-self

Before committing to any of these, run the SOTA verification first
(`docs/SOTA_VERIFICATION.md`) — if the pipeline is not currently SOTA-tier,
fixing that has higher leverage than adding new features.
