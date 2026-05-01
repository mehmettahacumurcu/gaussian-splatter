# Frontend Redesign — Design Spec

**Date:** 2026-05-01
**Status:** Draft (awaiting user review)
**Scope:** `4dgs-studio/frontend/src/**`
**Approach:** Option B — layered redesign with shared design-token + primitives layer.

---

## 1. Motivation

The current UI works but is hard to use and not very informative:

- The Static / Dynamic split lives only inside the Submit form; the rest of the
  app (Jobs / Viewer / Analiz / Eval) treats the two pipelines uniformly.
- Presets show one short description line — users have no way to learn *when*
  to pick a preset, *how it differs* from the others, or what hardware it needs.
- Hyperparameters panel has 14+ groups and ~50 fields with cryptic
  `placeholder="preset"` and `title=` tooltips that no one finds.
- Viewer top bar mixes scene loading, engine toggle, and a debug checkbox.
  No visible context about what the user is looking at (mode, training duration,
  PSNR, hyperparams that produced the run).
- Analiz shows raw SVG charts and a JSON dump. No verdict ("is this run going
  well?"), no timeline of the multi-hour preprocessing phases, no events parser.

Goals: **vanilla** (no UI library / icon library / Tailwind), **informative** at
every level (preset → hyperparam → run → events), **clearly split** by pipeline.

## 2. Stack & constraints

- React 19 + TypeScript + Vite + Tauri.
- Vanilla CSS with `:root` custom properties; no preprocessor.
- No new runtime dependencies. Existing `@mkkellogg/gaussian-splats-3d`,
  `@sparkjsdev/spark`, `three` stay.
- No router (single-window Tauri shell — pipeline + tab in `useState`,
  persisted to `localStorage`).
- Mixed Turkish / English UI labels — match existing tone, don't translate
  established Turkish strings ("Sahne adı", "Yeni Job"), but keep technical
  terms English ("Hyperparams", "Preset", "Pipeline").

## 3. Top-level shell

### 3.1 Layout

```
┌──────────────────────────────────────────────────────────────────┐
│  4DGS Studio    [📸 Static 3D] [🎬 4D Dynamic]    Backend ✓ GPU  │
├──────────────────────────────────────────────────────────────────┤
│  Submit   Jobs ●3   Viewer   Analiz   Eval                       │
│  ───────  (active tab indicator inherits pipeline accent)         │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   <tab content scoped to active pipeline>                         │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 State

- `pipeline: "static" | "dynamic"` (replaces today's `mode`); persisted to
  `localStorage["4dgs.pipeline"]`. On first load, falls back to reading the
  legacy `localStorage["4dgs.mode"]` key once and writes it under the new key
  (the old key is then ignored). Default: `"dynamic"` (matches current).
- `tab: Tab` unchanged.
- The pipeline class on `<div class="app">` (`.pipeline-static` /
  `.pipeline-dynamic`) drives `--accent` / `--accent-bg` via CSS, so the entire
  UI tints to the active pipeline without prop-drilling.

### 3.3 Tab scoping rules

| Tab    | Pipeline scope                                                                 |
|--------|--------------------------------------------------------------------------------|
| Submit | Renders `Static3DSubmit` *or* `Dynamic4DSubmit`. Inner mode picker removed.    |
| Jobs   | Filters `listJobs()` by mode. Optional `All / Static / Dynamic` chip override. |
| Viewer | Disk panel filters scenes by `mode` if available; manual job-id input bypasses.|
| Analiz | Scene `<select>` populated from `listJobs()` filtered by pipeline.             |
| Eval   | Same as Analiz.                                                                |

The `mode` field is already on jobs (`JobMode = "static" | "dynamic"` in
`api.ts`). Disk-scene filtering needs the backend to return mode in
`SceneListItem` — if not present, fall back to "show all" with a TODO.

## 4. Design tokens & primitives

### 4.1 New files

```
src/styles/tokens.css       — :root variables (typography, spacing, color, radii)
src/styles/primitives.css   — styles for the primitives below
src/components/ui/
  Card.tsx          — <Card title? actions?>
  Field.tsx         — labeled input/checkbox with optional <InfoButton>
  InfoButton.tsx    — ⓘ icon → toggles a sibling <details> region
  Banner.tsx        — status banner (info / ok / warn / error)
  StatPill.tsx      — compact key-value (e.g. "PSNR 28.4 dB")
  Sidebar.tsx       — right-side panel with collapse toggle
```

`App.css` is *kept* but trimmed: anything that becomes a primitive moves to
`primitives.css`; only legacy tab-specific blocks remain. `tokens.css` extends
the existing `:root` (current vars stay valid for now → low-risk migration).

### 4.2 Token highlights

```css
:root {
  /* Typography scale */
  --fs-11: 11px;  --fs-12: 12px;  --fs-13: 13px;
  --fs-14: 14px;  --fs-16: 16px;  --fs-18: 18px;  --fs-20: 20px;
  --fw-regular: 400; --fw-medium: 500; --fw-semibold: 600;
  --ff-mono: "SF Mono", Consolas, Monaco, monospace;

  /* Spacing scale (4px base) */
  --sp-1: 4px; --sp-2: 8px; --sp-3: 12px; --sp-4: 16px;
  --sp-5: 20px; --sp-6: 24px; --sp-8: 32px;

  /* Radii */
  --radius-sm: 3px; --radius-md: 6px; --radius-lg: 10px;

  /* Semantic color (kept stable) */
  --bg: #0e0e0e; --bg-1: #161616; --bg-2: #1a1a1a; --bg-3: #222;
  --border: #2a2a2a; --border-1: #333;
  --text: #e5e5e5; --text-muted: #888;
  --ok: #78d392; --warn: #d3b478; --err: #e58080;

  /* Per-pipeline accent */
  --accent-static:  #4a8cf0;
  --accent-dynamic: #f08c4a;
  --accent: var(--accent-static);
  --accent-bg: #2a5bdb;
}

.pipeline-dynamic {
  --accent: var(--accent-dynamic);
  --accent-bg: #c2682f;
}
```

### 4.3 Primitive: `InfoButton`

```tsx
<InfoButton>
  <Details>
    {/* arbitrary children — preset details, field help, etc. */}
  </Details>
</InfoButton>
```

- Renders a `ⓘ` glyph button. Click toggles `aria-expanded`; the sibling
  `<Details>` region animates open/closed via `max-height` transition.
- `<Details>` is a styled `<div role="region">` with consistent left-rule and
  reduced font-size. All "info" surfaces in the app use this.

### 4.4 Primitive: `Card`

Replaces ad-hoc `<div style={{background, border, borderRadius, padding}}>`
blocks scattered through `TrainingAnalytics.tsx`, `NvsEvalPanel.tsx`, etc.
Props: `title?`, `actions?`, `tone?: "default" | "ok" | "warn" | "err"`.

## 5. Submit (per pipeline)

### 5.1 Structural changes

- Mode picker (today's `<div class="mode-picker">`) is **removed** — the
  pipeline switch on the top bar drives everything.
- The mode banner becomes a single muted line under the title (not a colored
  box) — pipeline accent on the top bar already conveys mode.
- Each section becomes a `<Card>`: Input, Sahne adı, Preset, Options
  (depth/NVS toggles), Hyperparams (collapsible).

### 5.2 Preset chips with inline Details

Each preset chip gains:

- A `ⓘ` button (right-aligned within the chip header).
- Click → toggles a `<Details>` block under that chip; multiple presets can be
  expanded simultaneously for compare.

**New per-preset metadata** lives in `src/presets.ts` (alongside
`PRESET_DEFAULTS` and `PRESET_BASELINES` — see §6.3 / §9.5):

```ts
// src/presets.ts
export interface PresetMeta {
  useCase: string;          // "sosyal medya, demo videolari"
  vram: string;             // "6 GB+"
  diskEstimate: string;     // "~1.5 GB"
  pickWhen: string[];       // ["production-quality static scenes", ...]
  avoidWhen: string[];      // ["dev iteration on slow GPUs", ...]
  keyHyperparams: string[]; // ["LPIPS λ=0.05", "single-scale schedule", ...]
}

export const PRESET_META: Record<JobMode, Record<string, PresetMeta>> = { /* ... */ };
```

`PresetSpec` (the existing in-component type holding `name`, `badge`,
`duration`, `desc`) stays where it is. `PresetMeta` is rendered inside the
expandable Details block.

A `vsBaseline` line is rendered alongside `PresetMeta` but **not stored** —
it is computed at render time by diffing this preset's `PresetSpec` numeric
fields against the pipeline's reference preset (`balanced` for static,
`full` for dynamic).

### 5.3 Files touched

- `Static3DSubmit.tsx`, `Dynamic4DSubmit.tsx` — wrap sections in `Card`,
  upgrade preset chip to use new `PresetChip` component, add metadata table.
- `JobSubmitPanel.tsx` — strip mode picker, just dispatch to the active
  submit component.

## 6. Hyperparams panel (richer inline)

### 6.1 Behaviour additions

| Feature              | Detail                                                                   |
|----------------------|--------------------------------------------------------------------------|
| Search box           | Filters fields across all groups by case-insensitive substring on label. |
|                      | Matched fields' groups auto-expand.                                      |
| `preset:X` placeholder | Rendered next to the input as a small muted span (not just `placeholder=`). |
|                      | Reads from a `PRESET_DEFAULTS[mode][preset]` table (declarative).        |
| ⓘ on every field    | Click toggles the existing `FieldSpec.help` text inline beneath the field.|
| Override badge       | Each group title shows count of non-empty overrides.                     |
| Reset button         | Clears all overrides with a confirm step ("Reset 3 overrides?").         |

### 6.2 No layout regression

Field grid stays 2-column on wide viewports. The "richer panel" is *behaviour*,
not a different shape.

### 6.3 `PRESET_DEFAULTS` table

A new declarative module `src/presets.ts`:

```ts
export const PRESET_DEFAULTS: Record<JobMode, Record<string, Partial<HyperParams>>> = {
  static: { fast: {...}, balanced: {...}, high: {...}, premium: {...} },
  dynamic: { micro: {...}, smoke: {...}, full: {...}, /* ... */ },
};
```

This is **frontend-only metadata** — values mirror `scripts/static_3dgs.py` and
the dynamic pipeline's preset table for display purposes. There is no contract
that this table stay in sync with the backend (it's a UX hint, not a config
source). A header comment will state this.

### 6.4 Files touched

- `HyperparameterPanel.tsx` — adds search state, override-count derivation,
  reset action, per-field info toggle, preset-default placeholder rendering.

## 7. Jobs (per pipeline)

- Filter by current pipeline by default.
- Add `All / Static / Dynamic` filter chip — pipeline pre-selected.
- Each `JobRow` gains a small mode chip (`📸 static` / `🎬 dynamic`) so rows
  remain self-describing under "All".
- Empty state: "No <static|dynamic> jobs yet — start one from Submit."
- Status badges, progress bar, expand/collapse — unchanged.

## 8. Viewer (per pipeline)

### 8.1 Layout

```
┌── Toolbar ──────────────────────────────────────────────────────┐
│ [scene picker ▾] [Son tamamlanmış] [Diskten ▾]      [⚙ settings]│
├──────────────────────────────────────┬──────────────────────────┤
│                                       │                          │
│         <splat canvas>                │  Scene info (Sidebar)    │
│                                       │                          │
│   FPS · gauss · MB  ← HUD overlay     │                          │
│                                       │                          │
├──────────────────────────────────────┴──────────────────────────┤
│ [⏮] ━━━○━━━ [⏯] [⏭]  speed▾  loop☑   23/90                      │
└──────────────────────────────────────────────────────────────────┘
```

### 8.2 Right sidebar — `<SceneInfo>`

Reads from `getSplatInfo()` + (when available) `getJobSummary()`:

- Header: scene name, mode chip, preset chip.
- Stats: frame count, total size, training duration, final PSNR.
- Hyperparams used: top-N overrides, expandable to see full config.
- Files: ply count, link to download `.zip`.
- Collapsible via a divider tab. Default-open ≥1280px viewport,
  default-closed below.

### 8.3 Settings popover

Anchors to a gear button on the toolbar. Contains:

- Engine: `legacy (mkkellogg)` / `Spark (4DGS)` radio.
- HUD: on / off toggle.
- Single-frame mode: debug toggle (kept, but moved out of the main toolbar).

### 8.4 HUD overlay

Bottom-left of canvas, semi-transparent: `FPS · gauss count · approx VRAM`.
The viewers (`SplatViewer`, `SplatViewerSpark`) already track frame timing —
add a `onPerfTick(stats)` prop they can call.

### 8.5 Status bar

Today's `<div class="viewer-statusbar">` is removed. Its info migrates into the
sidebar header.

### 8.6 Files touched

- `App.tsx` — viewer-tab section restructured to use the new layout.
- `SplatViewer.tsx`, `SplatViewerSpark.tsx` — emit perf stats.
- New `components/ViewerSidebar.tsx`, `components/ViewerSettings.tsx`,
  `components/ViewerHUD.tsx`.

## 9. Analiz (richer)

### 9.1 Layout (top-down)

1. **Header row**: scene `<select>` + "Aktif/son job" + auto-refresh badge.
2. **Health verdict banner** (`<Card tone={ok|warn|err}>`).
3. **Pipeline timeline** (`<Card>`).
4. **Stat pills row** (today's stat cards, restyled via `<StatPill>`).
5. **Charts grid** (today's charts, unchanged content; new `<Card>` wrappers,
   shared cursor line driven by Events selection).
6. **Events log** (parsed, filterable, click-to-jump).

### 9.2 Health verdict — derivation

Pure-frontend computation from the metrics array. Looks at the *most recent
window* (last 20% of metrics, min 30 samples):

- Compute least-squares slope of `loss` vs `iter`.
- Compute fraction of NaN / non-finite samples.
- Compare current `psnr` to a per-preset baseline from `PRESET_BASELINES`
  (declarative, see 9.5).

```
slope < -ε  &&  no NaN          → ✅ Converging
|slope| < ε for ≥20% of window  → ⚠ Stalled
slope > ε  ||  NaN present      → ✗ Diverging
```

`ε` is loss-scale-relative (e.g. `1e-4 * mean_loss`). Constants live in
`src/analytics/health.ts`.

### 9.3 Pipeline timeline

Parses `events.log` for phase transitions (Foundation → COLMAP → Init →
Training → Eval). Backend already emits `phase_changed` events with timestamps.
Each segment is a `<div>` with width proportional to wall-time, labelled with
phase name + duration.

If `events.log` lacks phase-change records (older runs), the timeline shows a
"timeline unavailable for this run" placeholder.

### 9.4 Events log — parsed, filterable

Each line of `events` becomes `{ts, phase, level, message}`. Format inferred
from existing log lines (e.g. `12:04:11 INFO train iter 5000 ...`). Lines that
don't match the regex fall through as `{level: "raw", message: <line>}`.

Render: a virtualized table (no extra dep — manual windowing on a `<ul>`),
filter pills above for phase + level. Clicking a row dispatches a
`cursorIter` that the chart components draw as a vertical line.

### 9.5 `PRESET_BASELINES` — expected PSNR per preset

```ts
export const PRESET_BASELINES: Record<JobMode, Record<string, { psnr: number }>> = {
  static: { fast: {psnr: 23}, balanced: {psnr: 26}, high: {psnr: 28}, premium: {psnr: 30} },
  dynamic: { micro: {psnr: 18}, smoke: {psnr: 22}, full: {psnr: 26}, /* ... */ },
};
```

Initial values are rough; can be refined later from real run data. Same
disclaimer as `PRESET_DEFAULTS`: UX hint, not a contract.

### 9.6 Files touched

- `TrainingAnalytics.tsx` — restructured into sub-components: `<HealthBanner>`,
  `<PipelineTimeline>`, `<EventsLog>`. Existing `<LineChart>` and `<StatCard>`
  kept (with style refresh).
- New `src/analytics/health.ts`, `src/analytics/events.ts`.

## 10. Eval (NVS)

Pure visual pass:

- Wrap metrics card and orbit video block in `<Card>`.
- StatPill row replaces the inline `<Metric>` items.
- Add a small "Δ vs preset baseline" subscript to PSNR/SSIM/LPIPS.
- No structural change to data flow.

## 11. File-tree summary (after redesign)

```
src/
  App.tsx                      ← restructured (pipeline switch, scoped tabs)
  App.css                      ← trimmed (legacy tab CSS only)
  api.ts                       ← unchanged
  styles/
    tokens.css                 ← NEW
    primitives.css             ← NEW
  presets.ts                   ← NEW (PRESET_DEFAULTS, PRESET_BASELINES, PRESET_META)
  analytics/
    health.ts                  ← NEW
    events.ts                  ← NEW
  components/
    ui/                        ← NEW
      Card.tsx
      Field.tsx
      InfoButton.tsx
      Banner.tsx
      StatPill.tsx
      Sidebar.tsx
    Static3DSubmit.tsx         ← updated
    Dynamic4DSubmit.tsx        ← updated
    JobSubmitPanel.tsx         ← simplified (mode picker removed)
    HyperparameterPanel.tsx    ← richer (search, info, overrides, reset)
    JobsList.tsx               ← pipeline filter, mode chip per row
    SplatViewer.tsx            ← emit perf stats
    SplatViewerSpark.tsx       ← emit perf stats
    ViewerSidebar.tsx          ← NEW
    ViewerSettings.tsx         ← NEW (popover)
    ViewerHUD.tsx              ← NEW
    TrainingAnalytics.tsx      ← restructured
    HealthBanner.tsx           ← NEW
    PipelineTimeline.tsx       ← NEW
    EventsLog.tsx              ← NEW
    NvsEvalPanel.tsx           ← visual pass
    TimelineSlider.tsx         ← unchanged
```

## 12. Out of scope

- Backend changes. The redesign is frontend-only. If `SceneListItem` doesn't
  carry `mode`, the disk-panel filter degrades gracefully (TODO marker, no
  blocker).
- Light theme. Aesthetic decision was "refined dark IDE" — single theme.
- Routing / deep-linking. Single-window Tauri shell.
- New runtime dependencies. No charting / icon / UI library.
- Internationalisation. Mixed TR/EN tone preserved as-is; no string-table
  refactor.
- `react-router`, `framer-motion`, Tailwind, shadcn/ui — explicitly excluded.

## 13. Risks

| Risk                                                              | Mitigation                                                                       |
|-------------------------------------------------------------------|----------------------------------------------------------------------------------|
| `PRESET_DEFAULTS` drifting from backend reality                   | Header comment marks it as UX hint; values only used for placeholder display.    |
| Health verdict raising false alarms on legitimate plateaus        | Constants in `health.ts` adjustable; show the slope value alongside the verdict. |
| Older runs lack phase-change events                               | Pipeline timeline degrades to placeholder.                                       |
| Sidebar squeezing canvas on small viewports                       | Auto-collapse below 1280px.                                                      |
| Pipeline-scoping a job that was created cross-pipeline            | "All" filter chip on Jobs is the escape hatch.                                   |
| Removing engine toggle from main toolbar surprises power users    | Settings popover icon is visible; gear glyph + keyboard shortcut documented.     |

## 14. Acceptance criteria

A user opening the app should be able to, without reading docs:

1. See immediately whether they are in Static or Dynamic mode (top-bar accent).
2. Switch pipelines with one click; all tabs filter accordingly.
3. Click any preset's ⓘ and read its full use-case / VRAM / pick-when card.
4. Search hyperparams by name; see what value the preset would have used.
5. Open the Viewer and read a sidebar telling them which scene this is, what
   preset / training duration / PSNR produced it.
6. Open Analiz and see at a glance whether the run is converging, what phase
   it's in, and click an event to jump charts to that iter.

---

## Implementation note

This spec is intentionally implementation-light at the file level — the
implementation plan (next document) will sequence the work, add ordering
constraints, and define the test approach.
