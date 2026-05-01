# Frontend Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Polish the 4DGS Studio frontend per `docs/superpowers/specs/2026-05-01-frontend-redesign-design.md` — promote the Static/Dynamic split to a top-level pipeline switch, add inline preset Details, redesign the hyperparams panel, restructure Viewer and Analiz, and extract a shared design-token + primitives layer.

**Architecture:** Layered redesign (spec §1 Option B). Two new CSS files (`tokens.css`, `primitives.css`), a new `components/ui/` primitives folder, a new `presets.ts` metadata module, two new `analytics/*` helpers. Existing components are refactored on top of the primitives. No new runtime dependencies. No router; pipeline + tab in `useState` with `localStorage` persistence.

**Tech Stack:** React 19, TypeScript ~5.8, Vite 7, vanilla CSS with `:root` custom properties. No tests configured in the frontend today — verification is type-check (`tsc`) + production build (`vite build`) + manual smoke per task.

**Branch:** `feat/frontend-redesign` (already created off `feat/perf-and-quality-fixes`).

**Working dir for all paths:** `C:/Users/TAHA/Desktop/gaussian-splatter/Gaussian Splatter/4dgs-studio/frontend/`

---

## File map (locked in)

```
frontend/src/
  App.tsx                      MODIFY   pipeline switch, scoped tabs, viewer restructure
  App.css                      MODIFY   import tokens/primitives, trim duplicates
  api.ts                       NO CHANGE
  styles/
    tokens.css                 CREATE   :root variables (typography, spacing, color, radii)
    primitives.css             CREATE   styles for ui/ primitives + .pipeline-* accent flip
  presets.ts                   CREATE   PRESET_DEFAULTS, PRESET_BASELINES, PRESET_META
  analytics/
    health.ts                  CREATE   verdict computation
    events.ts                  CREATE   events log parser
  components/
    ui/                        CREATE
      Card.tsx
      Field.tsx
      InfoButton.tsx
      Banner.tsx
      StatPill.tsx
      Sidebar.tsx
    Static3DSubmit.tsx         MODIFY   Cards + PresetChip with Details
    Dynamic4DSubmit.tsx        MODIFY   Cards + PresetChip with Details
    JobSubmitPanel.tsx         MODIFY   strip mode picker (pipeline drives)
    HyperparameterPanel.tsx    MODIFY   search, info, preset placeholders, reset, badges
    PresetChip.tsx             CREATE   shared preset chip with Details (used in both submits)
    JobsList.tsx               MODIFY   pipeline filter chip, mode tag per row
    SplatViewer.tsx            MODIFY   onPerfTick callback
    SplatViewerSpark.tsx       MODIFY   onPerfTick callback
    ViewerSidebar.tsx          CREATE   scene info panel
    ViewerSettings.tsx         CREATE   gear popover (engine/HUD/single-frame)
    ViewerHUD.tsx              CREATE   FPS / gauss / VRAM overlay
    TrainingAnalytics.tsx      MODIFY   composition only (charts logic kept)
    HealthBanner.tsx           CREATE   converging/stalled/diverging banner
    PipelineTimeline.tsx       CREATE   phase wall-time bars
    EventsLog.tsx              CREATE   parsed/filterable rows
    NvsEvalPanel.tsx           MODIFY   wrap in Cards + StatPills, baseline delta
    TimelineSlider.tsx         NO CHANGE
```

---

## Task 1: Design tokens + global imports

**Files:**
- Create: `src/styles/tokens.css`
- Create: `src/styles/primitives.css`
- Modify: `src/main.tsx` (add CSS imports)
- Modify: `src/App.css` (remove duplicated `:root` declarations once tokens.css owns them)

- [ ] **Step 1.1: Create `src/styles/tokens.css`**

```css
/* Design tokens for 4DGS Studio frontend.
   Extends the existing :root in App.css. Keep current vars valid for
   incremental migration. */

:root {
  /* Typography */
  --fs-11: 11px;
  --fs-12: 12px;
  --fs-13: 13px;
  --fs-14: 14px;
  --fs-16: 16px;
  --fs-18: 18px;
  --fs-20: 20px;

  --fw-regular: 400;
  --fw-medium: 500;
  --fw-semibold: 600;

  --ff-mono: "SF Mono", Consolas, Monaco, monospace;
  --ff-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;

  /* Spacing scale (4px base) */
  --sp-1: 4px;
  --sp-2: 8px;
  --sp-3: 12px;
  --sp-4: 16px;
  --sp-5: 20px;
  --sp-6: 24px;
  --sp-8: 32px;

  /* Radii */
  --radius-sm: 3px;
  --radius-md: 6px;
  --radius-lg: 10px;

  /* Per-pipeline accent (defaults to static) */
  --accent-static:  #4a8cf0;
  --accent-static-bg: #2a5bdb;
  --accent-dynamic: #f08c4a;
  --accent-dynamic-bg: #c2682f;

  /* These two are overridden by .pipeline-* on .app */
  --accent: var(--accent-static);
  --accent-bg: var(--accent-static-bg);
}

/* Pipeline accent flip — applied on root .app div */
.pipeline-static {
  --accent: var(--accent-static);
  --accent-bg: var(--accent-static-bg);
}
.pipeline-dynamic {
  --accent: var(--accent-dynamic);
  --accent-bg: var(--accent-dynamic-bg);
}
```

- [ ] **Step 1.2: Create `src/styles/primitives.css`** (empty for now — primitives populate it)

```css
/* Styles for components/ui/* primitives. Populated by Task 2. */
```

- [ ] **Step 1.3: Add imports to `src/main.tsx`**

Read the current file first; the existing line probably already imports `./App.css`. Insert tokens before App.css and primitives after.

```tsx
import "./styles/tokens.css";
import "./App.css";
import "./styles/primitives.css";
```

- [ ] **Step 1.4: Build to verify CSS imports resolve**

Run from `frontend/`:

```bash
npm run build
```

Expected: build succeeds; `tsc` clean; bundle output mentions both new CSS files.

- [ ] **Step 1.5: Commit**

```bash
git add src/styles/tokens.css src/styles/primitives.css src/main.tsx
git commit -m "feat(frontend): add design tokens and primitives CSS scaffold"
```

---

## Task 2: UI primitives

Each primitive is a small, focused component. Single file each. All styles live in `primitives.css`.

**Files:**
- Create: `src/components/ui/Card.tsx`
- Create: `src/components/ui/InfoButton.tsx`
- Create: `src/components/ui/Banner.tsx`
- Create: `src/components/ui/StatPill.tsx`
- Create: `src/components/ui/Sidebar.tsx`
- Create: `src/components/ui/Field.tsx`
- Create: `src/components/ui/index.ts` (re-exports)
- Modify: `src/styles/primitives.css` (append styles)

- [ ] **Step 2.1: `Card.tsx`**

```tsx
import type { ReactNode } from "react";

interface Props {
  title?: ReactNode;
  actions?: ReactNode;
  tone?: "default" | "ok" | "warn" | "err";
  className?: string;
  children: ReactNode;
}

export function Card({ title, actions, tone = "default", className, children }: Props) {
  return (
    <section className={`ui-card ui-card-${tone} ${className ?? ""}`}>
      {(title || actions) && (
        <header className="ui-card-header">
          {title && <h3 className="ui-card-title">{title}</h3>}
          {actions && <div className="ui-card-actions">{actions}</div>}
        </header>
      )}
      <div className="ui-card-body">{children}</div>
    </section>
  );
}
```

- [ ] **Step 2.2: `InfoButton.tsx`**

```tsx
import { useId, useState, type ReactNode } from "react";

interface Props {
  label?: string;       // aria-label, default "More info"
  children: ReactNode;  // content shown when expanded
  initialOpen?: boolean;
}

export function InfoButton({ label = "More info", children, initialOpen = false }: Props) {
  const [open, setOpen] = useState(initialOpen);
  const id = useId();
  return (
    <>
      <button
        type="button"
        className={`ui-info-btn ${open ? "open" : ""}`}
        aria-expanded={open}
        aria-controls={id}
        aria-label={label}
        onClick={() => setOpen((o) => !o)}
      >
        <span aria-hidden>ⓘ</span>
      </button>
      <div
        id={id}
        role="region"
        className={`ui-info-details ${open ? "open" : ""}`}
        hidden={!open}
      >
        {children}
      </div>
    </>
  );
}
```

- [ ] **Step 2.3: `Banner.tsx`**

```tsx
import type { ReactNode } from "react";

type Tone = "info" | "ok" | "warn" | "err";

interface Props {
  tone?: Tone;
  title?: ReactNode;
  children?: ReactNode;
}

const ICON: Record<Tone, string> = { info: "ℹ", ok: "✓", warn: "⚠", err: "✗" };

export function Banner({ tone = "info", title, children }: Props) {
  return (
    <div className={`ui-banner ui-banner-${tone}`} role="status">
      <span className="ui-banner-icon" aria-hidden>{ICON[tone]}</span>
      <div className="ui-banner-body">
        {title && <div className="ui-banner-title">{title}</div>}
        {children && <div className="ui-banner-msg">{children}</div>}
      </div>
    </div>
  );
}
```

- [ ] **Step 2.4: `StatPill.tsx`**

```tsx
import type { ReactNode } from "react";

interface Props {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  tone?: "default" | "accent" | "ok" | "warn" | "err";
}

export function StatPill({ label, value, hint, tone = "default" }: Props) {
  return (
    <div className={`ui-stat ui-stat-${tone}`}>
      <div className="ui-stat-label">{label}</div>
      <div className="ui-stat-value">{value}</div>
      {hint && <div className="ui-stat-hint">{hint}</div>}
    </div>
  );
}
```

- [ ] **Step 2.5: `Sidebar.tsx`**

```tsx
import { useState, type ReactNode } from "react";

interface Props {
  side?: "right" | "left";
  defaultOpen?: boolean;
  width?: number;          // px
  children: ReactNode;
}

export function Sidebar({ side = "right", defaultOpen = true, width = 300, children }: Props) {
  const [open, setOpen] = useState(defaultOpen);
  const collapsedWidth = 18;
  return (
    <aside
      className={`ui-sidebar ui-sidebar-${side} ${open ? "open" : "collapsed"}`}
      style={{ width: open ? width : collapsedWidth }}
    >
      <button
        type="button"
        className="ui-sidebar-toggle"
        aria-label={open ? "Collapse sidebar" : "Expand sidebar"}
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        {open ? (side === "right" ? "▶" : "◀") : (side === "right" ? "◀" : "▶")}
      </button>
      {open && <div className="ui-sidebar-body">{children}</div>}
    </aside>
  );
}
```

- [ ] **Step 2.6: `Field.tsx`** — labeled input with optional info-help

```tsx
import type { ReactNode } from "react";
import { InfoButton } from "./InfoButton";

interface Props {
  label: ReactNode;
  help?: ReactNode;
  hintRight?: ReactNode;        // e.g. "preset: 80000"
  children: ReactNode;          // the input
}

export function Field({ label, help, hintRight, children }: Props) {
  return (
    <label className="ui-field">
      <div className="ui-field-row">
        <span className="ui-field-label">{label}</span>
        {help && <InfoButton>{help}</InfoButton>}
        {hintRight && <span className="ui-field-hint">{hintRight}</span>}
      </div>
      {children}
    </label>
  );
}
```

- [ ] **Step 2.7: `src/components/ui/index.ts`**

```ts
export { Card } from "./Card";
export { InfoButton } from "./InfoButton";
export { Banner } from "./Banner";
export { StatPill } from "./StatPill";
export { Sidebar } from "./Sidebar";
export { Field } from "./Field";
```

- [ ] **Step 2.8: Append styles to `src/styles/primitives.css`**

```css
/* ===== Card ===== */
.ui-card {
  background: var(--bg-1);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  margin-bottom: var(--sp-4);
}
.ui-card-ok   { border-color: rgba(120, 211, 146, 0.4); }
.ui-card-warn { border-color: rgba(211, 178, 120, 0.4); }
.ui-card-err  { border-color: rgba(229, 128, 128, 0.4); }

.ui-card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--sp-3) var(--sp-4);
  border-bottom: 1px solid var(--border);
}
.ui-card-title {
  margin: 0;
  font-size: var(--fs-13);
  font-weight: var(--fw-semibold);
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.ui-card-actions { display: flex; gap: var(--sp-2); }
.ui-card-body { padding: var(--sp-4); }

/* ===== InfoButton ===== */
.ui-info-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 18px;
  height: 18px;
  padding: 0;
  margin-left: var(--sp-1);
  background: transparent;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  border-radius: 50%;
  font-size: var(--fs-12);
  line-height: 1;
}
.ui-info-btn:hover { color: var(--accent); background: var(--bg-3); }
.ui-info-btn.open { color: var(--accent); }
.ui-info-details {
  margin-top: var(--sp-2);
  padding: var(--sp-2) var(--sp-3);
  border-left: 2px solid var(--accent);
  background: var(--bg-2);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  font-size: var(--fs-11);
  color: var(--text-muted);
  line-height: 1.5;
}
.ui-info-details > * + * { margin-top: var(--sp-1); }

/* ===== Banner ===== */
.ui-banner {
  display: flex;
  gap: var(--sp-3);
  align-items: flex-start;
  padding: var(--sp-3) var(--sp-4);
  border: 1px solid;
  border-radius: var(--radius-md);
  font-size: var(--fs-13);
  margin-bottom: var(--sp-4);
}
.ui-banner-info { background: rgba(74, 140, 240, 0.08); border-color: rgba(74, 140, 240, 0.4); color: #b8d4ff; }
.ui-banner-ok   { background: rgba(120, 211, 146, 0.08); border-color: rgba(120, 211, 146, 0.4); color: #b8e8c4; }
.ui-banner-warn { background: rgba(211, 178, 120, 0.08); border-color: rgba(211, 178, 120, 0.4); color: #e8d4b8; }
.ui-banner-err  { background: rgba(229, 128, 128, 0.08); border-color: rgba(229, 128, 128, 0.4); color: #e8b8b8; }
.ui-banner-icon { font-size: var(--fs-16); margin-top: 1px; flex-shrink: 0; }
.ui-banner-title { font-weight: var(--fw-semibold); margin-bottom: var(--sp-1); }
.ui-banner-msg { color: var(--text-muted); }

/* ===== StatPill ===== */
.ui-stat {
  background: var(--bg-1);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: var(--sp-2) var(--sp-3);
  min-width: 110px;
}
.ui-stat-accent { border-color: var(--accent); }
.ui-stat-ok   { border-color: rgba(120, 211, 146, 0.4); }
.ui-stat-warn { border-color: rgba(211, 178, 120, 0.4); }
.ui-stat-err  { border-color: rgba(229, 128, 128, 0.4); }
.ui-stat-label {
  font-size: var(--fs-11);
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.ui-stat-value {
  font-family: var(--ff-mono);
  font-size: var(--fs-16);
  font-weight: var(--fw-semibold);
  color: var(--text);
  margin-top: 2px;
}
.ui-stat-hint { font-size: var(--fs-11); color: var(--text-muted); margin-top: 2px; }

/* ===== Sidebar ===== */
.ui-sidebar {
  position: relative;
  background: var(--bg-1);
  border-left: 1px solid var(--border);
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  transition: width 0.18s ease-out;
}
.ui-sidebar-left { border-left: none; border-right: 1px solid var(--border); }
.ui-sidebar-toggle {
  position: absolute;
  top: var(--sp-2);
  left: 2px;
  width: 14px;
  height: 22px;
  background: var(--bg-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  color: var(--text-muted);
  cursor: pointer;
  font-size: 9px;
  line-height: 1;
  padding: 0;
  z-index: 1;
}
.ui-sidebar-toggle:hover { color: var(--accent); }
.ui-sidebar-body {
  padding: var(--sp-6) var(--sp-4) var(--sp-4);
  overflow-y: auto;
  flex: 1;
}

/* ===== Field ===== */
.ui-field {
  display: flex;
  flex-direction: column;
  gap: var(--sp-1);
}
.ui-field-row {
  display: flex;
  align-items: center;
  gap: var(--sp-1);
}
.ui-field-label {
  font-size: var(--fs-11);
  color: var(--text-muted);
}
.ui-field-hint {
  margin-left: auto;
  font-size: var(--fs-11);
  color: var(--text-muted);
  font-family: var(--ff-mono);
}
.ui-field input,
.ui-field select,
.ui-field textarea {
  padding: 5px 8px;
  background: #0a0a0a;
  border: 1px solid var(--border-1);
  color: var(--text);
  border-radius: var(--radius-sm);
  font-size: var(--fs-12);
  font-family: var(--ff-mono);
}
.ui-field input:focus { outline: none; border-color: var(--accent); }
```

- [ ] **Step 2.9: Build**

```bash
npm run build
```

Expected: clean tsc, vite build succeeds.

- [ ] **Step 2.10: Commit**

```bash
git add src/components/ui src/styles/primitives.css
git commit -m "feat(frontend): add ui primitives (Card, InfoButton, Banner, StatPill, Sidebar, Field)"
```

---

## Task 3: presets.ts metadata module

**Files:**
- Create: `src/presets.ts`

- [ ] **Step 3.1: Create `src/presets.ts`** with three exported tables.

```ts
/**
 * Frontend-only preset metadata.
 *
 * UX hint, NOT a contract with the backend. Backend preset behavior lives in
 * scripts/static_3dgs.py and the dynamic pipeline. Values here are for
 * displaying placeholders, info Details, and PSNR baselines. Drift is fine.
 */
import type { HyperParams, JobMode } from "./api";

export interface PresetMeta {
  useCase: string;
  vram: string;
  diskEstimate: string;
  pickWhen: string[];
  avoidWhen: string[];
  keyHyperparams: string[];
}

// ---- PRESET_DEFAULTS — values used in HyperparameterPanel placeholders ----

export const PRESET_DEFAULTS: Record<JobMode, Record<string, Partial<HyperParams>>> = {
  static: {
    fast:     { iters: 7000,  resolution: "1280x720",  max_gaussians: 100000, lambda_ssim: 0.2 },
    balanced: { iters: 30000, resolution: "1920x1080", max_gaussians: 250000, lambda_ssim: 0.2 },
    high:     { iters: 50000, resolution: "1920x1080", max_gaussians: 500000, lambda_ssim: 0.2 },
    premium:  { iters: 100000,resolution: "2560x1440", max_gaussians: 1000000,lambda_ssim: 0.2 },
  },
  dynamic: {
    micro:       { iters: 200,   resolution: "320x180",  num_timestamps: 5  },
    smoke:       { iters: 500,   resolution: "480x270",  num_timestamps: 10 },
    full:        { iters: 30000, resolution: "640x360",  num_timestamps: 60 },
    high:        { iters: 50000, resolution: "640x360",  num_timestamps: 90, fourier_K: 10, max_gaussians: 60000 },
    ultra:       { iters: 80000, resolution: "720x405",  num_timestamps: 90, fourier_K: 12, hexplane_resolution: 112, mlp_width: 640, mlp_depth: 4, max_gaussians: 80000 },
    ultra_clean: { iters: 80000, resolution: "720x405",  num_timestamps: 90, fourier_K: 12, lambda_aniso: 0.02, dpos_total_cap_frac: 0.05, lambda_rigidity: 0.01 },
    static_max:  { iters: 100000,resolution: "1280x720", num_timestamps: 30, fps: 20, metric3d_model: "metric3d_vit_large", colmap_matching: "exhaustive" },
    cloud:       { iters: 60000, resolution: "1920x1080",num_timestamps: 120 },
  },
};

// ---- PRESET_BASELINES — expected PSNR for the health verdict ----
// Ballpark numbers from prior runs; refined later. UX hint only.

export const PRESET_BASELINES: Record<JobMode, Record<string, { psnr: number }>> = {
  static: {
    fast: { psnr: 23 }, balanced: { psnr: 26 }, high: { psnr: 28 }, premium: { psnr: 30 },
  },
  dynamic: {
    micro: { psnr: 18 }, smoke: { psnr: 22 }, full: { psnr: 26 }, high: { psnr: 27 },
    ultra: { psnr: 28 }, ultra_clean: { psnr: 28 }, static_max: { psnr: 28 }, cloud: { psnr: 28 },
  },
};

// ---- PRESET_META — content for the inline Details block under each chip ----

export const PRESET_META: Record<JobMode, Record<string, PresetMeta>> = {
  static: {
    fast: {
      useCase: "Preview / dev iteration",
      vram: "4 GB+",
      diskEstimate: "~400 MB",
      pickWhen: [
        "İlk bakış — pipeline hızlıca çalışıyor mu kontrol",
        "Düşük VRAM (RTX 2060 ve aşağısı)",
      ],
      avoidWhen: ["Sosyal medyaya çıkacak final render"],
      keyHyperparams: ["7k iter", "1280×720", "100k cap", "LPIPS off"],
    },
    balanced: {
      useCase: "Sosyal medya / demo videoları",
      vram: "6 GB+",
      diskEstimate: "~1.5 GB",
      pickWhen: ["Kalite-süre trade-off önemli", "Standart sahneler"],
      avoidWhen: ["Profesyonel rendering — 'high' tercih et"],
      keyHyperparams: ["30k iter", "1920×1080", "250k cap", "LPIPS λ=0.05"],
    },
    high: {
      useCase: "Profesyonel rendering",
      vram: "8 GB+",
      diskEstimate: "~3 GB",
      pickWhen: ["Production-quality static scenes", "≥8GB VRAM kart"],
      avoidWhen: ["Dev iteration", "<8GB VRAM"],
      keyHyperparams: ["50k iter", "1920×1080", "500k cap", "LPIPS λ=0.10", "Multires schedule"],
    },
    premium: {
      useCase: "4dv.ai-tier final çıktı",
      vram: "12 GB+",
      diskEstimate: "~6 GB",
      pickWhen: ["En iyi kalite önemli", "RTX 3090/4090"],
      avoidWhen: ["8GB ve altı kartlar", "Hızlı iterasyon"],
      keyHyperparams: ["100k iter", "2560×1440", "1M cap", "LPIPS λ=0.15", "3-tier multires"],
    },
  },
  dynamic: {
    micro: {
      useCase: "Pre-flight smoke test",
      vram: "4 GB+",
      diskEstimate: "~100 MB",
      pickWhen: ["Pipeline'ı çalıştığını doğrula", "Yeni eklediğin koddan kuşkun var"],
      avoidWhen: ["Demo veya analiz amaçlı kullanma"],
      keyHyperparams: ["200 iter", "320×180", "5 timestamps"],
    },
    smoke: {
      useCase: "Pipeline sağlık kontrolü",
      vram: "4 GB+",
      diskEstimate: "~250 MB",
      pickWhen: ["Bir foundation modeli güncelledin", "Hızlı sanity check"],
      avoidWhen: ["Görsel kaliteye bakacaksan"],
      keyHyperparams: ["500 iter", "480×270", "10 timestamps"],
    },
    full: {
      useCase: "RTX 3060 Ti default — gerçek 4D çıktısı",
      vram: "8 GB+",
      diskEstimate: "~2 GB",
      pickWhen: ["Standart 4D sahne", "8GB civarı kart"],
      avoidWhen: ["12GB+ kartın varsa 'high' kullan"],
      keyHyperparams: ["30k iter", "640×360", "60 timestamps"],
    },
    high: {
      useCase: "Enhanced 4D — daha keskin motion",
      vram: "10 GB+",
      diskEstimate: "~3 GB",
      pickWhen: ["RTX 3080/4070 üstü kart", "Foureri trajectory'i denemek"],
      avoidWhen: ["8GB altı"],
      keyHyperparams: ["50k iter", "640×360", "90 timestamps", "Fourier K=10", "N cap 60k"],
    },
    ultra: {
      useCase: "Yüksek kalite 4D — uzun training",
      vram: "12 GB+",
      diskEstimate: "~5 GB",
      pickWhen: ["RTX 3090/4090", "Final çıktı"],
      avoidWhen: ["12GB altı kartlar — OOM riski"],
      keyHyperparams: ["80k iter", "720×405", "HexPlane 112/56", "MLP 640/4", "K=12", "N cap 80k"],
    },
    ultra_clean: {
      useCase: "Ultra ama anti-streak (v3.8)",
      vram: "12 GB+",
      diskEstimate: "~5 GB",
      pickWhen: ["Streak/spike artifact'ı görüyorsan", "Temiz motion önemli"],
      avoidWhen: ["İlk denemede ultra'yı dene, gerek yoksa atla"],
      keyHyperparams: ["aniso reg", "Δpos clamp", "rigid 5×", "fourier_reg 10×"],
    },
    static_max: {
      useCase: "Eski statik-leaning (legacy)",
      vram: "12 GB+",
      diskEstimate: "~6 GB",
      pickWhen: ["v3.9 davranışını test ediyorsun"],
      avoidWhen: ["Gerçek statik sahne için Static 3D modunu kullan"],
      keyHyperparams: ["fps=20", "vit_large depth", "COLMAP exhaustive"],
    },
    cloud: {
      useCase: "RunPod / RTX 4090 cloud GPU",
      vram: "24 GB",
      diskEstimate: "~8 GB",
      pickWhen: ["Cloud kart kiraladın", "Yüksek çözünürlük gerekli"],
      avoidWhen: ["Local 8GB kartta kullanma — OOM"],
      keyHyperparams: ["60k iter", "1920×1080", "120 timestamps"],
    },
  },
};

// ---- Reference preset per pipeline (for vsBaseline diff) ----
export const REFERENCE_PRESET: Record<JobMode, string> = {
  static: "balanced",
  dynamic: "full",
};
```

- [ ] **Step 3.2: Build**

```bash
npm run build
```

Expected: clean.

- [ ] **Step 3.3: Commit**

```bash
git add src/presets.ts
git commit -m "feat(frontend): add presets.ts (defaults, baselines, metadata)"
```

---

## Task 4: Top-level shell — pipeline switch + scoped tabs

**Files:**
- Modify: `src/App.tsx` (top-bar + state plumbing)
- Modify: `src/App.css` (top-bar styling for new pipeline switch)

- [ ] **Step 4.1: Read `src/App.tsx`** (already done in brainstorming context). Plan the diff:

  - Replace `mode` state with `pipeline` state.
  - Add migration: read legacy `4dgs.mode` once, write to `4dgs.pipeline`.
  - Apply `pipeline-static` / `pipeline-dynamic` class on `.app` div.
  - Move pipeline switch into the top bar (inline next to brand).
  - `JobSubmitPanel` is now invoked with `pipeline` (rename prop in step 5).

- [ ] **Step 4.2: Replace state + persistence in `App.tsx`**

In `App.tsx`, replace the existing `mode` state block with:

```tsx
const [pipeline, setPipeline] = useState<JobMode>(() => {
  try {
    const saved = window.localStorage.getItem("4dgs.pipeline");
    if (saved === "static" || saved === "dynamic") return saved;
    // One-time migration from legacy key
    const legacy = window.localStorage.getItem("4dgs.mode");
    if (legacy === "static" || legacy === "dynamic") {
      window.localStorage.setItem("4dgs.pipeline", legacy);
      return legacy;
    }
  } catch { /* ignore */ }
  return "dynamic";
});
const handlePipelineChange = useCallback((p: JobMode) => {
  setPipeline(p);
  try { window.localStorage.setItem("4dgs.pipeline", p); } catch { /* ignore */ }
}, []);
```

Remove the old `mode` state and `handleModeChange`.

- [ ] **Step 4.3: Apply pipeline class to root**

Change the root JSX in `App.tsx`:

```tsx
return (
  <div className={`app pipeline-${pipeline}`}>
    {/* ... */}
  </div>
);
```

- [ ] **Step 4.4: Add pipeline switch to top bar**

Inside `<header className="app-topbar">`, between `app-brand` and `app-tabs`, insert:

```tsx
<div className="pipeline-switch" role="tablist" aria-label="Pipeline">
  <button
    type="button"
    role="tab"
    aria-selected={pipeline === "static"}
    className={`pipeline-switch-btn ${pipeline === "static" ? "active" : ""}`}
    onClick={() => handlePipelineChange("static")}
  >
    📸 Static 3D
  </button>
  <button
    type="button"
    role="tab"
    aria-selected={pipeline === "dynamic"}
    className={`pipeline-switch-btn ${pipeline === "dynamic" ? "active" : ""}`}
    onClick={() => handlePipelineChange("dynamic")}
  >
    🎬 4D Dynamic
  </button>
</div>
```

- [ ] **Step 4.5: Pass `pipeline` instead of `mode` to `JobSubmitPanel`**

Change:

```tsx
{tab === "submit" && (
  <JobSubmitPanel
    pipeline={pipeline}
    onJobSubmitted={handleJobSubmitted}
  />
)}
```

(The `JobSubmitPanel` interface change happens in Task 5.)

- [ ] **Step 4.6: Append CSS for the pipeline switch in `App.css`**

Append at end of `App.css`:

```css
/* Pipeline switch on top bar */
.pipeline-switch {
  display: inline-flex;
  background: var(--bg-2);
  padding: 3px;
  border-radius: var(--radius-md);
  border: 1px solid var(--border);
  gap: 2px;
}
.pipeline-switch-btn {
  padding: 4px 12px;
  background: transparent;
  border: 1px solid transparent;
  color: var(--text-muted);
  cursor: pointer;
  border-radius: var(--radius-sm);
  font-size: var(--fs-12);
  font-weight: var(--fw-medium);
  transition: background 0.12s, color 0.12s, border-color 0.12s;
}
.pipeline-switch-btn:hover { color: var(--text); background: var(--bg-3); }
.pipeline-switch-btn.active {
  color: var(--text);
  background: var(--bg-3);
  border-color: var(--accent);
}

/* Tab indicator inherits pipeline accent */
.tab-btn.active {
  background: var(--accent-bg);
  border-color: var(--accent);
}
```

- [ ] **Step 4.7: Build**

```bash
npm run build
```

Expected: clean. Note — `JobSubmitPanel` will fail TS check because its `mode` prop was renamed but its file isn't yet updated. Apply the next task immediately, or temporarily keep both prop names. To keep build green, do this in App.tsx step 4.5:

```tsx
<JobSubmitPanel
  mode={pipeline}
  onModeChange={handlePipelineChange}
  onJobSubmitted={handleJobSubmitted}
/>
```

(Same as before — props renamed in Task 5.)

Re-run build:

```bash
npm run build
```

Expected: clean.

- [ ] **Step 4.8: Commit**

```bash
git add src/App.tsx src/App.css
git commit -m "feat(frontend): pipeline switch on top bar, localStorage migration"
```

---

## Task 5: Submit redesign — drop inner mode picker, shared `PresetChip`, Cards

**Files:**
- Modify: `src/components/JobSubmitPanel.tsx`
- Create: `src/components/PresetChip.tsx`
- Modify: `src/components/Static3DSubmit.tsx`
- Modify: `src/components/Dynamic4DSubmit.tsx`
- Modify: `src/App.tsx` (re-update prop name now that downstream is updated)
- Modify: `src/App.css` (add `.preset-chip` Details styling tweaks; remove obsolete mode-picker CSS)

- [ ] **Step 5.1: Create `src/components/PresetChip.tsx`**

```tsx
import { useState } from "react";
import { InfoButton } from "./ui/InfoButton";
import { PRESET_META, PRESET_DEFAULTS, REFERENCE_PRESET } from "../presets";
import type { JobMode } from "../api";

interface Props {
  pipeline: JobMode;
  presetId: string;
  name: string;
  badge?: string;
  duration: string;
  desc: string;
  selected: boolean;
  onSelect: () => void;
}

export function PresetChip({
  pipeline, presetId, name, badge, duration, desc, selected, onSelect,
}: Props) {
  const meta = PRESET_META[pipeline]?.[presetId];
  const refId = REFERENCE_PRESET[pipeline];
  const refDefaults = PRESET_DEFAULTS[pipeline]?.[refId];
  const myDefaults = PRESET_DEFAULTS[pipeline]?.[presetId];

  return (
    <label className={`preset-chip ${selected ? "active" : ""}`}>
      <div className="preset-chip-row">
        <input
          type="radio"
          checked={selected}
          onChange={onSelect}
        />
        <div className="preset-chip-content">
          <div className="preset-name">
            {name} {badge && <span aria-hidden>{badge}</span>}{" "}
            <span className="preset-duration">({duration})</span>
          </div>
          <div className="preset-desc">{desc}</div>
        </div>
        {meta && <InfoButton label={`${name} preset details`}>{renderDetails(meta, presetId, refId, myDefaults, refDefaults)}</InfoButton>}
      </div>
    </label>
  );
}

function renderDetails(
  meta: import("../presets").PresetMeta,
  presetId: string,
  refId: string,
  myDefaults: Partial<import("../api").HyperParams> | undefined,
  refDefaults: Partial<import("../api").HyperParams> | undefined,
) {
  return (
    <>
      <div><strong>Use case:</strong> {meta.useCase}</div>
      <div><strong>VRAM:</strong> {meta.vram} · <strong>Disk:</strong> {meta.diskEstimate}</div>
      <div><strong>Key hyperparams:</strong> {meta.keyHyperparams.join(" · ")}</div>
      <div>
        <strong>Pick when:</strong>
        <ul>{meta.pickWhen.map((s) => <li key={s}>{s}</li>)}</ul>
      </div>
      <div>
        <strong>Avoid when:</strong>
        <ul>{meta.avoidWhen.map((s) => <li key={s}>{s}</li>)}</ul>
      </div>
      {presetId !== refId && myDefaults && refDefaults && (
        <div className="preset-vs">
          <strong>vs {refId}:</strong> {diffSummary(myDefaults, refDefaults)}
        </div>
      )}
    </>
  );
}

function diffSummary(
  a: Partial<import("../api").HyperParams>,
  b: Partial<import("../api").HyperParams>,
): string {
  const parts: string[] = [];
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]) as Set<keyof import("../api").HyperParams>;
  for (const k of keys) {
    if (a[k] !== undefined && a[k] !== b[k]) {
      parts.push(`${String(k)}: ${b[k] ?? "—"} → ${a[k]}`);
    }
  }
  return parts.length ? parts.join(" · ") : "(same as reference)";
}
```

- [ ] **Step 5.2: Update `Static3DSubmit.tsx`** — replace inline preset chip JSX with `<PresetChip>`, wrap sections in `<Card>`. Read the current file first to see the exact JSX.

Replace the contents of `Static3DSubmit.tsx` with:

```tsx
import { useMemo, useState } from "react";
import type { HyperParams, ProcessResponse, StaticPreset } from "../api";
import { submitJob } from "../api";
import { HyperparameterPanel } from "./HyperparameterPanel";
import { Card } from "./ui/Card";
import { PresetChip } from "./PresetChip";

interface Props {
  onJobSubmitted: (response: ProcessResponse, sceneName: string) => void;
}

interface PresetSpec {
  id: StaticPreset;
  name: string;
  badge?: string;
  duration: string;
  desc: string;
}

const PRESETS: PresetSpec[] = [
  { id: "fast",     name: "Fast",     badge: "⚡", duration: "5-8 dk",   desc: "7k iter · 1280×720 · 100k cap · LPIPS off · preview kalite" },
  { id: "balanced", name: "Balanced", badge: "⭐", duration: "30-45 dk", desc: "30k iter · 1920×1080 · 250k cap · LPIPS 0.05 · sosyal medya" },
  { id: "high",     name: "High",     badge: "🎯", duration: "2-3 saat", desc: "50k iter · 1920×1080 · 500k cap · LPIPS 0.10 · multires schedule · profesyonel" },
  { id: "premium",  name: "Premium",  badge: "🔥", duration: "6-10 saat",desc: "100k iter · 2560×1440 · 1M cap · LPIPS 0.15 · 3-tier multires · 4dv.ai-tier" },
];

export function Static3DSubmit({ onJobSubmitted }: Props) {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [scene, setScene] = useState("");
  const [preset, setPreset] = useState<StaticPreset>("balanced");
  const [useDepthSupervision, setUseDepthSupervision] = useState(true);
  const [nvsEval, setNvsEval] = useState(true);
  const [hyperparams, setHyperparams] = useState<HyperParams>({});
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const handleFile = (f: File | null) => {
    setVideoFile(f);
    if (f && !scene) {
      const base = f.name.replace(/\.[^.]+$/, "");
      const safe = base.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 40);
      setScene(safe || "scene_" + Date.now().toString(36));
    }
  };

  const videoSize = useMemo(() => {
    if (!videoFile) return null;
    return `${(videoFile.size / 1024 / 1024).toFixed(1)} MB`;
  }, [videoFile]);

  const canSubmit = scene.trim() !== "" && !submitting;

  const handleSubmit = async () => {
    if (!scene.trim()) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      const res = await submitJob(videoFile, {
        scene: scene.trim(),
        mode: "static",
        preset,
        skip_foundation: !useDepthSupervision,
        nvs_eval: nvsEval,
        hyperparams,
      });
      onJobSubmitted(res, scene.trim());
    } catch (e) {
      setSubmitError(String(e));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="submit-panel">
      <h2 className="submit-title">Yeni Static 3D Job</h2>
      <p className="submit-context-line">
        📸 Photo set / sparse view — 4D dynamic features (deformation, Fourier, motion regs) kapalı.
      </p>

      <Card title="Input">
        <div className="file-picker">
          <input
            type="file"
            accept="video/mp4,video/quicktime,video/*"
            onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
            id="video-input-static"
          />
          <label htmlFor="video-input-static" className="file-picker-btn">
            {videoFile ? "Değiştir" : "Video seç (opsiyonel)"}
          </label>
          {videoFile && (
            <div className="file-info">
              <div className="file-name">{videoFile.name}</div>
              <div className="file-size">{videoSize}</div>
            </div>
          )}
        </div>
        <p className="submit-hint">
          <strong>Photo set varsa video gerek yok</strong>:{" "}
          <code>data/&lt;sahne&gt;/images/IMG_*.jpg</code> klasörünü hazırla,
          sahne adını gir, submit. Video varsa frame'lere ayrılır.
        </p>
      </Card>

      <Card title="Sahne adı">
        <input
          type="text"
          className="submit-input"
          value={scene}
          onChange={(e) => setScene(e.target.value)}
          placeholder="örn. truck, garden, my_object"
        />
        <p className="submit-hint">
          <code>data/&lt;scene&gt;/</code> altında çalışır. Aynı isim varsa üzerine yazar.
        </p>
      </Card>

      <Card title="Preset">
        <div className="preset-row">
          {PRESETS.map((p) => (
            <PresetChip
              key={p.id}
              pipeline="static"
              presetId={p.id}
              name={p.name}
              badge={p.badge}
              duration={p.duration}
              desc={p.desc}
              selected={preset === p.id}
              onSelect={() => setPreset(p.id)}
            />
          ))}
        </div>
      </Card>

      <Card title="Options">
        <label className="submit-checkbox">
          <input
            type="checkbox"
            checked={useDepthSupervision}
            onChange={(e) => setUseDepthSupervision(e.target.checked)}
          />
          Depth supervision (Metric3D) — önerilir
          <span className="submit-hint-inline">
            — geometric prior + sparse-view yardımcısı. Kapatırsan sadece RGB+SSIM+LPIPS.
          </span>
        </label>
        <label className="submit-checkbox" style={{ marginTop: 6 }}>
          <input
            type="checkbox"
            checked={nvsEval}
            onChange={(e) => setNvsEval(e.target.checked)}
          />
          NVS Evaluation — önerilir
          <span className="submit-hint-inline">
            — held-out PSNR/SSIM/LPIPS + orbit mp4. Eval tab'ında görülür.
          </span>
        </label>
      </Card>

      <Card title="Hiperparametreler" actions={
        <button
          className="btn-secondary"
          onClick={() => setShowAdvanced(!showAdvanced)}
          type="button"
        >
          {showAdvanced ? "▾ Gizle" : "▸ Göster"}
        </button>
      }>
        {showAdvanced && (
          <HyperparameterPanel
            value={hyperparams}
            onChange={setHyperparams}
            mode="static"
            preset={preset}
          />
        )}
        {!showAdvanced && (
          <p className="submit-hint">
            "{preset}" preset default'larını kullan. Override eklemek için Göster.
          </p>
        )}
      </Card>

      <div className="submit-actions">
        <button
          className="btn-primary submit-btn"
          disabled={!canSubmit}
          onClick={handleSubmit}
        >
          {submitting ? "Gönderiliyor..." : "Static 3D Job başlat"}
        </button>
        {submitError && <div className="submit-error">Hata: {submitError}</div>}
      </div>
    </div>
  );
}
```

- [ ] **Step 5.3: Update `Dynamic4DSubmit.tsx`** — analogous transform.

Replace its contents with:

```tsx
import { useMemo, useState } from "react";
import type { DynamicPreset, HyperParams, ProcessResponse } from "../api";
import { submitJob } from "../api";
import { HyperparameterPanel } from "./HyperparameterPanel";
import { Card } from "./ui/Card";
import { PresetChip } from "./PresetChip";

interface Props {
  onJobSubmitted: (response: ProcessResponse, sceneName: string) => void;
}

interface PresetSpec {
  id: DynamicPreset;
  name: string;
  badge?: string;
  duration: string;
  desc: string;
}

const PRESETS: PresetSpec[] = [
  { id: "micro",       name: "Micro",       badge: "⚡", duration: "30 sn–10 dk", desc: "200 iter · 320×180 · 5 ts · dev iteration / preflight smoke" },
  { id: "smoke",       name: "Smoke",       badge: "💨", duration: "5-15 dk",     desc: "500 iter · 480×270 · 10 ts · pipeline sağlık check" },
  { id: "full",        name: "Full",        badge: "🟢", duration: "30-60 dk",    desc: "30k iter · 640×360 · 60 ts · 3060 Ti default" },
  { id: "high",        name: "High",        badge: "⭐", duration: "3-4 saat",    desc: "50k iter · 640×360 · 90 ts · Fourier K=10 · N cap 60k · enhanced" },
  { id: "ultra",       name: "Ultra",       badge: "🔥", duration: "6-9 saat",    desc: "80k iter · 720×405 · 90 ts · HexPlane 112/56 · MLP 640/4 · K=12 · N cap 80k" },
  { id: "ultra_clean", name: "Ultra Clean", badge: "✨", duration: "7-9 saat",    desc: "v3.8 anti-streak · aniso reg + sıkı dpos clamp + rigid 5× + fourier_reg 10×" },
  { id: "static_max",  name: "Static Max",  badge: "🎯", duration: "10-13 saat",  desc: "v3.9 4D ama statik-leaning · fps=20 + vit_large depth + COLMAP exhaustive" },
  { id: "cloud",       name: "Cloud",       badge: "☁",  duration: "RunPod",      desc: "60k iter · 1920×1080 · 120 ts · cloud GPU" },
];

export function Dynamic4DSubmit({ onJobSubmitted }: Props) {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [scene, setScene] = useState("");
  const [preset, setPreset] = useState<DynamicPreset>("smoke");
  const [skipFoundation, setSkipFoundation] = useState(false);
  const [nvsEval, setNvsEval] = useState(true);
  const [hyperparams, setHyperparams] = useState<HyperParams>({});
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const handleFile = (f: File | null) => {
    setVideoFile(f);
    if (f && !scene) {
      const base = f.name.replace(/\.[^.]+$/, "");
      const safe = base.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 40);
      setScene(safe || "scene_" + Date.now().toString(36));
    }
  };

  const videoSize = useMemo(() => {
    if (!videoFile) return null;
    return `${(videoFile.size / 1024 / 1024).toFixed(1)} MB`;
  }, [videoFile]);

  const canSubmit = scene.trim() !== "" && !submitting;

  const handleSubmit = async () => {
    if (!scene.trim()) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      const res = await submitJob(videoFile, {
        scene: scene.trim(),
        mode: "dynamic",
        preset,
        skip_foundation: skipFoundation,
        nvs_eval: nvsEval,
        hyperparams,
      });
      onJobSubmitted(res, scene.trim());
    } catch (e) {
      setSubmitError(String(e));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="submit-panel">
      <h2 className="submit-title">Yeni 4D Dynamic Job</h2>
      <p className="submit-context-line">
        🎬 Zaman-dinamik sahneler. Deformation MLP + per-Gaussian Fourier trajectory
        + foundation modeller (depth/track/flow) aktif. Multi-view auto-detect:{" "}
        <code>data/&lt;sahne&gt;/videos/cam*.mp4</code>.
      </p>

      <Card title="Video dosyası">
        <div className="file-picker">
          <input
            type="file"
            accept="video/mp4,video/quicktime,video/*"
            onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
            id="video-input-dynamic"
          />
          <label htmlFor="video-input-dynamic" className="file-picker-btn">
            {videoFile ? "Değiştir" : "Video seç"}
          </label>
          {videoFile && (
            <div className="file-info">
              <div className="file-name">{videoFile.name}</div>
              <div className="file-size">{videoSize}</div>
            </div>
          )}
        </div>
        <p className="submit-hint">
          MP4 veya MOV. Multi-view scene için sahne önceden hazır ise upload skip edilebilir.
        </p>
      </Card>

      <Card title="Sahne adı">
        <input
          type="text"
          className="submit-input"
          value={scene}
          onChange={(e) => setScene(e.target.value)}
          placeholder="örn. banana_demo, flame_steak"
        />
      </Card>

      <Card title="Preset">
        <div className="preset-row">
          {PRESETS.map((p) => (
            <PresetChip
              key={p.id}
              pipeline="dynamic"
              presetId={p.id}
              name={p.name}
              badge={p.badge}
              duration={p.duration}
              desc={p.desc}
              selected={preset === p.id}
              onSelect={() => setPreset(p.id)}
            />
          ))}
        </div>
      </Card>

      <Card title="Options">
        <label className="submit-checkbox">
          <input
            type="checkbox"
            checked={skipFoundation}
            onChange={(e) => setSkipFoundation(e.target.checked)}
          />
          Foundation modelleri atla (Metric3D / CoTracker / Farneback / RAFT)
          <span className="submit-hint-inline">— micro/smoke için önerilir</span>
        </label>
        <label className="submit-checkbox" style={{ marginTop: 6 }}>
          <input
            type="checkbox"
            checked={nvsEval}
            onChange={(e) => setNvsEval(e.target.checked)}
          />
          NVS Evaluation — önerilir
          <span className="submit-hint-inline">
            — held-out PSNR/SSIM/LPIPS + orbit mp4
          </span>
        </label>
      </Card>

      <Card title="Hiperparametreler" actions={
        <button
          className="btn-secondary"
          onClick={() => setShowAdvanced(!showAdvanced)}
          type="button"
        >
          {showAdvanced ? "▾ Gizle" : "▸ Göster"}
        </button>
      }>
        {showAdvanced && (
          <HyperparameterPanel
            value={hyperparams}
            onChange={setHyperparams}
            mode="dynamic"
            preset={preset}
          />
        )}
        {!showAdvanced && (
          <p className="submit-hint">
            "{preset}" preset default'larını kullan. Override eklemek için Göster.
          </p>
        )}
      </Card>

      <div className="submit-actions">
        <button
          className="btn-primary submit-btn"
          disabled={!canSubmit}
          onClick={handleSubmit}
        >
          {submitting ? "Gönderiliyor..." : "4D Dynamic Job başlat"}
        </button>
        {submitError && <div className="submit-error">Hata: {submitError}</div>}
      </div>
    </div>
  );
}
```

- [ ] **Step 5.4: Simplify `JobSubmitPanel.tsx`** — drop the mode picker.

Replace its contents:

```tsx
import type { JobMode, ProcessResponse } from "../api";
import { Static3DSubmit } from "./Static3DSubmit";
import { Dynamic4DSubmit } from "./Dynamic4DSubmit";

interface Props {
  pipeline: JobMode;
  onJobSubmitted: (response: ProcessResponse, sceneName: string) => void;
}

export function JobSubmitPanel({ pipeline, onJobSubmitted }: Props) {
  return pipeline === "static"
    ? <Static3DSubmit onJobSubmitted={onJobSubmitted} />
    : <Dynamic4DSubmit onJobSubmitted={onJobSubmitted} />;
}
```

- [ ] **Step 5.5: Update `App.tsx`** — pass `pipeline` to `JobSubmitPanel` (its props now match).

```tsx
{tab === "submit" && (
  <JobSubmitPanel
    pipeline={pipeline}
    onJobSubmitted={handleJobSubmitted}
  />
)}
```

(If the temporary `mode={pipeline}` form was used in 4.7, remove it.)

- [ ] **Step 5.6: Append CSS to `App.css`** for new submit-panel chrome.

Append:

```css
.submit-context-line {
  font-size: var(--fs-12);
  color: var(--text-muted);
  margin: -8px 0 var(--sp-4);
}
.preset-chip-row {
  display: flex;
  align-items: flex-start;
  gap: var(--sp-2);
  width: 100%;
}
.preset-chip-content { flex: 1; min-width: 0; }
.preset-chip ul {
  margin: 4px 0 0;
  padding-left: 18px;
  font-size: var(--fs-11);
}
.preset-chip li { margin-bottom: 2px; }
.preset-vs {
  margin-top: 6px;
  padding-top: 6px;
  border-top: 1px dashed var(--border);
  font-family: var(--ff-mono);
}
```

Also remove the now-dead `.submit-panel-mode-banner`, `.submit-panel-mode-static`, `.submit-panel-mode-dynamic`, `.mode-picker`, `.mode-btn*` rules in `App.css` (they are no longer referenced).

- [ ] **Step 5.7: Build**

```bash
npm run build
```

Expected: clean.

- [ ] **Step 5.8: Commit**

```bash
git add src/components/PresetChip.tsx src/components/JobSubmitPanel.tsx \
        src/components/Static3DSubmit.tsx src/components/Dynamic4DSubmit.tsx \
        src/App.tsx src/App.css
git commit -m "feat(frontend): submit redesign — Cards + PresetChip with inline Details"
```

---

## Task 6: Hyperparams panel — search, info, preset placeholders, override badges, reset

**Files:**
- Modify: `src/components/HyperparameterPanel.tsx`
- Modify: `src/App.css` (panel-specific tweaks)

- [ ] **Step 6.1: Read `src/components/HyperparameterPanel.tsx`** to confirm current structure (already done in brainstorming context).

- [ ] **Step 6.2: Update `HyperparameterPanel.tsx`**

Replace its body. Key changes:

- Import `PRESET_DEFAULTS` from `../presets`.
- Add `search` state.
- For each field, compute `presetDefault = PRESET_DEFAULTS[mode][preset]?.[field.key]`.
- For each field, render an `InfoButton` with `field.help` (only when help present).
- Filter groups/fields when `search` is non-empty.
- Auto-expand groups whose fields match the search.
- Add a header reset button that wipes all overrides.

Replace the entire file with:

```tsx
import { useEffect, useMemo, useState } from "react";
import type { HyperParams, JobMode } from "../api";
import { PRESET_DEFAULTS } from "../presets";
import { InfoButton } from "./ui/InfoButton";

interface Props {
  value: HyperParams;
  onChange: (next: HyperParams) => void;
  mode?: JobMode;
  preset?: string;
}

interface FieldSpec {
  key: keyof HyperParams;
  label: string;
  placeholder: string;
  help?: string;
  kind: "number" | "string";
  min?: number;
  max?: number;
  step?: number;
}

interface GroupSpec {
  title: string;
  icon: string;
  dynamicOnly?: boolean;
  fields: FieldSpec[];
}

const GROUPS: GroupSpec[] = [
  {
    title: "Temel",
    icon: "⚙",
    fields: [
      { key: "iters", label: "Iterations", placeholder: "preset", kind: "number", min: 50, step: 100 },
      { key: "resolution", label: "Resolution", placeholder: "640x360", kind: "string" },
      { key: "num_timestamps", label: "Timestamps (export)", placeholder: "preset", kind: "number", min: 2, step: 1 },
      { key: "fps", label: "Frame extraction FPS", placeholder: "10", kind: "number", min: 1, step: 1 },
      { key: "warmup_iters", label: "Warmup iters", placeholder: "500", kind: "number", min: 0, step: 100, help: "Regularizer'lar linear 0→full over this many iter. v3 full default: 500." },
    ],
  },
  {
    title: "Loss ağırlıkları (mode-shared)",
    icon: "λ",
    fields: [
      { key: "lambda_ssim", label: "λ SSIM", placeholder: "0.2", kind: "number", step: 0.05, help: "0=L1 only, 1=SSIM only" },
      { key: "lambda_scale", label: "λ scale reg", placeholder: "0.005", kind: "number", step: 0.001, help: "Asimetrik hinge: scale > 5% × scene_extent olanları cezalandırır." },
      { key: "lambda_depth", label: "λ depth (Metric3D)", placeholder: "0.1", kind: "number", step: 0.05, help: "Scale-invariant L1 between rendered & Metric3D depth. 0 = depth supervision off." },
    ],
  },
  {
    title: "Motion / 4D loss",
    icon: "🎬",
    dynamicOnly: true,
    fields: [
      { key: "lambda_deform_reg", label: "λ deform L2", placeholder: "0.0003", kind: "number", step: 0.0001, help: "Δpos/Δquat/Δscale magnitude regularizer." },
      { key: "lambda_smoothness", label: "λ temporal smoothness", placeholder: "0.002", kind: "number", step: 0.001, help: "D(t) vs D(t+dt)." },
      { key: "lambda_rigidity", label: "λ isometric rigidity", placeholder: "0.002", kind: "number", step: 0.001, help: "Local geometry koruma." },
      { key: "lambda_mask_motion", label: "λ mask-weighted recon", placeholder: "1.0", kind: "number", step: 0.1, help: "Dynamic mask'li bölgelerde reconstruction weight boost." },
      { key: "lambda_track", label: "λ track (CoTracker)", placeholder: "0.1", kind: "number", step: 0.01, help: "CoTracker 3D-anchored track L1." },
      { key: "track_sample_k", label: "Track sample K", placeholder: "256", kind: "number", min: 16, step: 32, help: "Her iter kaç track örneklenir." },
    ],
  },
  {
    title: "Learning rates",
    icon: "↓",
    fields: [
      { key: "lr_means", label: "LR means", placeholder: "0.00016", kind: "number", step: 0.0001 },
    ],
  },
  {
    title: "Learning rates (4D)",
    icon: "↓",
    dynamicOnly: true,
    fields: [
      { key: "lr_deform", label: "LR deformation", placeholder: "0.003", kind: "number", step: 0.0005, help: "v3 default: 3e-3 — MLP motion'u 3× daha hızlı öğrenir." },
      { key: "lr_fourier", label: "LR fourier coeffs", placeholder: "0.005", kind: "number", step: 0.001, help: "Per-Gaussian Fourier katsayıları için ayrı LR." },
    ],
  },
  {
    title: "Density control",
    icon: "●",
    fields: [
      { key: "density_start_iter", label: "Density start iter", placeholder: "preset", kind: "number", min: 0, step: 100 },
      { key: "density_end_iter", label: "Density end iter", placeholder: "preset", kind: "number", min: 0, step: 100, help: "v3 default full'de 22k — final prune'lar için." },
      { key: "density_interval", label: "Density interval", placeholder: "100", kind: "number", min: 1, step: 10 },
      { key: "densify_grad_threshold", label: "Densify grad threshold", placeholder: "0.0002", kind: "number", step: 0.0001 },
      { key: "prune_min_opacity", label: "Prune min opacity", placeholder: "0.005", kind: "number", step: 0.001 },
      { key: "prune_max_scale", label: "Prune max scale (fraction)", placeholder: "0.02", kind: "number", step: 0.005, help: "v3.2: scene_extent'in fraction'u. 0.02×scene_extent units cap." },
      { key: "max_gaussians", label: "Max gauss (N hard cap)", placeholder: "0 (unlimited)", kind: "number", min: 0, step: 10000, help: "v3.7.2: Bu sayıya ulaşınca split+clone durdurulur, sadece prune devam. 0 = sınırsız." },
      { key: "opacity_reset_interval", label: "Opacity reset aralığı", placeholder: "0 (kapalı)", kind: "number", min: 0, step: 500, help: "v3.1 default: 0 (kapalı). >0 koyarsan her N iter reset yapar." },
    ],
  },
  {
    title: "Model — Gaussian (mode-shared)",
    icon: "◇",
    fields: [
      { key: "sh_degree", label: "SH degree", placeholder: "3", kind: "number", min: 0, max: 3, step: 1 },
    ],
  },
  {
    title: "Model — Deformation MLP / Fourier (4D)",
    icon: "◇",
    dynamicOnly: true,
    fields: [
      { key: "hexplane_resolution", label: "HexPlane resolution", placeholder: "96", kind: "number", min: 16, step: 16 },
      { key: "hexplane_feat_dim", label: "HexPlane feature dim", placeholder: "48", kind: "number", min: 8, step: 8 },
      { key: "mlp_width", label: "Deformation MLP width", placeholder: "512", kind: "number", min: 32, step: 32 },
      { key: "mlp_depth", label: "Deformation MLP depth", placeholder: "4", kind: "number", min: 1, max: 8, step: 1 },
      { key: "num_time_freqs", label: "Fourier time freqs", placeholder: "6", kind: "number", min: 0, max: 12, step: 1, help: "0 = kapalı" },
      { key: "deform_pos_mode", label: "Deform pos mode", placeholder: "hybrid", kind: "string", help: "mlp | fourier | hybrid (default: hybrid)" },
      { key: "fourier_K", label: "Fourier trajectory K", placeholder: "8", kind: "number", min: 0, max: 32, step: 1, help: "Per-Gaussian frekans sayısı (0 = kapalı)" },
      { key: "lambda_fourier_reg", label: "λ fourier reg", placeholder: "0.0001", kind: "number", step: 0.0001, help: "High-freq bastırma." },
    ],
  },
  {
    title: "Foundation modeller (4D only)",
    icon: "🜚",
    dynamicOnly: true,
    fields: [
      { key: "metric3d_model", label: "Metric3D model", placeholder: "metric3d_vit_small", kind: "string", help: "small (hızlı) | large (keskin) | giant2 (en iyi)" },
      { key: "cotracker_num_points", label: "CoTracker nokta sayısı", placeholder: "2048", kind: "number", min: 256, step: 256 },
      { key: "cotracker_grid_size", label: "CoTracker grid NxN", placeholder: "30", kind: "number", min: 10, max: 60, step: 5 },
      { key: "sam2_threshold", label: "SAM2 threshold", placeholder: "0.5", kind: "number", min: 0, max: 1, step: 0.05 },
    ],
  },
  {
    title: "Preprocessing",
    icon: "🎬",
    fields: [
      { key: "resize_long_edge", label: "Frame extract long edge (px)", placeholder: "960", kind: "number", min: 480, step: 160, help: "1280-1440 daha çok detail ama COLMAP yavaşlar." },
      { key: "colmap_matching", label: "COLMAP matching", placeholder: "sequential", kind: "string", help: "sequential (hızlı) | exhaustive (yavaş, orbital camera için loop closure)" },
      { key: "init_subsample_mode", label: "Init subsample mode", placeholder: "random", kind: "string", help: "random | confidence (track length / reproj error tabanlı)" },
    ],
  },
  {
    title: "v6.1 Quality (mode-shared)",
    icon: "✨",
    fields: [
      { key: "mip_scale_floor_frac", label: "Mip-Splatting scale floor frac", placeholder: "0", kind: "number", step: 0.0005, min: 0, max: 0.01, help: "3D scale floor = frac × distance_to_nearest_cam. Anti-aliasing approx." },
      { key: "sh_progressive_schedule", label: "SH progressive schedule (0/1)", placeholder: "1", kind: "number", min: 0, max: 1, step: 1, help: "1=ON (default), 0=OFF (sabit max degree). PSNR +0.3-0.5 dB." },
      { key: "init_method", label: "Init method", placeholder: "colmap", kind: "string", help: "colmap | dust3r (sparse-view) | auto (frame<20 ise dust3r)" },
    ],
  },
  {
    title: "v6.1 Quality (4D only)",
    icon: "✨",
    dynamicOnly: true,
    fields: [
      { key: "lambda_accel", label: "λ accel (2nd-order smoothness)", placeholder: "0", kind: "number", step: 0.0001, min: 0, max: 0.01, help: "Slow-motion render titremesini azalt. 1e-4 to 5e-4 önerilen." },
      { key: "cam_grad_clip_norm", label: "Cam refine grad clip", placeholder: "1.0", kind: "number", step: 0.1, min: 0, help: "cam_K + cam_w2c grad norm budget. 0=off." },
      { key: "dynamic_densify_scale", label: "Dynamic densify scale", placeholder: "0.5", kind: "number", step: 0.1, min: 0, max: 2, help: "0.5 = 2× hassas. 1.0 = no-op." },
    ],
  },
  {
    title: "Anti-streak (4D only)",
    icon: "★",
    dynamicOnly: true,
    fields: [
      { key: "lambda_aniso", label: "λ anisotropy", placeholder: "0", kind: "number", step: 0.005, help: "max/min scale ratio threshold üstü gauss'ları cezalandır." },
      { key: "aniso_threshold", label: "Aniso threshold", placeholder: "5", kind: "number", min: 1, step: 0.5, help: "Düşük=daha agresif streak fix." },
      { key: "dpos_total_cap_frac", label: "Δpos total cap fraction", placeholder: "0.2", kind: "number", min: 0.01, max: 0.5, step: 0.01, help: "Per-iter motion cap × scene_extent." },
    ],
  },
];

export function HyperparameterPanel({ value, onChange, mode = "dynamic", preset }: Props) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set(["Temel"]));
  const [search, setSearch] = useState("");

  const presetDefaults = useMemo(
    () => (preset ? PRESET_DEFAULTS[mode]?.[preset] : undefined) ?? {},
    [mode, preset],
  );

  const visibleGroups = useMemo(() => {
    const q = search.trim().toLowerCase();
    return GROUPS
      .filter((g) => !(mode === "static" && g.dynamicOnly))
      .map((g) => {
        if (!q) return g;
        const fields = g.fields.filter(
          (f) =>
            f.label.toLowerCase().includes(q) ||
            String(f.key).toLowerCase().includes(q),
        );
        return fields.length > 0 ? { ...g, fields } : null;
      })
      .filter((g): g is GroupSpec => g !== null);
  }, [mode, search]);

  // When user types a search, auto-expand all matching groups
  const effectiveExpanded = useMemo(() => {
    if (!search.trim()) return expanded;
    const next = new Set(expanded);
    for (const g of visibleGroups) next.add(g.title);
    return next;
  }, [expanded, search, visibleGroups]);

  const toggle = (title: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(title)) next.delete(title);
      else next.add(title);
      return next;
    });
  };

  const overrideCount = Object.values(value).filter(
    (v) => v !== null && v !== undefined && v !== "",
  ).length;

  const handleReset = () => {
    if (overrideCount === 0) return;
    if (window.confirm(`Reset ${overrideCount} override${overrideCount > 1 ? "s" : ""}?`)) {
      onChange({});
    }
  };

  return (
    <div className="hparam-panel">
      <div className="hparam-panel-header">
        <span>
          Hiperparametreler{" "}
          <span className="hparam-mode-tag">
            {mode === "static" ? "📸 Static" : "🎬 Dynamic"}
          </span>
        </span>
        <span className="hparam-summary">
          {overrideCount === 0
            ? `${preset ?? "default"} preset default'ları`
            : `${overrideCount} override`}
        </span>
        <button
          type="button"
          className="hparam-reset-btn"
          onClick={handleReset}
          disabled={overrideCount === 0}
          title="Tüm override'ları temizle"
        >
          Reset
        </button>
      </div>
      <div className="hparam-search-row">
        <input
          type="text"
          className="hparam-search"
          placeholder="🔍 Field ara…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>
      {visibleGroups.length === 0 && (
        <div className="hparam-empty">Eşleşen field yok.</div>
      )}
      {visibleGroups.map((g) => {
        const isOpen = effectiveExpanded.has(g.title);
        const groupOverrides = g.fields.filter(
          (f) => value[f.key] !== null && value[f.key] !== undefined && value[f.key] !== "",
        ).length;
        return (
          <div key={g.title} className={`hparam-group ${isOpen ? "open" : ""}`}>
            <button
              className="hparam-group-header"
              onClick={() => toggle(g.title)}
              type="button"
            >
              <span className="hparam-group-icon">{g.icon}</span>
              <span className="hparam-group-title">{g.title}</span>
              {groupOverrides > 0 && (
                <span className="hparam-badge">{groupOverrides}</span>
              )}
              <span className="hparam-chevron">{isOpen ? "▾" : "▸"}</span>
            </button>
            {isOpen && (
              <div className="hparam-fields">
                {g.fields.map((f) => (
                  <NumberOrTextField
                    key={f.key}
                    spec={f}
                    value={value[f.key] as any}
                    presetDefault={presetDefaults[f.key]}
                    onChange={(val) => onChange({ ...value, [f.key]: val })}
                  />
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function NumberOrTextField({
  spec,
  value,
  presetDefault,
  onChange,
}: {
  spec: FieldSpec;
  value: number | string | null | undefined;
  presetDefault: number | string | null | undefined;
  onChange: (val: number | string | null) => void;
}) {
  const isNum = spec.kind === "number";
  const [raw, setRaw] = useState<string>(
    value === null || value === undefined ? "" : String(value),
  );
  useEffect(() => {
    const ext = value === null || value === undefined ? "" : String(value);
    const parsed = isNum ? parseFloat(raw.replace(",", ".")) : NaN;
    const sameAsValue =
      (raw === "" && (value === null || value === undefined)) ||
      (isNum && Number.isFinite(parsed) && parsed === value) ||
      (!isNum && raw === value);
    if (!sameAsValue) setRaw(ext);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const r = e.target.value;
    setRaw(r);
    if (r === "") { onChange(null); return; }
    if (isNum) {
      const normalized = r.replace(",", ".").trim();
      if (normalized === "-" || normalized === "." || normalized === "-.") return;
      const n = parseFloat(normalized);
      if (Number.isFinite(n)) onChange(n);
    } else {
      onChange(r);
    }
  };

  const presetHint =
    presetDefault !== undefined && presetDefault !== null && presetDefault !== ""
      ? `preset: ${presetDefault}`
      : null;

  return (
    <div className="hparam-field">
      <div className="hparam-field-row">
        <span className="hparam-label">{spec.label}</span>
        {spec.help && <InfoButton label={`${spec.label} help`}>{spec.help}</InfoButton>}
        {presetHint && <span className="hparam-preset-hint">{presetHint}</span>}
      </div>
      <input
        type="text"
        inputMode={isNum ? "decimal" : "text"}
        placeholder={spec.placeholder}
        value={raw}
        onChange={handleChange}
      />
    </div>
  );
}
```

- [ ] **Step 6.3: Append CSS to `App.css`**

```css
.hparam-search-row {
  padding: var(--sp-2) var(--sp-3);
  border-bottom: 1px solid var(--border);
}
.hparam-search {
  width: 100%;
  padding: 5px 10px;
  background: #0a0a0a;
  border: 1px solid var(--border-1);
  color: var(--text);
  border-radius: var(--radius-sm);
  font-size: var(--fs-12);
}
.hparam-search:focus { outline: none; border-color: var(--accent); }

.hparam-reset-btn {
  margin-left: var(--sp-2);
  padding: 2px 8px;
  background: transparent;
  border: 1px solid var(--border-1);
  color: var(--text-muted);
  border-radius: var(--radius-sm);
  font-size: var(--fs-11);
  cursor: pointer;
}
.hparam-reset-btn:hover:enabled { background: var(--bg-3); color: var(--text); }
.hparam-reset-btn:disabled { opacity: 0.4; cursor: not-allowed; }

.hparam-empty {
  padding: var(--sp-3) var(--sp-4);
  font-size: var(--fs-12);
  color: var(--text-muted);
  text-align: center;
}

.hparam-field {
  display: flex;
  flex-direction: column;
  gap: 3px;
}
.hparam-field-row {
  display: flex;
  align-items: center;
  gap: var(--sp-1);
}
.hparam-preset-hint {
  margin-left: auto;
  font-size: var(--fs-11);
  color: var(--text-muted);
  font-family: var(--ff-mono);
}
```

The existing `.hparam-field` rule in `App.css` has a different selector tree (it was `.hparam-field input`, now still works for our new layout because `input` is still a direct descendant). Don't remove the existing rule.

- [ ] **Step 6.4: Build**

```bash
npm run build
```

Expected: clean.

- [ ] **Step 6.5: Commit**

```bash
git add src/components/HyperparameterPanel.tsx src/App.css
git commit -m "feat(frontend): hyperparams panel — search, info, preset placeholders, reset"
```

---

## Task 7: Jobs — pipeline filter chip + mode tag per row

**Files:**
- Modify: `src/components/JobsList.tsx`
- Modify: `src/App.tsx` (pass `pipeline` to `JobsList`)
- Modify: `src/App.css` (filter chip styling)

- [ ] **Step 7.1: Update `JobsList.tsx`**

Replace its contents with:

```tsx
import { useEffect, useMemo, useState } from "react";
import type { Job, JobMode } from "../api";
import { listJobs } from "../api";

interface Props {
  pipeline: JobMode;
  onViewJob: (job: Job) => void;
}

type Filter = "current" | "all" | "static" | "dynamic";

export function JobsList({ pipeline, onViewJob }: Props) {
  const [jobs, setJobs] = useState<Job[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>("current");

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const res = await listJobs();
        if (!cancelled) {
          setJobs(res.jobs);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) setError(String(e));
      }
    };
    tick();
    const id = window.setInterval(tick, 2000);
    return () => { cancelled = true; window.clearInterval(id); };
  }, []);

  const filtered = useMemo(() => {
    if (!jobs) return null;
    const target =
      filter === "current" ? pipeline :
      filter === "all" ? null :
      filter;
    if (target === null) return jobs;
    return jobs.filter((j) => (j.mode ?? "dynamic") === target);
  }, [jobs, filter, pipeline]);

  if (error) {
    return (
      <div className="jobs-list-empty">
        <p className="hint hint-err">Jobs yüklenemedi: {error}</p>
        <p className="hint">Backend çalışıyor mu? (http://127.0.0.1:8000)</p>
      </div>
    );
  }
  if (jobs === null) {
    return <div className="jobs-list-empty"><p className="hint">Yükleniyor…</p></div>;
  }

  return (
    <>
      <div className="jobs-filter-row">
        <FilterChip label={pipeline === "static" ? "📸 Static" : "🎬 Dynamic"} active={filter === "current"} onClick={() => setFilter("current")} />
        <FilterChip label="All" active={filter === "all"} onClick={() => setFilter("all")} />
        <FilterChip label="📸 Static" active={filter === "static"} onClick={() => setFilter("static")} />
        <FilterChip label="🎬 Dynamic" active={filter === "dynamic"} onClick={() => setFilter("dynamic")} />
      </div>
      {filtered && filtered.length === 0 ? (
        <div className="jobs-list-empty">
          <p className="hint">Bu filtrede iş yok.</p>
          <p className="hint">"Submit" sekmesinden bir job başlat.</p>
        </div>
      ) : (
        <div className="jobs-list">
          {filtered!.map((job) => (
            <JobRow
              key={job.id}
              job={job}
              expanded={expanded === job.id}
              onToggleExpand={() => setExpanded(expanded === job.id ? null : job.id)}
              onView={() => onViewJob(job)}
            />
          ))}
        </div>
      )}
    </>
  );
}

function FilterChip({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      className={`jobs-filter-chip ${active ? "active" : ""}`}
      onClick={onClick}
    >
      {label}
    </button>
  );
}

interface RowProps {
  job: Job;
  expanded: boolean;
  onToggleExpand: () => void;
  onView: () => void;
}

function JobRow({ job, expanded, onToggleExpand, onView }: RowProps) {
  const duration = (() => {
    const start = job.started_at ?? job.created_at;
    const end = job.finished_at ?? Date.now() / 1000;
    return end - start;
  })();
  const mode = job.mode ?? "dynamic";
  return (
    <div className={`job-row job-${job.status}`}>
      <div className="job-row-top" onClick={onToggleExpand}>
        <div className="job-row-left">
          <span className={`job-badge job-badge-${job.status}`}>{job.status}</span>
          <span className={`job-mode-chip job-mode-${mode}`}>
            {mode === "static" ? "📸 static" : "🎬 dynamic"}
          </span>
          <span className="job-scene">{job.scene}</span>
          {job.smoke_test && <span className="job-tag">smoke</span>}
        </div>
        <div className="job-row-right">
          <span className="job-duration">{formatDuration(duration)}</span>
          <span className="job-chevron">{expanded ? "▾" : "▸"}</span>
        </div>
      </div>
      {(job.status === "running" || job.status === "queued") && (
        <div className="job-progress-row">
          <div className="progress-bar">
            <div
              className={`progress-fill progress-fill-${job.status}`}
              style={{ width: `${job.overall_progress * 100}%` }}
            />
          </div>
          <span className="progress-pct">%{(job.overall_progress * 100).toFixed(0)}</span>
        </div>
      )}
      {job.status === "running" && (
        <div className="job-phase-row">
          <span className="job-phase-name">{job.phase.name}</span>
          <span className="job-phase-msg">{job.phase.message}</span>
        </div>
      )}
      {expanded && (
        <div className="job-expand">
          <div className="job-meta">
            <span>ID: <code>{job.id}</code></span>
            <span>Oluşturuldu: {new Date(job.created_at * 1000).toLocaleString()}</span>
            {job.finished_at && (
              <span>Bitiş: {new Date(job.finished_at * 1000).toLocaleString()}</span>
            )}
          </div>
          {job.error && (
            <details className="job-error">
              <summary>Hata</summary>
              <pre>{job.error}</pre>
            </details>
          )}
          {job.status === "completed" && (
            <div className="job-actions">
              <button className="btn-primary" onClick={onView}>Viewer'da aç</button>
              <a
                className="btn-secondary"
                href={`http://127.0.0.1:8000/download/${job.id}`}
                target="_blank"
                rel="noreferrer"
              >
                .zip indir
              </a>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function formatDuration(sec: number): string {
  if (sec < 60) return `${sec.toFixed(0)}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ${Math.floor(sec % 60)}s`;
  return `${Math.floor(sec / 3600)}h ${Math.floor((sec % 3600) / 60)}m`;
}
```

- [ ] **Step 7.2: Verify `Job` type includes `mode`**

Read `src/api.ts` to confirm `Job.mode` exists. If it doesn't, the cast `(j.mode ?? "dynamic")` in step 7.1 still works — it falls back to `"dynamic"`. No api.ts edit needed.

```bash
grep -n "interface Job" src/api.ts
```

If `mode` is missing on `Job`, add it (optional `string`):

In `src/api.ts`, find the `interface Job { ... }` declaration and add:

```ts
  mode?: JobMode;
```

If it's already there, skip.

- [ ] **Step 7.3: Pass `pipeline` from `App.tsx`**

In the jobs tab block:

```tsx
{tab === "jobs" && (
  <div className="tab-content">
    <h2 className="tab-title">Tüm Jobs</h2>
    <JobsList pipeline={pipeline} onViewJob={handleViewJob} />
  </div>
)}
```

- [ ] **Step 7.4: Append CSS to `App.css`**

```css
.jobs-filter-row {
  display: flex;
  gap: var(--sp-1);
  padding: 0 0 var(--sp-3);
  flex-wrap: wrap;
}
.jobs-filter-chip {
  padding: 4px 10px;
  background: var(--bg-2);
  border: 1px solid var(--border);
  color: var(--text-muted);
  border-radius: var(--radius-sm);
  font-size: var(--fs-12);
  cursor: pointer;
}
.jobs-filter-chip:hover { color: var(--text); }
.jobs-filter-chip.active {
  background: var(--accent-bg);
  border-color: var(--accent);
  color: var(--text);
}

.job-mode-chip {
  font-size: var(--fs-11);
  padding: 1px 6px;
  border-radius: var(--radius-sm);
  background: var(--bg-3);
  color: var(--text-muted);
  margin-right: var(--sp-1);
}
.job-mode-static { color: var(--accent-static); }
.job-mode-dynamic { color: var(--accent-dynamic); }
```

- [ ] **Step 7.5: Build**

```bash
npm run build
```

Expected: clean.

- [ ] **Step 7.6: Commit**

```bash
git add src/components/JobsList.tsx src/App.tsx src/App.css src/api.ts
git commit -m "feat(frontend): jobs filter chips + mode tag per row"
```

---

## Task 8: Viewer — sidebar, settings popover, HUD

**Files:**
- Create: `src/components/ViewerSidebar.tsx`
- Create: `src/components/ViewerSettings.tsx`
- Create: `src/components/ViewerHUD.tsx`
- Modify: `src/components/SplatViewer.tsx` (emit perf stats)
- Modify: `src/components/SplatViewerSpark.tsx` (emit perf stats)
- Modify: `src/App.tsx` (viewer-tab restructure)
- Modify: `src/App.css` (viewer layout updates)

- [ ] **Step 8.1: Define perf stats type**

In `src/api.ts`, near other types, add:

```ts
export interface PerfStats {
  fps: number;
  gaussCount: number;
  vramMB?: number;  // optional — engines may not expose it
}
```

- [ ] **Step 8.2: Create `ViewerHUD.tsx`**

```tsx
import type { PerfStats } from "../api";

interface Props {
  stats: PerfStats | null;
  visible: boolean;
}

export function ViewerHUD({ stats, visible }: Props) {
  if (!visible || !stats) return null;
  return (
    <div className="viewer-hud" aria-live="polite">
      <span><strong>{stats.fps.toFixed(0)}</strong> FPS</span>
      <span>·</span>
      <span><strong>{(stats.gaussCount / 1000).toFixed(0)}k</strong> gauss</span>
      {stats.vramMB !== undefined && (
        <>
          <span>·</span>
          <span><strong>{stats.vramMB.toFixed(0)}</strong> MB</span>
        </>
      )}
    </div>
  );
}
```

- [ ] **Step 8.3: Create `ViewerSettings.tsx`** (popover)

```tsx
import { useEffect, useRef, useState } from "react";

export type ViewerEngine = "legacy" | "spark";

interface Props {
  engine: ViewerEngine;
  onEngineChange: (e: ViewerEngine) => void;
  hudVisible: boolean;
  onHudToggle: (v: boolean) => void;
  singleFrameMode: boolean;
  onSingleFrameToggle: (v: boolean) => void;
}

export function ViewerSettings(props: Props) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener("mousedown", onClick);
    return () => window.removeEventListener("mousedown", onClick);
  }, [open]);

  return (
    <div className="viewer-settings" ref={ref}>
      <button
        type="button"
        className="viewer-settings-btn"
        aria-label="Viewer settings"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        ⚙
      </button>
      {open && (
        <div className="viewer-settings-popover">
          <div className="viewer-settings-section">
            <div className="viewer-settings-label">Engine</div>
            <label className="viewer-settings-row">
              <input
                type="radio"
                checked={props.engine === "legacy"}
                onChange={() => props.onEngineChange("legacy")}
              />
              legacy (mkkellogg)
            </label>
            <label className="viewer-settings-row">
              <input
                type="radio"
                checked={props.engine === "spark"}
                onChange={() => props.onEngineChange("spark")}
              />
              Spark (4DGS) ✨
            </label>
          </div>
          <div className="viewer-settings-section">
            <div className="viewer-settings-label">Overlays</div>
            <label className="viewer-settings-row">
              <input
                type="checkbox"
                checked={props.hudVisible}
                onChange={(e) => props.onHudToggle(e.target.checked)}
              />
              Performance HUD
            </label>
          </div>
          <div className="viewer-settings-section">
            <div className="viewer-settings-label">Debug</div>
            <label className="viewer-settings-row">
              <input
                type="checkbox"
                checked={props.singleFrameMode}
                onChange={(e) => props.onSingleFrameToggle(e.target.checked)}
              />
              Single-frame mode
            </label>
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 8.4: Create `ViewerSidebar.tsx`**

```tsx
import { useEffect, useState } from "react";
import { Sidebar } from "./ui/Sidebar";
import { StatPill } from "./ui/StatPill";
import { getJobSummary, type SplatInfo } from "../api";

interface Props {
  info: SplatInfo;
}

export function ViewerSidebar({ info }: Props) {
  const [summary, setSummary] = useState<Record<string, any> | null>(null);
  const [loadingSummary, setLoadingSummary] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setSummary(null);
    setLoadingSummary(true);
    getJobSummary(info.scene)
      .then((s) => { if (!cancelled) setSummary(s as Record<string, any>); })
      .catch(() => { /* summary may not exist (still training); ignore */ })
      .finally(() => { if (!cancelled) setLoadingSummary(false); });
    return () => { cancelled = true; };
  }, [info.scene]);

  const wide = typeof window !== "undefined" && window.innerWidth >= 1280;

  const mode = (summary?.mode as string) ?? "—";
  const preset = (summary?.preset as string) ?? "—";
  const psnr = (summary?.final_psnr as number | undefined);
  const trainingDuration = (summary?.training_duration_sec as number | undefined);
  const overrides = (summary?.hyperparam_overrides as Record<string, unknown>) ?? {};
  const overrideEntries = Object.entries(overrides).filter(([, v]) => v !== null && v !== undefined);

  return (
    <Sidebar side="right" defaultOpen={wide} width={300}>
      <div className="vsidebar">
        <div className="vsidebar-title">{info.scene}</div>
        <div className="vsidebar-mode">
          <span className={`job-mode-chip job-mode-${mode === "static" ? "static" : "dynamic"}`}>
            {mode === "static" ? "📸 static" : mode === "dynamic" ? "🎬 dynamic" : mode}
          </span>
          {preset !== "—" && <span className="vsidebar-preset">preset: {preset}</span>}
        </div>

        <div className="vsidebar-stats">
          <StatPill label="Frames" value={info.num_frames} />
          <StatPill label="Size" value={formatBytes(info.total_size_bytes)} />
          {psnr !== undefined && <StatPill label="PSNR" value={`${psnr.toFixed(2)} dB`} tone="accent" />}
          {trainingDuration !== undefined && (
            <StatPill label="Training" value={formatDuration(trainingDuration)} />
          )}
        </div>

        {overrideEntries.length > 0 && (
          <div className="vsidebar-section">
            <div className="vsidebar-section-title">Hyperparam overrides</div>
            <ul className="vsidebar-kv">
              {overrideEntries.slice(0, 8).map(([k, v]) => (
                <li key={k}>
                  <span className="vsidebar-k">{k}</span>
                  <span className="vsidebar-v">{String(v)}</span>
                </li>
              ))}
              {overrideEntries.length > 8 && (
                <li className="vsidebar-more">+{overrideEntries.length - 8} more</li>
              )}
            </ul>
          </div>
        )}

        <div className="vsidebar-section">
          <div className="vsidebar-section-title">Files</div>
          <a
            className="btn-secondary"
            href={`http://127.0.0.1:8000/download/${info.job_id}`}
            target="_blank"
            rel="noreferrer"
          >
            .zip indir ({info.num_frames} ply)
          </a>
        </div>

        {loadingSummary && <div className="hint">Summary yükleniyor…</div>}
        {!loadingSummary && summary === null && (
          <div className="hint">Summary yok (training devam ediyor olabilir).</div>
        )}
      </div>
    </Sidebar>
  );
}

function formatBytes(b: number): string {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  if (b < 1024 * 1024 * 1024) return `${(b / 1024 / 1024).toFixed(1)} MB`;
  return `${(b / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function formatDuration(sec: number): string {
  if (sec < 60) return `${sec.toFixed(0)}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m`;
  return `${Math.floor(sec / 3600)}h ${Math.floor((sec % 3600) / 60)}m`;
}
```

- [ ] **Step 8.5: Add `onPerfTick` prop to `SplatViewer.tsx`**

Read the current `SplatViewer.tsx`. The component already runs a render loop. Add `onPerfTick?: (stats: PerfStats) => void` to its `Props` interface and call it once per frame from inside the render loop, computing FPS via a moving average of frame deltas, gauss count from the loaded scene, vramMB left undefined.

```tsx
import type { PerfStats } from "../api";

interface SplatViewerProps {
  // ... existing props
  onPerfTick?: (stats: PerfStats) => void;
}

// Inside the component, alongside other refs:
const lastFrameTimeRef = useRef<number>(performance.now());
const fpsBufferRef = useRef<number[]>([]);

// In the render-loop callback (where rAF or animation tick runs), append:
const now = performance.now();
const dt = now - lastFrameTimeRef.current;
lastFrameTimeRef.current = now;
const buf = fpsBufferRef.current;
buf.push(1000 / Math.max(dt, 1));
if (buf.length > 30) buf.shift();
const fps = buf.reduce((a, b) => a + b, 0) / buf.length;

if (onPerfTick) {
  // gaussCount source is engine-specific; if the existing scene/viewer
  // exposes a count, read it here. Otherwise use the latest known frame total.
  const gaussCount = /* engine-provided count */ 0;
  onPerfTick({ fps, gaussCount });
}
```

The exact insertion point depends on the existing render loop. Open the file, find the `requestAnimationFrame` (or three.js `setAnimationLoop`) callback, and place the perf-tick computation there. If the engine doesn't expose a gauss count cheaply, leave `gaussCount` at 0 — the HUD shows "0k gauss" until the engine wires it up (TODO marker is acceptable inline).

- [ ] **Step 8.6: Same for `SplatViewerSpark.tsx`** — analogous addition.

- [ ] **Step 8.7: Restructure viewer tab in `App.tsx`**

Replace the `{tab === "viewer" && (...)}` block. Import the new components and refactor:

```tsx
import { ViewerSidebar } from "./components/ViewerSidebar";
import { ViewerSettings, type ViewerEngine } from "./components/ViewerSettings";
import { ViewerHUD } from "./components/ViewerHUD";
import type { PerfStats } from "./api";

// add state at top of App:
const [hudVisible, setHudVisible] = useState(true);
const [perfStats, setPerfStats] = useState<PerfStats | null>(null);
```

(Replace the existing `viewerEngine` typing with `ViewerEngine`.)

The viewer tab JSX:

```tsx
{tab === "viewer" && (
  <div className="viewer-tab">
    <div className="viewer-toolbar">
      <input
        type="text"
        value={jobIdInput}
        placeholder="job_id veya sahne adı"
        onChange={(e) => setJobIdInput(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter") loadInViewer(jobIdInput); }}
        className="viewer-toolbar-input"
      />
      <button className="btn-primary" onClick={() => loadInViewer(jobIdInput)}>Yükle</button>
      <button className="btn-secondary" onClick={loadLatestCompleted}>Son tamamlanmış</button>
      <button className="btn-secondary" onClick={toggleDiskPanel}>
        Diskten {diskPanelOpen ? "▲" : "▼"}
      </button>
      <ViewerSettings
        engine={viewerEngine}
        onEngineChange={setViewerEngine}
        hudVisible={hudVisible}
        onHudToggle={setHudVisible}
        singleFrameMode={singleFrameMode}
        onSingleFrameToggle={setSingleFrameMode}
      />
    </div>

    {diskPanelOpen && (
      <div className="disk-panel">
        {diskLoading && <p className="hint">Disk taranıyor…</p>}
        {!diskLoading && diskScenes && diskScenes.length === 0 && (
          <p className="hint hint-err">Diskte .ply çıktısı yok.</p>
        )}
        {!diskLoading && diskScenes && diskScenes.length > 0 && (
          <ul className="scene-list">
            {diskScenes.map((s) => (
              <li
                key={s.name}
                className="scene-item"
                onClick={() => { setJobIdInput(s.name); loadInViewer(s.name); }}
              >
                <span className="scene-name">{s.name}</span>
                <span className="scene-meta">
                  {s.num_frames} frame · {formatBytes(s.total_size_bytes)} · {formatTime(s.modified_ts)}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    )}

    {viewerState.kind === "idle" && (
      <div className="viewer-empty">
        <p className="hint">Bir job_id yapıştır veya "Diskten" butonundan seç.</p>
      </div>
    )}
    {viewerState.kind === "loading-info" && (
      <div className="viewer-empty"><p className="hint">Yükleniyor…</p></div>
    )}
    {viewerState.kind === "error" && (
      <div className="viewer-empty">
        <p className="hint hint-err">Hata: {viewerState.message}</p>
      </div>
    )}

    {viewerState.kind === "ready" && (
      <div className="viewer-body">
        <div className="viewer-canvas-wrap">
          <div className="viewer-canvas">
            {viewerEngine === "spark" ? (
              <SplatViewerSpark
                key={`spark-${viewerState.info.job_id}`}
                jobId={viewerState.info.job_id}
                numFrames={viewerState.info.num_frames}
                currentFrame={currentFrame}
                onLoadProgress={(loaded, total) => setSceneProgress({ loaded, total, done: false })}
                onReady={() => setSceneProgress((prev) => prev ? { ...prev, done: true } : prev)}
                onError={(msg) => setViewerState({ kind: "error", message: msg })}
                onPerfTick={setPerfStats}
              />
            ) : (
              <SplatViewer
                key={`${viewerState.info.job_id}-${singleFrameMode ? "single" : "multi"}`}
                jobId={viewerState.info.job_id}
                numFrames={viewerState.info.num_frames}
                currentFrame={currentFrame}
                singleFrameMode={singleFrameMode}
                onLoadProgress={(loaded, total) => setSceneProgress({ loaded, total, done: false })}
                onReady={() => setSceneProgress((prev) => prev ? { ...prev, done: true } : prev)}
                onError={(msg) => setViewerState({ kind: "error", message: msg })}
                onPerfTick={setPerfStats}
              />
            )}
            <ViewerHUD stats={perfStats} visible={hudVisible} />
            {sceneProgress && !sceneProgress.done && (
              <div className="viewer-progress-overlay">
                Scene'ler yükleniyor: {sceneProgress.loaded}/{sceneProgress.total}
              </div>
            )}
          </div>
          <div className="viewer-timeline">
            <TimelineSlider
              numFrames={viewerState.info.num_frames}
              currentFrame={currentFrame}
              onFrameChange={setCurrentFrame}
              baseFps={10}
            />
          </div>
        </div>
        <ViewerSidebar info={viewerState.info} />
      </div>
    )}
  </div>
)}
```

Remove: the old `<div className="viewer-controls">` (replaced by `viewer-toolbar`), the `<div className="viewer-statusbar">` (gone — its info migrated to the sidebar), and the inline single-frame-mode disabled checkbox (now in the settings popover).

- [ ] **Step 8.8: CSS for new viewer chrome — append to `App.css`**

```css
/* New viewer toolbar — replaces .viewer-controls */
.viewer-toolbar {
  display: flex;
  gap: var(--sp-2);
  padding: var(--sp-2) var(--sp-3);
  background: var(--bg-1);
  border-bottom: 1px solid var(--border);
  align-items: center;
}
.viewer-toolbar-input {
  flex: 1;
  padding: 5px 10px;
  background: #0a0a0a;
  border: 1px solid var(--border-1);
  color: var(--text);
  border-radius: var(--radius-sm);
  font-family: var(--ff-mono);
  font-size: var(--fs-12);
}
.viewer-toolbar-input:focus { outline: none; border-color: var(--accent); }

.viewer-body {
  display: flex;
  flex: 1;
  min-height: 0;
}
.viewer-canvas-wrap {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}
.viewer-canvas {
  flex: 1;
  position: relative;
  background: #1a1a1a;
}

.viewer-progress-overlay {
  position: absolute;
  top: var(--sp-3);
  left: var(--sp-3);
  background: rgba(0,0,0,0.6);
  color: var(--text);
  padding: var(--sp-2) var(--sp-3);
  border-radius: var(--radius-sm);
  font-size: var(--fs-12);
}

/* HUD overlay */
.viewer-hud {
  position: absolute;
  bottom: var(--sp-3);
  left: var(--sp-3);
  background: rgba(0,0,0,0.55);
  color: var(--text);
  padding: 4px 10px;
  border-radius: var(--radius-sm);
  font-family: var(--ff-mono);
  font-size: var(--fs-11);
  display: flex;
  gap: 6px;
  pointer-events: none;
}

/* Settings popover */
.viewer-settings { position: relative; }
.viewer-settings-btn {
  width: 30px;
  height: 28px;
  background: var(--bg-3);
  border: 1px solid var(--border-1);
  border-radius: var(--radius-sm);
  color: var(--text);
  cursor: pointer;
  font-size: var(--fs-14);
}
.viewer-settings-btn:hover { background: var(--bg-2); border-color: var(--accent); }
.viewer-settings-popover {
  position: absolute;
  top: calc(100% + 4px);
  right: 0;
  width: 240px;
  background: var(--bg-1);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: var(--sp-2);
  z-index: 10;
}
.viewer-settings-section { padding: var(--sp-2); }
.viewer-settings-section + .viewer-settings-section { border-top: 1px solid var(--border); }
.viewer-settings-label {
  font-size: var(--fs-11);
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: var(--sp-1);
}
.viewer-settings-row {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: var(--fs-12);
  padding: 3px 0;
  cursor: pointer;
}

/* Sidebar contents */
.vsidebar { display: flex; flex-direction: column; gap: var(--sp-3); }
.vsidebar-title {
  font-family: var(--ff-mono);
  font-size: var(--fs-14);
  font-weight: var(--fw-semibold);
  word-break: break-all;
}
.vsidebar-mode {
  display: flex;
  align-items: center;
  gap: var(--sp-2);
  font-size: var(--fs-12);
  color: var(--text-muted);
}
.vsidebar-preset {
  font-family: var(--ff-mono);
  font-size: var(--fs-11);
}
.vsidebar-stats {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--sp-2);
}
.vsidebar-section {
  padding-top: var(--sp-3);
  border-top: 1px solid var(--border);
}
.vsidebar-section-title {
  font-size: var(--fs-11);
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: var(--sp-2);
}
.vsidebar-kv { list-style: none; margin: 0; padding: 0; font-size: var(--fs-11); font-family: var(--ff-mono); }
.vsidebar-kv li { display: flex; justify-content: space-between; padding: 2px 0; }
.vsidebar-k { color: var(--text-muted); }
.vsidebar-v { color: var(--text); }
.vsidebar-more { color: var(--text-muted); font-style: italic; }
```

Remove from `App.css`: the legacy `.viewer-controls`, `.viewer-statusbar`, `.viewer-engine-btn` rules (no longer referenced). Keep `.viewer-tab`, `.viewer-empty`, `.viewer-timeline`, `.disk-panel`, `.scene-list`, `.scene-item` etc.

- [ ] **Step 8.9: Build**

```bash
npm run build
```

Expected: clean.

- [ ] **Step 8.10: Commit**

```bash
git add src/components/ViewerSidebar.tsx src/components/ViewerSettings.tsx \
        src/components/ViewerHUD.tsx src/components/SplatViewer.tsx \
        src/components/SplatViewerSpark.tsx src/api.ts \
        src/App.tsx src/App.css
git commit -m "feat(frontend): viewer redesign — toolbar, settings popover, HUD, sidebar"
```

---

## Task 9: Analiz — health, pipeline timeline, parsed events

**Files:**
- Create: `src/analytics/health.ts`
- Create: `src/analytics/events.ts`
- Create: `src/components/HealthBanner.tsx`
- Create: `src/components/PipelineTimeline.tsx`
- Create: `src/components/EventsLog.tsx`
- Modify: `src/components/TrainingAnalytics.tsx`
- Modify: `src/App.tsx` (pass `pipeline` to `TrainingAnalytics`; scene picker becomes `<select>`)
- Modify: `src/App.css` (analytics styles)

- [ ] **Step 9.1: `src/analytics/health.ts`**

```ts
import type { TrainMetric } from "../api";
import { PRESET_BASELINES } from "../presets";
import type { JobMode } from "../api";

export type Verdict = "converging" | "stalled" | "diverging" | "insufficient";

export interface HealthReport {
  verdict: Verdict;
  windowSize: number;
  lossSlope: number;            // per-iter slope of recent window
  meanLoss: number;
  hasNaN: boolean;
  psnrCurrent?: number;
  psnrBaseline?: number;
  psnrDelta?: number;
}

/**
 * Linear least-squares slope of y vs x.
 */
function slope(x: number[], y: number[]): number {
  const n = x.length;
  if (n < 2) return 0;
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) {
    sx += x[i]; sy += y[i]; sxx += x[i] * x[i]; sxy += x[i] * y[i];
  }
  const denom = n * sxx - sx * sx;
  if (denom === 0) return 0;
  return (n * sxy - sx * sy) / denom;
}

export function computeHealth(
  metrics: TrainMetric[],
  mode: JobMode,
  preset?: string,
): HealthReport {
  if (metrics.length < 30) {
    return {
      verdict: "insufficient",
      windowSize: metrics.length,
      lossSlope: 0,
      meanLoss: 0,
      hasNaN: false,
    };
  }

  const windowSize = Math.max(30, Math.floor(metrics.length * 0.2));
  const recent = metrics.slice(-windowSize);
  const xs = recent.map((m) => m.iter);
  const ys = recent.map((m) => m.loss);

  const hasNaN = recent.some(
    (m) => !Number.isFinite(m.loss) || !Number.isFinite(m.psnr),
  );

  const lossSlope = slope(xs, ys);
  const meanLoss = ys.reduce((a, b) => a + b, 0) / ys.length;
  // Relative epsilon: ~0.01% of mean loss per iter is "no movement"
  const eps = Math.max(1e-7, Math.abs(meanLoss) * 1e-4);

  let verdict: Verdict;
  if (hasNaN || lossSlope > eps) verdict = "diverging";
  else if (lossSlope < -eps) verdict = "converging";
  else verdict = "stalled";

  const last = metrics[metrics.length - 1];
  const psnrCurrent = Number.isFinite(last.psnr) ? last.psnr : undefined;
  const baseline = preset ? PRESET_BASELINES[mode]?.[preset]?.psnr : undefined;
  const psnrDelta = psnrCurrent !== undefined && baseline !== undefined
    ? psnrCurrent - baseline
    : undefined;

  return {
    verdict,
    windowSize,
    lossSlope,
    meanLoss,
    hasNaN,
    psnrCurrent,
    psnrBaseline: baseline,
    psnrDelta,
  };
}
```

- [ ] **Step 9.2: `src/analytics/events.ts`**

```ts
export type EventLevel = "info" | "warn" | "err" | "raw";

export interface ParsedEvent {
  ts?: string;          // raw timestamp text from log line, when present
  phase?: string;       // foundation / colmap / init / train / eval / dens / etc.
  level: EventLevel;
  message: string;
  iter?: number;        // parsed from "iter N" if present
  raw: string;          // original line
}

/**
 * Best-effort parser for backend events.log lines.
 * Format examples (loose; tolerated variations):
 *   "12:04:11 INFO train iter 5000 PSNR 24.1 / it/s 8.2"
 *   "[colmap] sequential matching done in 18m"
 *   "WARN density: split count 5234 (+12% N)"
 */
const TS_RE = /^(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)/;
const LEVEL_RE = /\b(INFO|WARN|WARNING|ERR|ERROR|DEBUG)\b/i;
const PHASE_RE = /\b(foundation|colmap|init|train|training|eval|dens|density)\b/i;
const ITER_RE = /\biter[:\s]+(\d+)/i;

const LEVEL_MAP: Record<string, EventLevel> = {
  INFO: "info",
  DEBUG: "info",
  WARN: "warn",
  WARNING: "warn",
  ERR: "err",
  ERROR: "err",
};

export function parseEvent(line: string): ParsedEvent {
  const tsMatch = TS_RE.exec(line);
  const ts = tsMatch ? tsMatch[1] : undefined;

  const lvlMatch = LEVEL_RE.exec(line);
  const level: EventLevel = lvlMatch ? LEVEL_MAP[lvlMatch[1].toUpperCase()] : "info";

  const phaseMatch = PHASE_RE.exec(line);
  const phase = phaseMatch ? phaseMatch[1].toLowerCase() : undefined;

  const iterMatch = ITER_RE.exec(line);
  const iter = iterMatch ? parseInt(iterMatch[1], 10) : undefined;

  // Strip ts + level + phase from the message body for cleaner display
  let message = line;
  if (tsMatch) message = message.slice(tsMatch[0].length).trim();
  if (lvlMatch) message = message.replace(lvlMatch[0], "").trim();

  return { ts, phase, level, message, iter, raw: line };
}

export function parseEvents(lines: string[]): ParsedEvent[] {
  return lines.map(parseEvent);
}

/**
 * Phase wall-time segments, from events that look like "phase X started/done".
 * We accept any line whose text contains "phase: <name>" or "[phase] start/done".
 */
export interface PhaseSegment {
  phase: string;
  startTs?: string;
  endTs?: string;
  durationSec?: number;
}

export function deriveTimeline(lines: string[]): PhaseSegment[] {
  // Pragmatic: count lines per detected phase as a rough proxy. If backend
  // emits explicit "phase foo started"/"phase foo done" lines, prefer those;
  // otherwise the segments are not time-accurate.
  const segs: PhaseSegment[] = [];
  let currentPhase: string | undefined;
  let currentStart: string | undefined;
  for (const line of lines) {
    const m = /\bphase[:\s]+([a-z_]+)\b/i.exec(line);
    if (m) {
      const name = m[1].toLowerCase();
      if (currentPhase && currentPhase !== name) {
        segs.push({ phase: currentPhase, startTs: currentStart });
      }
      currentPhase = name;
      const tsm = TS_RE.exec(line);
      currentStart = tsm ? tsm[1] : undefined;
    }
  }
  if (currentPhase) segs.push({ phase: currentPhase, startTs: currentStart });
  return segs;
}
```

- [ ] **Step 9.3: `src/components/HealthBanner.tsx`**

```tsx
import { Banner } from "./ui/Banner";
import type { HealthReport } from "../analytics/health";

interface Props {
  report: HealthReport;
}

const VERDICT_TONE: Record<HealthReport["verdict"], "ok" | "warn" | "err" | "info"> = {
  converging: "ok",
  stalled: "warn",
  diverging: "err",
  insufficient: "info",
};

const VERDICT_TITLE: Record<HealthReport["verdict"], string> = {
  converging: "Converging",
  stalled: "Stalled",
  diverging: "Diverging",
  insufficient: "Insufficient data",
};

export function HealthBanner({ report }: Props) {
  return (
    <Banner tone={VERDICT_TONE[report.verdict]} title={VERDICT_TITLE[report.verdict]}>
      <div>
        Loss slope <code>{report.lossSlope.toExponential(2)}</code> per iter
        {" · "}window {report.windowSize}
        {report.hasNaN && <span> · contains NaN</span>}
      </div>
      {report.psnrCurrent !== undefined && (
        <div>
          PSNR {report.psnrCurrent.toFixed(2)} dB
          {report.psnrBaseline !== undefined && report.psnrDelta !== undefined && (
            <> (preset baseline ≈ {report.psnrBaseline} —{" "}
              <strong>{report.psnrDelta >= 0 ? "+" : ""}{report.psnrDelta.toFixed(2)}</strong>)
            </>
          )}
        </div>
      )}
    </Banner>
  );
}
```

- [ ] **Step 9.4: `src/components/PipelineTimeline.tsx`**

```tsx
import type { PhaseSegment } from "../analytics/events";
import { Card } from "./ui/Card";

interface Props {
  segments: PhaseSegment[];
}

const PHASE_COLOR: Record<string, string> = {
  foundation: "#a8dadc",
  colmap: "#ffe66d",
  init: "#dda0dd",
  train: "#4ecdc4",
  training: "#4ecdc4",
  eval: "#95e1d3",
};

export function PipelineTimeline({ segments }: Props) {
  if (segments.length === 0) {
    return (
      <Card title="Pipeline timeline">
        <div className="hint">Timeline unavailable for this run (no phase events).</div>
      </Card>
    );
  }
  // Equal-width segments for now; backend doesn't always emit phase end timestamps.
  const w = 100 / segments.length;
  return (
    <Card title="Pipeline timeline">
      <div className="pl-timeline">
        {segments.map((s, i) => (
          <div
            key={i}
            className="pl-timeline-seg"
            style={{
              width: `${w}%`,
              background: PHASE_COLOR[s.phase] ?? "#888",
            }}
            title={`${s.phase}${s.startTs ? ` · ${s.startTs}` : ""}`}
          >
            <span className="pl-timeline-label">{s.phase}</span>
          </div>
        ))}
      </div>
    </Card>
  );
}
```

- [ ] **Step 9.5: `src/components/EventsLog.tsx`**

```tsx
import { useMemo, useState } from "react";
import type { ParsedEvent, EventLevel } from "../analytics/events";

interface Props {
  events: ParsedEvent[];
  onSelectIter?: (iter: number) => void;
}

const LEVEL_COLOR: Record<EventLevel, string> = {
  info: "var(--text-muted)",
  warn: "var(--warn)",
  err: "var(--err)",
  raw: "var(--text-muted)",
};

export function EventsLog({ events, onSelectIter }: Props) {
  const [phaseFilter, setPhaseFilter] = useState<string>("all");
  const [levelFilter, setLevelFilter] = useState<EventLevel | "all">("all");

  const phases = useMemo(() => {
    const set = new Set<string>();
    for (const e of events) if (e.phase) set.add(e.phase);
    return ["all", ...Array.from(set).sort()];
  }, [events]);

  const filtered = useMemo(() => {
    return events.filter((e) =>
      (phaseFilter === "all" || e.phase === phaseFilter) &&
      (levelFilter === "all" || e.level === levelFilter)
    );
  }, [events, phaseFilter, levelFilter]);

  return (
    <div className="events-log">
      <div className="events-log-filters">
        <label>
          phase:
          <select value={phaseFilter} onChange={(e) => setPhaseFilter(e.target.value)}>
            {phases.map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
        </label>
        <label>
          level:
          <select value={levelFilter} onChange={(e) => setLevelFilter(e.target.value as EventLevel | "all")}>
            <option value="all">all</option>
            <option value="info">info</option>
            <option value="warn">warn</option>
            <option value="err">err</option>
          </select>
        </label>
        <span className="events-log-count">{filtered.length} / {events.length}</span>
      </div>
      <ul className="events-log-list">
        {filtered.slice(-200).map((e, i) => (
          <li
            key={i}
            className={`events-log-row events-log-${e.level}`}
            style={{ color: LEVEL_COLOR[e.level] }}
            onClick={() => e.iter !== undefined && onSelectIter?.(e.iter)}
          >
            {e.ts && <span className="events-log-ts">{e.ts}</span>}
            {e.phase && <span className="events-log-phase">{e.phase}</span>}
            <span className="events-log-msg">{e.message}</span>
          </li>
        ))}
        {filtered.length === 0 && (
          <li className="events-log-empty">No events match filter.</li>
        )}
      </ul>
    </div>
  );
}
```

- [ ] **Step 9.6: Refactor `TrainingAnalytics.tsx`**

The existing file has the chart code, stat cards, events pre, summary pre. Refactor to compose: `<HealthBanner>`, `<PipelineTimeline>`, the existing chart grid (kept as-is, wrapped in `<Card>`), `<EventsLog>`, summary card. Add a `cursorIter` state used by the chart to draw a vertical line at the selected iter.

Open `src/components/TrainingAnalytics.tsx`. Replace the body (the `<LineChart>` and `<StatCard>` definitions stay). The new component body:

Add at the imports:

```tsx
import type { JobMode } from "../api";
import { computeHealth } from "../analytics/health";
import { parseEvents, deriveTimeline } from "../analytics/events";
import { HealthBanner } from "./HealthBanner";
import { PipelineTimeline } from "./PipelineTimeline";
import { EventsLog } from "./EventsLog";
import { Card } from "./ui/Card";
import { StatPill } from "./ui/StatPill";
```

Add to `Props`:

```tsx
interface Props {
  scene: string;
  pipeline: JobMode;
  preset?: string;
  autoRefresh?: boolean;
}
```

Inside the component, after the existing `metrics`/`events` state and `load` effect, add:

```tsx
const [cursorIter, setCursorIter] = useState<number | null>(null);

const health = useMemo(
  () => computeHealth(metrics, pipeline, preset),
  [metrics, pipeline, preset],
);
const parsedEvents = useMemo(() => parseEvents(events), [events]);
const timeline = useMemo(() => deriveTimeline(events), [events]);
```

Replace the JSX body (everything from `return (` through the final `</div>`) with:

```tsx
return (
  <div style={{ padding: "10px 20px", color: "#eee" }}>
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
      <h3 style={{ margin: 0 }}>Training Analytics</h3>
      <span style={{ color: "#888", fontSize: 12 }}>{scene}</span>
      {loading && <span style={{ color: "#ffe66d", fontSize: 11 }}>refreshing…</span>}
      {autoRefresh && !loading && (
        <span style={{ color: "#444", fontSize: 10 }}>auto-refresh 3s</span>
      )}
      <button
        onClick={load}
        style={{ marginLeft: "auto", padding: "4px 10px", fontSize: 11 }}
        className="btn-secondary"
      >
        Refresh
      </button>
    </div>

    {err && (
      <div style={{ color: "#ff6b6b", marginBottom: 10, fontSize: 12 }}>Hata: {err}</div>
    )}

    {metrics.length > 0 && <HealthBanner report={health} />}

    <PipelineTimeline segments={timeline} />

    {last && (
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 16 }}>
        <StatPill label="Progress" value={`${last.iter.toLocaleString()} / ${last.n_iters.toLocaleString()}`} hint={`${progress.toFixed(1)}%`} />
        <StatPill label="Loss" value={last.loss.toFixed(4)} />
        <StatPill label="PSNR" value={`${last.psnr.toFixed(2)} dB`} tone="accent" />
        <StatPill label="N points" value={last.n_points.toLocaleString()} />
        <StatPill label="Δpos mean" value={last.dpos_mean.toFixed(4)} hint={`max ${last.dpos_max.toFixed(3)}`} />
        <StatPill label="Speed" value={`${last.it_per_sec.toFixed(1)} it/s`} hint={`${(last.t / 60).toFixed(1)} dk geçti`} />
        <StatPill label="Warmup" value={`${(last.warmup * 100).toFixed(0)}%`} />
      </div>
    )}

    <Card title="Charts">
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(500px, 1fr))",
          gap: 14,
        }}
      >
        <div>
          <div style={{ fontSize: 13, color: "#ccc", marginBottom: 4 }}>Total Loss</div>
          <LineChart data={metrics} yValues={metrics.map((m) => m.loss)} color="#ff6b6b" yLog cursorIter={cursorIter} />
        </div>
        <div>
          <div style={{ fontSize: 13, color: "#ccc", marginBottom: 4 }}>PSNR</div>
          <LineChart data={metrics} yValues={metrics.map((m) => m.psnr)} color="#4ecdc4" cursorIter={cursorIter} />
        </div>
        <div>
          <div style={{ fontSize: 13, color: "#ccc", marginBottom: 4 }}>Gaussian count (N)</div>
          <LineChart data={metrics} yValues={metrics.map((m) => m.n_points)} color="#ffe66d" cursorIter={cursorIter} />
        </div>
        <div>
          <div style={{ fontSize: 13, color: "#ccc", marginBottom: 4 }}>Δpos mean (motion magnitude)</div>
          <LineChart data={metrics} yValues={metrics.map((m) => m.dpos_mean)} color="#95e1d3" cursorIter={cursorIter} />
        </div>
        {lossComps && (
          <div style={{ gridColumn: "1 / -1" }}>
            <div style={{ fontSize: 13, color: "#ccc", marginBottom: 4 }}>Loss components (log-scale)</div>
            <LineChart data={metrics} multiSeries={lossComps} yLog width={1000} height={220} cursorIter={cursorIter} />
          </div>
        )}
      </div>
    </Card>

    <Card title="Events">
      <EventsLog events={parsedEvents} onSelectIter={setCursorIter} />
    </Card>

    {summary && (
      <Card title="Run summary">
        <pre
          style={{
            margin: 0,
            background: "#1a1a1a",
            border: "1px solid #2a2a2a",
            padding: 10,
            fontSize: 11,
            fontFamily: "monospace",
            color: "#bbb",
            maxHeight: 300,
            overflow: "auto",
            borderRadius: 4,
          }}
        >
          {JSON.stringify(summary, null, 2)}
        </pre>
      </Card>
    )}
  </div>
);
```

Update the `LineChart` signature to accept `cursorIter?: number | null` and draw a vertical line at that iter. In the existing `LineChart` function, after the existing `<path>` rendering, add:

```tsx
{cursorIter !== null && cursorIter !== undefined && cursorIter >= xMin && cursorIter <= xMax && (
  <line
    x1={xNorm(cursorIter)}
    y1={padT}
    x2={xNorm(cursorIter)}
    y2={height - padB}
    stroke="#ffffff"
    strokeOpacity="0.5"
    strokeDasharray="2 2"
  />
)}
```

Add to its Props interface:

```tsx
cursorIter?: number | null;
```

- [ ] **Step 9.7: Update `App.tsx`** — pass `pipeline` (and optionally `preset`) to `TrainingAnalytics`. The preset isn't tracked at app level today; pass `undefined` for now (the health verdict still works minus the PSNR baseline delta). Also, keep the existing `analyticsScene` text-input flow — promoting it to a `<select>` is out of scope for this pass; the input remains.

```tsx
<TrainingAnalytics scene={analyticsScene} pipeline={pipeline} autoRefresh={true} />
```

- [ ] **Step 9.8: Append CSS to `App.css`**

```css
/* Pipeline timeline */
.pl-timeline {
  display: flex;
  height: 28px;
  border-radius: var(--radius-sm);
  overflow: hidden;
}
.pl-timeline-seg {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #111;
  font-size: var(--fs-11);
  font-weight: var(--fw-semibold);
}
.pl-timeline-seg + .pl-timeline-seg { border-left: 1px solid rgba(0,0,0,0.3); }
.pl-timeline-label { text-transform: uppercase; letter-spacing: 0.5px; }

/* Events log */
.events-log {
  display: flex;
  flex-direction: column;
  gap: var(--sp-2);
}
.events-log-filters {
  display: flex;
  gap: var(--sp-3);
  align-items: center;
  font-size: var(--fs-12);
  color: var(--text-muted);
}
.events-log-filters select {
  margin-left: 4px;
  background: #0a0a0a;
  border: 1px solid var(--border-1);
  color: var(--text);
  font-size: var(--fs-12);
  padding: 2px 4px;
}
.events-log-count { margin-left: auto; font-family: var(--ff-mono); font-size: var(--fs-11); }
.events-log-list {
  list-style: none;
  margin: 0;
  padding: 0;
  background: #1a1a1a;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  max-height: 280px;
  overflow-y: auto;
  font-family: var(--ff-mono);
  font-size: var(--fs-11);
}
.events-log-row {
  display: grid;
  grid-template-columns: 70px 70px 1fr;
  gap: var(--sp-2);
  padding: 3px var(--sp-3);
  cursor: pointer;
}
.events-log-row:hover { background: var(--bg-2); }
.events-log-ts { color: var(--text-muted); }
.events-log-phase { color: var(--accent); text-transform: uppercase; }
.events-log-msg { word-break: break-word; }
.events-log-empty {
  padding: var(--sp-3) var(--sp-4);
  color: var(--text-muted);
  text-align: center;
  font-style: italic;
}
```

- [ ] **Step 9.9: Build**

```bash
npm run build
```

Expected: clean.

- [ ] **Step 9.10: Commit**

```bash
git add src/analytics src/components/HealthBanner.tsx src/components/PipelineTimeline.tsx \
        src/components/EventsLog.tsx src/components/TrainingAnalytics.tsx \
        src/App.tsx src/App.css
git commit -m "feat(frontend): analytics — health banner, pipeline timeline, parsed events"
```

---

## Task 10: Eval — visual pass

**Files:**
- Modify: `src/components/NvsEvalPanel.tsx`

- [ ] **Step 10.1: Wrap content in `Card` and replace `<Metric>` with `StatPill`**

Replace the file body with:

```tsx
import { useEffect, useState } from "react";
import { getJobEval, orbitVideoUrl, type NvsEvalReport } from "../api";
import { Card } from "./ui/Card";
import { StatPill } from "./ui/StatPill";

interface Props {
  scene: string;
}

export function NvsEvalPanel({ scene }: Props) {
  const [report, setReport] = useState<NvsEvalReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const loadReport = async () => {
    if (!scene.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const r = await getJobEval(scene.trim());
      setReport(r);
    } catch (e) {
      setError(String(e));
      setReport(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (scene.trim()) loadReport();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scene]);

  if (!scene.trim()) {
    return <div style={{ padding: 20, color: "#888" }}>Sahne adı gir, NVS eval raporu yüklensin.</div>;
  }

  return (
    <div style={{ padding: 20 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
        <h2 style={{ margin: 0 }}>NVS Evaluation — {scene}</h2>
        <button className="btn-secondary" onClick={loadReport} disabled={loading}>
          {loading ? "Yükleniyor..." : "Yenile"}
        </button>
      </div>

      {error && <div className="hint hint-err" style={{ marginBottom: 12 }}>Hata: {error}</div>}

      {report && !report.available && (
        <Card title="NVS evaluation yok">
          <p className="hint">
            Bu sahne için çalıştırılmamış. {report.message && <span>({report.message})</span>}
          </p>
          <p className="hint">
            Submit ederken "NVS Evaluation" toggle'ını aç — training sonrası otomatik çalışır.
          </p>
        </Card>
      )}

      {report && report.available && (
        <>
          <Card title={report.held_out_cam ? `Held-out cam: ${report.held_out_cam}` : "Temporal hold-out (single-view)"}>
            {(() => {
              const m = report.held_out_metrics ?? report.temporal_holdout;
              if (!m) return <p className="hint">Metric yok.</p>;
              return (
                <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                  <StatPill label="PSNR" value={`${m.psnr.toFixed(2)} dB`} tone="accent" />
                  <StatPill label="SSIM" value={m.ssim.toFixed(4)} />
                  <StatPill label="LPIPS ↓" value={m.lpips.toFixed(4)} />
                  <StatPill label="Frames" value={String(m.n_frames)} />
                </div>
              );
            })()}
          </Card>

          {report.orbit?.success && report.orbit_url && (
            <Card title={`Smooth Orbit Camera (${report.orbit.n_frames} frame)`}>
              <video
                src={orbitVideoUrl(scene)}
                controls
                style={{ width: "100%", maxWidth: 800, borderRadius: 4 }}
              />
              <p className="hint" style={{ marginTop: 8 }}>
                Catmull-Rom spline ile train cam'lardan smooth path. Sahnenin "demo" videosu.
              </p>
            </Card>
          )}
          {report.orbit && !report.orbit.success && (
            <div className="hint hint-err">Orbit render başarısız: {report.orbit.error}</div>
          )}
        </>
      )}
    </div>
  );
}
```

- [ ] **Step 10.2: Build**

```bash
npm run build
```

Expected: clean.

- [ ] **Step 10.3: Commit**

```bash
git add src/components/NvsEvalPanel.tsx
git commit -m "feat(frontend): eval panel visual pass (Cards + StatPills)"
```

---

## Task 11: Final verification + cleanup

**Files:**
- Modify: `src/App.css` (drop now-dead rules, audit one more time)

- [ ] **Step 11.1: Audit `App.css` for dead rules** — search for any of these that are no longer referenced anywhere in `src/components/**/*.tsx` or `src/App.tsx`:

```
.submit-mode-wrapper, .mode-picker, .mode-btn, .mode-btn-icon, .mode-btn-content,
.mode-btn-name, .mode-btn-sub, .submit-panel-mode-banner, .submit-panel-mode-static,
.submit-panel-mode-dynamic, .viewer-controls, .viewer-statusbar, .viewer-engine-btn
```

For each that has no remaining reference (verify with grep), delete the rule from `App.css`. Use Grep tool, e.g.:

```
Grep pattern="submit-panel-mode-banner" path="src"
```

- [ ] **Step 11.2: Final production build**

```bash
npm run build
```

Expected: clean tsc, vite build succeeds, no warnings beyond what existed before.

- [ ] **Step 11.3: Smoke run** (manual)

```bash
npm run dev
```

Open the dev server URL. Manual checklist:

- [ ] Top bar shows pipeline switch; clicking it tints the entire UI accent (blue/orange).
- [ ] Submit tab shows the right form per pipeline; preset chips show ⓘ; clicking expands Details.
- [ ] Hyperparams advanced — search filters fields; preset hint shows next to inputs; reset works.
- [ ] Jobs tab — filter chips work; mode chip per row.
- [ ] Viewer tab — toolbar replaces old controls; gear popover toggles engine + HUD; sidebar shows scene info; HUD overlays canvas.
- [ ] Analiz tab — health banner appears once enough metrics; pipeline timeline shows or "unavailable" placeholder; events log filterable; click row → chart cursor moves.
- [ ] Eval tab — Cards wrap metrics + video.

Document any obvious regressions you can't fix inline as TODOs, but the build is the gate.

- [ ] **Step 11.4: Final commit if any cleanup was applied**

```bash
git add src/App.css
git commit -m "chore(frontend): drop dead CSS rules after redesign"
```

If nothing changed in 11.1, skip this step.

---

## Done

The branch `feat/frontend-redesign` now contains:

- Design tokens + primitives (`tokens.css`, `primitives.css`, `components/ui/`).
- Top-level pipeline switch with localStorage migration.
- Submit redesigned with `Card` blocks, shared `PresetChip` with inline Details.
- Hyperparams panel with search, info, preset placeholders, reset.
- Jobs filter chips + mode tag per row.
- Viewer toolbar, settings popover, HUD, scene-info sidebar.
- Analiz health banner, pipeline timeline, parsed events log with click-to-jump.
- Eval visual pass.

No new runtime dependencies. Backend unchanged. All tasks build cleanly.
