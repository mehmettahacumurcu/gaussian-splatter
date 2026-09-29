import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { Vector3 } from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { getJobStatus, type Job } from "../api";
import { AssetPicker } from "./AssetPicker";
import { ComposeViewport } from "./ComposeViewport";
import {
  assetFileUrl,
  backendUrl,
  createScene,
  exportScene,
  getAssetOrientation,
  getScene,
  listAssets,
  listScenes,
  saveScene,
  uploadAsset,
} from "./composeApi";
import { errorMessage, firstLine } from "./errorMessage";
import { InspectorPanel } from "./InspectorPanel";
import { ObjectListPanel } from "./ObjectListPanel";
import { ObjectRegistry } from "./registry";
import { composeReducer, initialComposeState, newObjectId } from "./sceneDoc";
import { contentPivot, snapToGround } from "./snap";
import { meshInsertQuaternion } from "./transformMath";
import type { Asset, CropBox, GizmoMode, SceneDoc, SceneObject, SceneSummary, Transform, ViewUp } from "./types";
import { effectiveUp } from "./upVector";
import "./compose.css";

type Status = { kind: "info" | "error"; text: string } | null;

const GIZMO_MODES: { mode: GizmoMode; label: string }[] = [
  { mode: "translate", label: "Taşı (W)" },
  { mode: "rotate", label: "Döndür (E)" },
  { mode: "scale", label: "Ölçekle (R)" },
];

const BACKEND_ERROR = "Backend'e ulaşılamadı";
const POLL_MS = 1000;
/** Consecutive failed status polls before the export is given up locally. */
const MAX_POLL_FAILURES = 5;

function isEditable(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  const tag = el?.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || !!el?.isContentEditable;
}

export function ComposePage({ active }: { active: boolean }) {
  const [state, dispatch] = useReducer(composeReducer, initialComposeState);
  const { doc, selectedId, dirty } = state;
  // Latest state for async actions and window listeners (never a stale closure).
  const stateRef = useRef(state);
  stateRef.current = state;
  const [scenes, setScenes] = useState<SceneSummary[]>([]);
  const [assets, setAssets] = useState<Asset[]>([]);
  const [status, setStatus] = useState<Status>(null);
  // busyRef guards re-entry synchronously; `busy` mirrors it for rendering.
  const busyRef = useRef(false);
  const [busy, setBusy] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [gizmoMode, setGizmoMode] = useState<GizmoMode>("translate");
  const [cropEditing, setCropEditing] = useState(false);
  const [newName, setNewName] = useState("");
  const [exportJob, setExportJob] = useState<Job | null>(null);
  const pollFailures = useRef(0);
  const [pollTick, setPollTick] = useState(0);
  // Last doc object the server has (dirty may lag one render behind a save).
  const lastSavedRef = useRef<SceneDoc | null>(null);
  const registry = useMemo(() => new ObjectRegistry(), []);
  const controlsRef = useRef<OrbitControlsImpl | null>(null);
  const selected = doc?.objects.find((o) => o.id === selectedId) ?? null;

  const refresh = useCallback(async () => {
    try {
      const [s, a] = await Promise.all([listScenes(), listAssets()]);
      setScenes(s);
      setAssets(a);
      setStatus((prev) => (prev?.kind === "error" && prev.text.startsWith(BACKEND_ERROR) ? null : prev));
    } catch (e) {
      setStatus({ kind: "error", text: `${BACKEND_ERROR}: ${errorMessage(e)}` });
    }
  }, []);

  useEffect(() => {
    if (active) void refresh();
  }, [active, refresh]);

  useEffect(() => {
    if (!dirty) return;
    const warn = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  useEffect(() => {
    if (!selected || !selected.crop) setCropEditing(false);
  }, [selected]);

  /**
   * Runs one backend action at a time. `current()` is false once the open
   * scene changed (closed / another one opened) — skip updates then.
   */
  const run = async (label: string, fn: (current: () => boolean) => Promise<void>) => {
    if (busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    setStatus({ kind: "info", text: label });
    const docId = stateRef.current.doc?.id ?? null;
    const current = () => (stateRef.current.doc?.id ?? null) === docId;
    try {
      await fn(current);
    } catch (e) {
      if (current()) setStatus({ kind: "error", text: errorMessage(e) });
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  };

  const resetSceneUi = () => {
    setErrors({});
    setExportJob(null);
    setCropEditing(false);
    pollFailures.current = 0;
    lastSavedRef.current = null;
  };

  const openScene = (id: string) =>
    run("Sahne açılıyor…", async (current) => {
      const loaded = await getScene(id);
      if (!current()) return;
      resetSceneUi();
      dispatch({ type: "load", doc: loaded });
      setStatus(null);
    });

  const createNew = (base: Asset) =>
    run("Sahne oluşturuluyor…", async (current) => {
      const created = await createScene(newName.trim() || base.name, base.id);
      if (!current()) return;
      resetSceneUi();
      dispatch({ type: "load", doc: created });
      setNewName("");
      setStatus(null);
      void refresh();
    });

  const closeScene = () => {
    if (busyRef.current) return;
    if (stateRef.current.dirty && !window.confirm("Kaydedilmemiş değişiklikler kaybolacak. Devam edilsin mi?")) return;
    resetSceneUi();
    setStatus(null);
    dispatch({ type: "close" });
    void refresh();
  };

  const addAsset = (asset: Asset) => {
    const current = stateRef.current.doc;
    if (!current) return;
    const target = controlsRef.current?.target;
    const object: SceneObject = {
      id: newObjectId(),
      kind: asset.kind,
      asset: asset.id,
      name: asset.name,
      role: "object",
      visible: true,
      transform: {
        position: target ? [target.x, target.y, target.z] : [0, 0, 0],
        // glTF is +Y up; turn meshes so their up matches the scene's (any direction).
        quaternion: asset.kind === "mesh" ? meshInsertQuaternion(effectiveUp(current)) : [0, 0, 0, 1],
        scale: 1,
      },
    };
    dispatch({ type: "add", object });
  };

  const upload = (file: File, thenAdd: boolean) =>
    run(`${file.name} yükleniyor…`, async (current) => {
      const asset = await uploadAsset(file);
      const list = await listAssets();
      if (!current()) return;
      setAssets(list);
      if (thenAdd) addAsset(asset);
      setStatus({ kind: "info", text: `${asset.name} yüklendi` });
    });

  const needsSave = () => {
    const { doc: snap, dirty: isDirty } = stateRef.current;
    return !!snap && isDirty && snap !== lastSavedRef.current;
  };

  const save = () => {
    // Snapshot: dirty is cleared only if nothing changed while the request was in flight.
    const snap = stateRef.current.doc;
    if (!snap || !needsSave()) return;
    return run("Kaydediliyor…", async (current) => {
      await saveScene(snap);
      lastSavedRef.current = snap;
      if (!current()) return;
      dispatch({ type: "markSaved", doc: snap });
      setStatus({ kind: "info", text: "Kaydedildi" });
    });
  };
  const saveRef = useRef(save);
  saveRef.current = save;

  const startExport = () => {
    const snap = stateRef.current.doc;
    if (!snap) return;
    const saveFirst = needsSave();
    return run("Export başlatılıyor…", async (current) => {
      if (saveFirst) {
        await saveScene(snap);
        lastSavedRef.current = snap;
        if (current()) dispatch({ type: "markSaved", doc: snap });
      }
      const { job_id } = await exportScene(snap.id);
      const started = await getJobStatus(job_id);
      if (!current()) return;
      pollFailures.current = 0;
      setExportJob(started);
      setStatus(null);
    });
  };

  // Poll the export job; a failed poll retries with backoff, and after
  // MAX_POLL_FAILURES in a row the export is marked failed locally.
  useEffect(() => {
    if (!exportJob || exportJob.status === "completed" || exportJob.status === "failed") return;
    let cancelled = false;
    const delay = POLL_MS * 2 ** pollFailures.current;
    const timer = window.setTimeout(async () => {
      try {
        const next = await getJobStatus(exportJob.id);
        if (cancelled) return;
        pollFailures.current = 0;
        setExportJob(next);
      } catch (e) {
        if (cancelled) return;
        pollFailures.current += 1;
        if (pollFailures.current >= MAX_POLL_FAILURES) {
          pollFailures.current = 0;
          setExportJob({ ...exportJob, status: "failed", error: `Export durumu alınamadı: ${errorMessage(e)}` });
        } else {
          setPollTick((t) => t + 1);
        }
      }
    }, delay);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [exportJob, pollTick]);

  const toggleCrop = (enabled: boolean) => {
    if (!selected) return;
    if (!enabled) {
      dispatch({ type: "setCrop", id: selected.id, crop: null });
      return;
    }
    // Start with the splat's bounds (object-local frame) or a unit box if not loaded yet.
    const bounds = registry.get(selected.id)?.localBounds;
    let crop: CropBox = { center: [0, 0, 0], halfSize: [1, 1, 1], quaternion: [0, 0, 0, 1] };
    if (bounds && !bounds.isEmpty()) {
      const center = bounds.getCenter(new Vector3());
      const size = bounds.getSize(new Vector3());
      crop = {
        center: [center.x, center.y, center.z],
        halfSize: [Math.max(size.x / 2, 1e-3), Math.max(size.y / 2, 1e-3), Math.max(size.z / 2, 1e-3)],
        quaternion: [0, 0, 0, 1],
      };
    }
    dispatch({ type: "setCrop", id: selected.id, crop });
    setCropEditing(true);
  };

  const snap = () => {
    if (!doc || !selected) return;
    const position = snapToGround(doc, registry, selected.id);
    if (!position) {
      setStatus({ kind: "error", text: "Altında yüzey bulunamadı (obje yüklenmemiş olabilir)" });
      return;
    }
    dispatch({ type: "setTransform", id: selected.id, transform: { ...selected.transform, position } });
    setStatus(null);
  };

  /** Toolbar "Yukarı": "auto" = floor-plane up of the base asset, else a fixed ±Y. */
  const chooseUp = (choice: string) => {
    if (choice === "y" || choice === "-y") {
      dispatch({ type: "setUp", up: null, viewUp: choice as ViewUp });
      return;
    }
    if (choice !== "auto") return;
    const base = stateRef.current.doc?.objects.find((o) => o.role === "base");
    if (!base) return;
    void run("Zemin yönü hesaplanıyor…", async (current) => {
      const o = await getAssetOrientation(base.asset);
      if (!current()) return;
      if (!o.measured) {
        setStatus({ kind: "error", text: "Zemin düzlemi bulunamadı; +Y/−Y seçin" });
        return;
      }
      dispatch({ type: "setUp", up: o.up });
      setStatus(null);
    });
  };

  /**
   * "Dikleştir": rotate a splat object so its own floor up matches the scene
   * up, pivoting on its visible content. The reducer applies it to the
   * object's current transform, so edits made during the request are kept.
   */
  const straighten = (id: string) => {
    const obj = stateRef.current.doc?.objects.find((o) => o.id === id);
    if (!obj || obj.kind !== "splat" || obj.role === "base") return;
    void run("Obje yönü hesaplanıyor…", async (current) => {
      const o = await getAssetOrientation(obj.asset);
      if (!current()) return;
      const latest = stateRef.current.doc?.objects.find((x) => x.id === id);
      if (!latest) {
        setStatus(null); // deleted meanwhile
        return;
      }
      if (!o.measured) {
        setStatus({ kind: "error", text: "Bu objede zemin düzlemi bulunamadı" });
        return;
      }
      const entry = registry.get(id);
      const localPivot = (entry && contentPivot(latest, entry)) ?? latest.crop?.center ?? [0, 0, 0];
      dispatch({ type: "straighten", id, objUp: o.up, localPivot });
      setStatus(null);
    });
  };

  const hasDoc = !!doc;
  useEffect(() => {
    if (!active || !hasDoc) return;
    const onKey = (e: KeyboardEvent) => {
      const mod = e.ctrlKey || e.metaKey;
      const key = e.key.toLowerCase();
      if (mod && key === "s") {
        e.preventDefault();
        // Blur first so a pending field value commits (onBlur), then save the result.
        const focused = document.activeElement;
        if (focused instanceof HTMLElement) focused.blur();
        window.setTimeout(() => void saveRef.current(), 0);
        return;
      }
      if (mod && key === "d") {
        e.preventDefault();
        const id = stateRef.current.selectedId;
        if (!e.repeat && id) dispatch({ type: "duplicate", id, newId: newObjectId() });
        return;
      }
      if (mod || e.altKey || isEditable(e.target)) return;
      switch (key) {
        case "w":
          setGizmoMode("translate");
          break;
        case "e":
          setGizmoMode("rotate");
          break;
        case "r":
          setGizmoMode("scale");
          break;
        case "escape":
          dispatch({ type: "select", id: null });
          setCropEditing(false);
          break;
        case "delete":
        case "backspace": {
          e.preventDefault();
          const id = stateRef.current.selectedId;
          if (!e.repeat && id) dispatch({ type: "remove", id });
          break;
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [active, hasDoc]);

  const onError = useCallback((id: string, msg: string) => setErrors((prev) => ({ ...prev, [id]: msg })), []);
  const onSelect = useCallback((id: string | null) => dispatch({ type: "select", id }), []);
  const onTransformCommit = useCallback(
    (id: string, transform: Transform) => dispatch({ type: "setTransform", id, transform }),
    [],
  );
  const onCropCommit = useCallback((id: string, crop: CropBox) => dispatch({ type: "setCrop", id, crop }), []);
  const onControls = useCallback((next: OrbitControlsImpl | null, released?: OrbitControlsImpl) => {
    if (next) controlsRef.current = next;
    else if (controlsRef.current === released) controlsRef.current = null;
  }, []);

  const statusBar = status && <div className={`compose-status ${status.kind}`}>{status.text}</div>;

  if (!doc) {
    return (
      <div className="compose-home">
        <section>
          <h2>Sahneler</h2>
          {scenes.length === 0 && <p className="muted">Henüz kayıtlı sahne yok.</p>}
          <ul className="compose-scene-list">
            {scenes.map((s) => (
              <li key={s.id}>
                <span>{s.name}</span>
                <small>
                  {s.object_count} obje · {new Date(s.updated_ts * 1000).toLocaleString()}
                </small>
                <button type="button" className="btn-primary" disabled={busy} onClick={() => openScene(s.id)}>
                  Aç
                </button>
              </li>
            ))}
          </ul>
        </section>
        <section>
          <h2>Yeni sahne</h2>
          <p className="muted">Base sahne olarak bir splat seç (bizim pipeline sonucu ya da Spirula'dan indirilen .ply).</p>
          <label className="compose-field">
            <span>Ad</span>
            <input value={newName} maxLength={128} onChange={(e) => setNewName(e.target.value)} placeholder="ör. bahce_heykel" />
          </label>
          <AssetPicker
            assets={assets}
            kinds={["splat"]}
            busy={busy}
            actionLabel="Oluştur"
            onPick={createNew}
            onUpload={(f) => upload(f, false)}
          />
        </section>
        {statusBar}
      </div>
    );
  }

  const exporting = exportJob?.status === "running" || exportJob?.status === "queued";

  return (
    <div className="compose-editor">
      <div className="compose-toolbar">
        <button type="button" className="btn-secondary" disabled={busy} onClick={closeScene}>
          ← Sahneler
        </button>
        <strong className="compose-title">
          {doc.name}
          {dirty ? " •" : ""}
        </strong>
        <div className="compose-segment" role="group" aria-label="Gizmo modu">
          {GIZMO_MODES.map(({ mode, label }) => (
            <button
              key={mode}
              type="button"
              className={gizmoMode === mode ? "active" : ""}
              aria-pressed={gizmoMode === mode}
              onClick={() => setGizmoMode(mode)}
            >
              {label}
            </button>
          ))}
        </div>
        <label className="compose-inline">
          Yukarı
          <select value={doc.up ? "auto" : doc.viewUp} disabled={busy} onChange={(e) => chooseUp(e.target.value)}>
            <option value="auto">Otomatik (zemin)</option>
            <option value="y">+Y</option>
            <option value="-y">−Y (COLMAP)</option>
          </select>
        </label>
        <span className="compose-spacer" />
        <button type="button" className="btn-secondary" disabled={busy || !dirty} onClick={() => void save()}>
          Kaydet
        </button>
        <button type="button" className="btn-primary" disabled={busy || exporting} onClick={() => void startExport()}>
          Export
        </button>
        {exportJob && (
          <span className="compose-export">
            {exportJob.status === "completed" && exportJob.download_url ? (
              <a className="btn-secondary" href={backendUrl(exportJob.download_url)} target="_blank" rel="noreferrer">
                .zip indir
              </a>
            ) : exportJob.status === "failed" ? (
              <details className="compose-export-error">
                <summary>Export başarısız{exportJob.error ? `: ${firstLine(exportJob.error)}` : ""}</summary>
                {exportJob.error && <pre>{exportJob.error}</pre>}
              </details>
            ) : (
              <span>
                {Math.round(exportJob.overall_progress * 100)}% · {exportJob.phase?.message ?? ""}
              </span>
            )}
          </span>
        )}
      </div>
      <div className="compose-body">
        <aside className="compose-left">
          <h3>Objeler</h3>
          <ObjectListPanel
            objects={doc.objects}
            selectedId={selectedId}
            errors={errors}
            onSelect={onSelect}
            onToggleVisible={(id, visible) => dispatch({ type: "setVisible", id, visible })}
            onDuplicate={(id) => dispatch({ type: "duplicate", id, newId: newObjectId() })}
            onRemove={(id) => dispatch({ type: "remove", id })}
          />
          <h3>Obje ekle</h3>
          <AssetPicker assets={assets} busy={busy} actionLabel="Sahneye ekle" onPick={addAsset} onUpload={(f) => upload(f, true)} />
        </aside>
        <section className="compose-viewport" aria-label="Sahne görünümü">
          <ComposeViewport
            doc={doc}
            active={active}
            selectedId={selectedId}
            gizmoMode={gizmoMode}
            cropEditing={cropEditing}
            registry={registry}
            assetUrl={assetFileUrl}
            onSelect={onSelect}
            onTransformCommit={onTransformCommit}
            onCropCommit={onCropCommit}
            onError={onError}
            onControls={onControls}
          />
        </section>
        <aside className="compose-right">
          {selected ? (
            <InspectorPanel
              key={selected.id}
              object={selected}
              cropEditing={cropEditing}
              onRename={(name) => dispatch({ type: "rename", id: selected.id, name })}
              onTransform={(transform) => dispatch({ type: "setTransform", id: selected.id, transform })}
              onToggleCrop={toggleCrop}
              onCropEditing={setCropEditing}
              onColor={(color) => dispatch({ type: "setColor", id: selected.id, color })}
              onSnap={snap}
              onStraighten={() => straighten(selected.id)}
              busy={busy}
            />
          ) : (
            <p className="muted">Düzenlemek için bir obje seç.</p>
          )}
        </aside>
      </div>
      {statusBar}
    </div>
  );
}
