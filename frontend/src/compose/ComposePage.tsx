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
  getScene,
  listAssets,
  listScenes,
  saveScene,
  uploadAsset,
} from "./composeApi";
import { errorMessage } from "./errorMessage";
import { InspectorPanel } from "./InspectorPanel";
import { ObjectListPanel } from "./ObjectListPanel";
import { ObjectRegistry } from "./registry";
import { composeReducer, initialComposeState, newObjectId } from "./sceneDoc";
import { snapToGround } from "./snap";
import type { Asset, CropBox, GizmoMode, SceneObject, SceneSummary, Transform } from "./types";
import "./compose.css";

type Status = { kind: "info" | "error"; text: string } | null;

const GIZMO_MODES: { mode: GizmoMode; label: string }[] = [
  { mode: "translate", label: "Taşı (W)" },
  { mode: "rotate", label: "Döndür (E)" },
  { mode: "scale", label: "Ölçekle (R)" },
];

export function ComposePage({ active }: { active: boolean }) {
  const [state, dispatch] = useReducer(composeReducer, initialComposeState);
  const { doc, selectedId, dirty } = state;
  const [scenes, setScenes] = useState<SceneSummary[]>([]);
  const [assets, setAssets] = useState<Asset[]>([]);
  const [status, setStatus] = useState<Status>(null);
  const [busy, setBusy] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [gizmoMode, setGizmoMode] = useState<GizmoMode>("translate");
  const [cropEditing, setCropEditing] = useState(false);
  const [newName, setNewName] = useState("");
  const [exportJob, setExportJob] = useState<Job | null>(null);
  const registry = useMemo(() => new ObjectRegistry(), []);
  const controlsRef = useRef<OrbitControlsImpl | null>(null);
  const selected = doc?.objects.find((o) => o.id === selectedId) ?? null;

  const refresh = useCallback(async () => {
    try {
      const [s, a] = await Promise.all([listScenes(), listAssets()]);
      setScenes(s);
      setAssets(a);
    } catch (e) {
      setStatus({ kind: "error", text: `Backend'e ulaşılamadı: ${errorMessage(e)}` });
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

  const run = async (label: string, fn: () => Promise<void>) => {
    setBusy(true);
    setStatus({ kind: "info", text: label });
    try {
      await fn();
    } catch (e) {
      setStatus({ kind: "error", text: errorMessage(e) });
    } finally {
      setBusy(false);
    }
  };

  const resetSceneUi = () => {
    setErrors({});
    setExportJob(null);
    setCropEditing(false);
  };

  const openScene = (id: string) =>
    run("Sahne açılıyor…", async () => {
      const loaded = await getScene(id);
      resetSceneUi();
      dispatch({ type: "load", doc: loaded });
      setStatus(null);
    });

  const createNew = (base: Asset) =>
    run("Sahne oluşturuluyor…", async () => {
      const created = await createScene(newName.trim() || base.name, base.id);
      resetSceneUi();
      dispatch({ type: "load", doc: created });
      setNewName("");
      setStatus(null);
      void refresh();
    });

  const closeScene = () => {
    if (dirty && !window.confirm("Kaydedilmemiş değişiklikler kaybolacak. Devam edilsin mi?")) return;
    resetSceneUi();
    setStatus(null);
    dispatch({ type: "close" });
    void refresh();
  };

  const addAsset = (asset: Asset) => {
    if (!doc) return;
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
        // glTF is +Y up; flip meshes into COLMAP-style (−Y up) scenes.
        quaternion: asset.kind === "mesh" && doc.viewUp === "-y" ? [1, 0, 0, 0] : [0, 0, 0, 1],
        scale: 1,
      },
    };
    dispatch({ type: "add", object });
  };

  const upload = (file: File, thenAdd: boolean) =>
    run(`${file.name} yükleniyor…`, async () => {
      const asset = await uploadAsset(file);
      setAssets(await listAssets());
      if (thenAdd) addAsset(asset);
      setStatus({ kind: "info", text: `${asset.name} yüklendi` });
    });

  const save = () => {
    // Snapshot: dirty is cleared only if nothing changed while the request was in flight.
    const snap = doc;
    if (!snap) return;
    return run("Kaydediliyor…", async () => {
      await saveScene(snap);
      dispatch({ type: "markSaved", doc: snap });
      setStatus({ kind: "info", text: "Kaydedildi" });
    });
  };

  const startExport = () => {
    const snap = doc;
    if (!snap) return;
    const saveFirst = dirty;
    return run("Export başlatılıyor…", async () => {
      if (saveFirst) {
        await saveScene(snap);
        dispatch({ type: "markSaved", doc: snap });
      }
      const { job_id } = await exportScene(snap.id);
      setExportJob(await getJobStatus(job_id));
      setStatus(null);
    });
  };

  useEffect(() => {
    if (!exportJob || exportJob.status === "completed" || exportJob.status === "failed") return;
    let cancelled = false;
    const timer = window.setTimeout(async () => {
      try {
        const next = await getJobStatus(exportJob.id);
        if (!cancelled) setExportJob(next);
      } catch (e) {
        if (!cancelled) setStatus({ kind: "error", text: errorMessage(e) });
      }
    }, 1000);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [exportJob]);

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

  useEffect(() => {
    if (!active || !doc) return;
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      const tag = el?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el?.isContentEditable) return;
      const mod = e.ctrlKey || e.metaKey;
      if (mod && e.key.toLowerCase() === "d") {
        e.preventDefault();
        if (selectedId) dispatch({ type: "duplicate", id: selectedId, newId: newObjectId() });
        return;
      }
      if (mod && e.key.toLowerCase() === "s") {
        e.preventDefault();
        void save();
        return;
      }
      if (mod || e.altKey) return;
      switch (e.key.toLowerCase()) {
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
          if (selectedId) dispatch({ type: "remove", id: selectedId });
          break;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, doc, selectedId]);

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
        <button type="button" className="btn-secondary" onClick={closeScene}>
          ← Sahneler
        </button>
        <strong className="compose-title">
          {doc.name}
          {dirty ? " •" : ""}
        </strong>
        <div className="compose-segment">
          {GIZMO_MODES.map(({ mode, label }) => (
            <button key={mode} type="button" className={gizmoMode === mode ? "active" : ""} onClick={() => setGizmoMode(mode)}>
              {label}
            </button>
          ))}
        </div>
        <label className="compose-inline">
          Yukarı
          <select value={doc.viewUp} onChange={(e) => dispatch({ type: "setViewUp", viewUp: e.target.value as "y" | "-y" })}>
            <option value="y">+Y</option>
            <option value="-y">−Y (COLMAP)</option>
          </select>
        </label>
        <span className="compose-spacer" />
        <button type="button" className="btn-secondary" disabled={busy || !dirty} onClick={save}>
          Kaydet
        </button>
        <button type="button" className="btn-primary" disabled={busy || exporting} onClick={startExport}>
          Export
        </button>
        {exportJob && (
          <span className="compose-export">
            {exportJob.status === "completed" && exportJob.download_url ? (
              <a className="btn-secondary" href={backendUrl(exportJob.download_url)} target="_blank" rel="noreferrer">
                .zip indir
              </a>
            ) : exportJob.status === "failed" ? (
              <span className="err" title={exportJob.error ?? ""}>
                Export başarısız
              </span>
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
        <main className="compose-viewport">
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
        </main>
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
