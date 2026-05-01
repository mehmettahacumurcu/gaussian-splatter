/**
 * ConnectionSettings — modal for choosing the backend.
 *
 * Two inputs (API Base URL, Auth Token) + Test Connection + Save / Cancel.
 * Persists to localStorage via connection.ts; api.ts reads it per-call so
 * a save flips behavior immediately.
 */
import { useEffect, useRef, useState } from "react";
import {
  getConnection,
  setConnection,
  type Connection,
} from "../connection";

interface Props {
  open: boolean;
  onClose: () => void;
}

type TestState =
  | { kind: "idle" }
  | { kind: "testing" }
  | { kind: "ok"; gpu: string | null; gpuAvailable: boolean }
  | { kind: "err"; message: string };

export function ConnectionSettings({ open, onClose }: Props) {
  const initial = getConnection();
  const [apiBase, setApiBase] = useState(initial.apiBase);
  const [authToken, setAuthToken] = useState(initial.authToken);
  const [test, setTest] = useState<TestState>({ kind: "idle" });
  const [showToken, setShowToken] = useState(false);
  const dialogRef = useRef<HTMLDivElement | null>(null);

  // Sync local state when the modal re-opens.
  useEffect(() => {
    if (open) {
      const c = getConnection();
      setApiBase(c.apiBase);
      setAuthToken(c.authToken);
      setTest({ kind: "idle" });
    }
  }, [open]);

  // Click-outside to close.
  useEffect(() => {
    if (!open) return;
    const onDocClick = (e: MouseEvent) => {
      if (dialogRef.current && !dialogRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    const onEsc = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("mousedown", onDocClick);
    window.addEventListener("keydown", onEsc);
    return () => {
      window.removeEventListener("mousedown", onDocClick);
      window.removeEventListener("keydown", onEsc);
    };
  }, [open, onClose]);

  if (!open) return null;

  const cleanedBase = apiBase.trim().replace(/\/+$/, "");

  const handleTest = async () => {
    if (!cleanedBase) return;
    setTest({ kind: "testing" });
    try {
      const headers = new Headers();
      if (authToken.trim()) headers.set("Authorization", `Bearer ${authToken.trim()}`);
      const res = await fetch(`${cleanedBase}/`, { headers });
      if (!res.ok) {
        const body = await res.text().catch(() => "");
        setTest({ kind: "err", message: `HTTP ${res.status} ${res.statusText}: ${body || "(empty)"}` });
        return;
      }
      const data = await res.json();
      setTest({
        kind: "ok",
        gpu: typeof data?.gpu_name === "string" ? data.gpu_name : null,
        gpuAvailable: Boolean(data?.gpu_available),
      });
    } catch (e) {
      setTest({ kind: "err", message: String(e) });
    }
  };

  const handleSave = () => {
    const next: Connection = {
      apiBase: cleanedBase || "http://127.0.0.1:8000",
      authToken: authToken.trim(),
    };
    setConnection(next);
    onClose();
  };

  const handleResetLocal = () => {
    setApiBase("http://127.0.0.1:8000");
    setAuthToken("");
    setTest({ kind: "idle" });
  };

  return (
    <div className="conn-settings-overlay" role="dialog" aria-modal="true">
      <div className="conn-settings" ref={dialogRef}>
        <div className="conn-settings-header">
          <h3>Backend Connection</h3>
          <button
            type="button"
            className="conn-settings-close"
            aria-label="Close"
            onClick={onClose}
          >
            ×
          </button>
        </div>

        <div className="conn-settings-body">
          <label className="conn-settings-field">
            <span>API Base URL</span>
            <input
              type="text"
              spellCheck={false}
              autoComplete="off"
              placeholder="http://127.0.0.1:8000"
              value={apiBase}
              onChange={(e) => setApiBase(e.target.value)}
            />
            <div className="conn-settings-hint">
              For RunPod: <code>https://&lt;pod-id&gt;-8000.proxy.runpod.net</code>
            </div>
          </label>

          <label className="conn-settings-field">
            <span>Auth Token (Bearer)</span>
            <div style={{ display: "flex", gap: 6 }}>
              <input
                type={showToken ? "text" : "password"}
                spellCheck={false}
                autoComplete="off"
                placeholder="(empty for local dev)"
                value={authToken}
                onChange={(e) => setAuthToken(e.target.value)}
                style={{ flex: 1 }}
              />
              <button
                type="button"
                className="btn-secondary"
                onClick={() => setShowToken((s) => !s)}
                style={{ padding: "4px 10px" }}
              >
                {showToken ? "Hide" : "Show"}
              </button>
            </div>
            <div className="conn-settings-hint">
              Set on the pod with{" "}
              <code>export RUNPOD_AUTH_TOKEN=$(openssl rand -hex 32)</code>{" "}
              before starting the backend.
            </div>
          </label>

          {test.kind === "ok" && (
            <div className="conn-settings-status conn-settings-ok">
              ✓ Reachable.
              {test.gpuAvailable
                ? <> GPU available{test.gpu ? <>: <code>{test.gpu}</code></> : null}.</>
                : <> No GPU detected.</>}
            </div>
          )}
          {test.kind === "err" && (
            <div className="conn-settings-status conn-settings-err">
              ✗ {test.message}
            </div>
          )}
          {test.kind === "testing" && (
            <div className="conn-settings-status">Testing…</div>
          )}
        </div>

        <div className="conn-settings-actions">
          <button
            type="button"
            className="btn-secondary"
            onClick={handleResetLocal}
            title="Reset to http://127.0.0.1:8000 + empty token"
          >
            Reset to local
          </button>
          <div style={{ flex: 1 }} />
          <button
            type="button"
            className="btn-secondary"
            onClick={handleTest}
            disabled={test.kind === "testing" || !cleanedBase}
          >
            Test connection
          </button>
          <button
            type="button"
            className="btn-primary"
            onClick={handleSave}
            disabled={!cleanedBase}
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
}
