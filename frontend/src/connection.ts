/**
 * Backend connection state — apiBase + authToken.
 *
 * Module-level + localStorage-backed. Components subscribe via
 * onConnectionChange(); api.ts reads getApiBase() / getAuthToken() at every
 * call so a settings change flips behavior immediately, no app reload.
 */

export interface Connection {
  apiBase: string;
  authToken: string;
}

const STORAGE_KEY = "4dgs.connection";

const DEFAULT: Connection = {
  apiBase: "http://127.0.0.1:8000",
  authToken: "",
};

function loadFromStorage(): Connection {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULT };
    const parsed = JSON.parse(raw);
    return {
      apiBase: typeof parsed.apiBase === "string" && parsed.apiBase ? parsed.apiBase : DEFAULT.apiBase,
      authToken: typeof parsed.authToken === "string" ? parsed.authToken : "",
    };
  } catch {
    return { ...DEFAULT };
  }
}

let _state: Connection = loadFromStorage();
const _listeners = new Set<(c: Connection) => void>();

export function getConnection(): Connection {
  return _state;
}

export function getApiBase(): string {
  return _state.apiBase.replace(/\/+$/, "");
}

export function getAuthToken(): string {
  return _state.authToken;
}

export function setConnection(next: Connection): void {
  const cleaned: Connection = {
    apiBase: next.apiBase.trim().replace(/\/+$/, ""),
    authToken: next.authToken.trim(),
  };
  _state = cleaned;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(cleaned));
  } catch {
    /* ignore quota / private-mode errors */
  }
  for (const cb of _listeners) cb(cleaned);
}

export function onConnectionChange(cb: (c: Connection) => void): () => void {
  _listeners.add(cb);
  return () => _listeners.delete(cb);
}

export function isLocal(): boolean {
  const base = _state.apiBase;
  return base.includes("127.0.0.1") || base.includes("localhost");
}

/** Append ?token=... to a URL when an auth token is set. Used for endpoints
 *  loaded via <video src=...> or by libraries that don't carry our headers. */
export function withTokenParam(url: string): string {
  const tok = _state.authToken;
  if (!tok) return url;
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}token=${encodeURIComponent(tok)}`;
}
