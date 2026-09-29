/**
 * Short readable text for an error thrown by `fetchJson`
 * (`HTTP <code> <status>: <body>`). FastAPI bodies are `{"detail": ...}`:
 * a string for 400/404/409, a list of `{loc, msg, type}` for 422.
 */
export function errorMessage(e: unknown): string {
  const text = e instanceof Error ? e.message : String(e);
  const start = text.indexOf("{");
  if (start < 0) return text;
  let body: unknown;
  try {
    body = JSON.parse(text.slice(start));
  } catch {
    return text;
  }
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length > 0) {
    return detail
      .map((d) => {
        const item = d as { loc?: unknown; msg?: unknown };
        const msg = typeof item.msg === "string" ? item.msg : JSON.stringify(d);
        const loc = Array.isArray(item.loc) ? item.loc.slice(1).join(".") : "";
        return loc ? `${loc}: ${msg}` : msg;
      })
      .join("; ");
  }
  return text;
}
