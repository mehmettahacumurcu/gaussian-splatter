import type { SceneObject } from "./types";

interface Props {
  objects: SceneObject[];
  selectedId: string | null;
  errors: Record<string, string>;
  onSelect: (id: string) => void;
  onToggleVisible: (id: string, visible: boolean) => void;
  onDuplicate: (id: string) => void;
  onRemove: (id: string) => void;
}

export function ObjectListPanel({ objects, selectedId, errors, onSelect, onToggleVisible, onDuplicate, onRemove }: Props) {
  return (
    <ul className="compose-object-list">
      {objects.map((o) => (
        <li
          key={o.id}
          className={o.id === selectedId ? "selected" : ""}
          tabIndex={0}
          aria-current={o.id === selectedId ? "true" : undefined}
          onClick={() => onSelect(o.id)}
          onKeyDown={(e) => {
            if (e.target !== e.currentTarget || (e.key !== "Enter" && e.key !== " ")) return;
            e.preventDefault();
            onSelect(o.id);
          }}
        >
          <button
            type="button"
            className="compose-icon-btn"
            title={o.visible ? "Gizle" : "Göster"}
            aria-label={`${o.name} ${o.visible ? "gizle" : "göster"}`}
            aria-pressed={!o.visible}
            onClick={(e) => {
              e.stopPropagation();
              onToggleVisible(o.id, !o.visible);
            }}
          >
            {o.visible ? "👁" : "—"}
          </button>
          <span className="compose-object-name" title={o.name}>
            {o.name}
          </span>
          <span className="compose-badge">{o.kind === "splat" ? "splat" : "mesh"}</span>
          {o.role === "base" && <span className="compose-badge base">base</span>}
          {errors[o.id] && (
            <span className="compose-badge error" title={errors[o.id]}>
              hata
            </span>
          )}
          {o.role !== "base" && o.id === selectedId && (
            <span className="compose-object-actions">
              <button
                type="button"
                className="compose-icon-btn"
                title="Kopyala (Ctrl+D)"
                aria-label={`${o.name} kopyala`}
                onClick={(e) => {
                  e.stopPropagation();
                  onDuplicate(o.id);
                }}
              >
                ⧉
              </button>
              <button
                type="button"
                className="compose-icon-btn"
                title="Sil (Del)"
                aria-label={`${o.name} sil`}
                onClick={(e) => {
                  e.stopPropagation();
                  onRemove(o.id);
                }}
              >
                ✕
              </button>
            </span>
          )}
        </li>
      ))}
    </ul>
  );
}
