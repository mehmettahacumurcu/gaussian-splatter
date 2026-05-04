interface Props {
  onSelect: (mode: "A" | "B") => void;
  onCancel: () => void;
}

export function EditQualityDialog({ onSelect, onCancel }: Props) {
  return (
    <div style={{
      position: "fixed", inset: 0, background: "rgba(0,0,0,0.7)",
      display: "flex", alignItems: "center", justifyContent: "center",
      zIndex: 1000,
    }}>
      <div style={{
        background: "#1c1c1c", padding: 24, borderRadius: 8,
        minWidth: 360, maxWidth: 480, color: "#eee",
      }}>
        <h3 style={{ marginTop: 0 }}>Quality mode?</h3>
        <p>The edit will SAM-segment, inpaint, and refit. Pick:</p>
        <button
          onClick={() => onSelect("A")}
          style={{ display: "block", width: "100%", margin: "8px 0", padding: 12 }}
        >
          Quick preview (~5 min total, LaMa)
        </button>
        <button
          onClick={() => onSelect("B")}
          style={{ display: "block", width: "100%", margin: "8px 0", padding: 12 }}
        >
          Full quality (~15-20 min total, SD)
        </button>
        <button
          onClick={onCancel}
          style={{ display: "block", width: "100%", margin: "12px 0 0", padding: 8, background: "#333" }}
        >
          Cancel
        </button>
      </div>
    </div>
  );
}
