import { act, fireEvent, render, screen } from "@testing-library/react";
import { useLayoutEffect, useRef, useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { InspectorPanel } from "../InspectorPanel";
import type { SceneObject, Transform } from "../types";

const OBJ: SceneObject = {
  id: "o1",
  kind: "splat",
  asset: "a1",
  name: "statue",
  role: "object",
  visible: true,
  transform: { position: [0, 0, 0], quaternion: [0, 0, 0, 1], scale: 1 },
};

const noop = () => {};

/**
 * Stateful parent. When `focusOnCommit` holds a label, the next committed
 * transform moves focus to that field from a layout effect, i.e. after the
 * new props are committed but before passive effects run: the window in
 * which the old NumberField re-committed its stale text on blur.
 */
function Harness({
  onTransform,
  focusOnCommit,
  onStraighten,
}: {
  onTransform: (t: Transform) => void;
  focusOnCommit: { current: string | null };
  onStraighten?: () => void;
}) {
  const [obj, setObj] = useState(OBJ);
  const first = useRef(true);
  useLayoutEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    const label = focusOnCommit.current;
    focusOnCommit.current = null;
    if (label) (document.querySelector(`[aria-label="${label}"]`) as HTMLInputElement).focus();
  }, [obj, focusOnCommit]);
  return (
    <InspectorPanel
      object={obj}
      cropEditing={false}
      onRename={noop}
      onTransform={(t) => {
        onTransform(t);
        setObj((o) => ({ ...o, transform: t }));
      }}
      onToggleCrop={noop}
      onCropEditing={noop}
      onColor={noop}
      onSnap={noop}
      onStraighten={onStraighten}
    />
  );
}

function field(label: string) {
  return screen.getByLabelText(label) as HTMLInputElement;
}

describe("InspectorPanel NumberField", () => {
  it("keeps an Enter-committed value when focus moves on before the prop update settles", () => {
    const onTransform = vi.fn();
    const focusOnCommit = { current: null as string | null };
    render(<Harness onTransform={onTransform} focusOnCommit={focusOnCommit} />);

    act(() => field("Konum Y").focus());
    fireEvent.change(field("Konum Y"), { target: { value: "0.5" } });
    focusOnCommit.current = "Konum Z";
    fireEvent.keyDown(field("Konum Y"), { key: "Enter" });
    expect(document.activeElement).toBe(field("Konum Z"));

    fireEvent.change(field("Konum Z"), { target: { value: "0.7" } });
    fireEvent.keyDown(field("Konum Z"), { key: "Enter" });
    act(() => field("Konum Z").blur());

    const positions = onTransform.mock.calls.map(([t]) => (t as Transform).position);
    expect(positions).toEqual([
      [0, 0.5, 0],
      [0, 0.5, 0.7],
    ]);
    expect(field("Konum Y").value).toBe("0.500");
    expect(field("Konum Z").value).toBe("0.700");
  });

  it("commits once on Enter followed by blur", () => {
    const onTransform = vi.fn();
    render(<Harness onTransform={onTransform} focusOnCommit={{ current: null }} />);
    act(() => field("Konum X").focus());
    fireEvent.change(field("Konum X"), { target: { value: "2" } });
    fireEvent.keyDown(field("Konum X"), { key: "Enter" });
    act(() => field("Konum X").blur());
    expect(onTransform).toHaveBeenCalledTimes(1);
    expect(field("Konum X").value).toBe("2.000");
  });

  it("restores the shown value after a rejected edit", () => {
    const onTransform = vi.fn();
    render(<Harness onTransform={onTransform} focusOnCommit={{ current: null }} />);
    act(() => field("Ölçek").focus());
    fireEvent.change(field("Ölçek"), { target: { value: "-1" } });
    fireEvent.keyDown(field("Ölçek"), { key: "Enter" });
    expect(onTransform).not.toHaveBeenCalled();
    expect(field("Ölçek").value).toBe("1.000");

    fireEvent.change(field("Konum X"), { target: { value: "" } });
    fireEvent.blur(field("Konum X"));
    expect(onTransform).not.toHaveBeenCalled();
    expect(field("Konum X").value).toBe("0.000");
  });

  it("shows Dikleştir for non-base splats only", () => {
    const onStraighten = vi.fn();
    const { rerender } = render(
      <Harness onTransform={noop} focusOnCommit={{ current: null }} onStraighten={onStraighten} />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Dikleştir" }));
    expect(onStraighten).toHaveBeenCalledTimes(1);

    rerender(
      <InspectorPanel
        object={{ ...OBJ, kind: "mesh" }}
        cropEditing={false}
        onRename={noop}
        onTransform={noop}
        onToggleCrop={noop}
        onCropEditing={noop}
        onColor={noop}
        onSnap={noop}
        onStraighten={onStraighten}
      />,
    );
    expect(screen.queryByRole("button", { name: "Dikleştir" })).not.toBeInTheDocument();
  });
});
