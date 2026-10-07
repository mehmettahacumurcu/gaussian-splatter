import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MorphPanel } from "../MorphPanel";
import { INITIAL_MORPH_STATE, selectMorphSlot, type MorphState } from "../morphTypes";
import { ObjectListPanel } from "../ObjectListPanel";
import type { SceneObject } from "../types";

const objects: SceneObject[] = ["a", "b"].map((id) => ({
  id,
  kind: "splat",
  asset: id,
  name: `splat ${id}`,
  role: "object",
  visible: true,
  transform: { position: [0, 0, 0], quaternion: [0, 0, 0, 1], scale: 1 },
}));
const prepared: MorphState = { ...INITIAL_MORPH_STATE, sourceId: "a", targetId: "b", enabled: true };

afterEach(() => vi.useRealTimers());

describe("Morph controls", () => {
  it("requires two distinct splats and offers cancellation while loading", () => {
    const onChange = vi.fn();
    const { rerender } = render(
      <MorphPanel morph={INITIAL_MORPH_STATE} playback={{ t: 0, playing: false }} status={{ phase: "idle" }} objects={objects} onChange={onChange} />,
    );
    expect(screen.getByRole("button", { name: "Morph hazırla" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Oynat" })).toBeDisabled();
    rerender(<MorphPanel morph={prepared} playback={{ t: 0, playing: false }} status={{ phase: "loading" }} objects={objects} onChange={onChange} />);
    expect(screen.getByRole("button", { name: "Oynat" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Önizlemeyi kapat" }));
    expect(onChange).toHaveBeenCalledWith({ enabled: false, playing: false, t: 0 });
  });

  it("samples live renderer time and pauses at that time without dispatching ticks", () => {
    vi.useFakeTimers();
    const onChange = vi.fn();
    const playback = { t: 0.43, playing: true };
    render(<MorphPanel morph={{ ...prepared, playing: true }} playback={playback} status={{ phase: "ready", count: 200000 }} objects={objects} onChange={onChange} />);
    act(() => vi.advanceTimersByTime(100));
    expect(screen.getByLabelText("Morph ilerleme")).toHaveTextContent("43%");
    expect(onChange).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Duraklat" }));
    expect(onChange).toHaveBeenCalledWith({ playing: false, t: 0.43 });
  });

  it("scrubs exactly to either endpoint, pauses, and can replay from B", () => {
    const onChange = vi.fn();
    render(<MorphPanel morph={{ ...prepared, t: 1 }} playback={{ t: 1, playing: false }} status={{ phase: "ready" }} objects={objects} onChange={onChange} />);
    fireEvent.click(screen.getByRole("button", { name: "Oynat" }));
    expect(onChange).toHaveBeenLastCalledWith({ playing: true, t: 0 });
    fireEvent.change(screen.getByLabelText("Zaman çizelgesi"), { target: { value: "0" } });
    expect(onChange).toHaveBeenLastCalledWith({ playing: false, t: 0 });
    fireEvent.change(screen.getByLabelText("Zaman çizelgesi"), { target: { value: "0.625" } });
    expect(onChange).toHaveBeenLastCalledWith({ playing: false, t: 0.625 });
  });

  it("clamps duration and seed before committing and rejects an empty value", () => {
    const onChange = vi.fn();
    render(<MorphPanel morph={prepared} playback={{ t: 0, playing: false }} status={{ phase: "ready" }} objects={objects} onChange={onChange} />);
    const duration = screen.getByLabelText("Süre (sn)");
    fireEvent.change(duration, { target: { value: "0" } });
    fireEvent.blur(duration);
    expect(onChange).toHaveBeenLastCalledWith({ duration: 0.5 });
    const seed = screen.getByLabelText("Seed");
    fireEvent.change(seed, { target: { value: "-1" } });
    fireEvent.blur(seed);
    expect(onChange).toHaveBeenLastCalledWith({ seed: 0, playing: false, t: 0 });
    onChange.mockClear();
    fireEvent.change(seed, { target: { value: "" } });
    fireEvent.blur(seed);
    expect(seed).toHaveValue(42);
    expect(onChange).not.toHaveBeenCalled();
  });

  it("reports preparation errors and leaves close available", () => {
    render(<MorphPanel morph={prepared} playback={{ t: 0, playing: false }} status={{ phase: "error", message: "No splats" }} objects={objects} onChange={vi.fn()} />);
    expect(screen.getByRole("status", { name: "Morph durumu" })).toHaveTextContent("No splats");
    expect(screen.getByRole("button", { name: "Oynat" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Önizlemeyi kapat" })).not.toBeDisabled();
  });

  it("offers recording only for a ready pair with a recording handler", () => {
    const onRecord = vi.fn();
    const props = { morph: prepared, playback: { t: 0, playing: false }, objects, onChange: vi.fn(), onRecord };
    const { rerender } = render(<MorphPanel {...props} status={{ phase: "loading" }} />);
    expect(screen.getByRole("button", { name: "Videoyu kaydet" })).toBeDisabled();
    rerender(<MorphPanel {...props} status={{ phase: "error" }} />);
    expect(screen.getByRole("button", { name: "Videoyu kaydet" })).toBeDisabled();
    rerender(<MorphPanel {...props} status={{ phase: "ready" }} morph={{ ...prepared, enabled: false }} />);
    expect(screen.getByRole("button", { name: "Videoyu kaydet" })).toBeDisabled();
    rerender(<MorphPanel {...props} status={{ phase: "ready" }} morph={{ ...prepared, targetId: "a" }} />);
    expect(screen.getByRole("button", { name: "Videoyu kaydet" })).toBeDisabled();
    rerender(<MorphPanel {...props} status={{ phase: "ready" }} onRecord={undefined} />);
    expect(screen.getByRole("button", { name: "Videoyu kaydet" })).toBeDisabled();
    rerender(<MorphPanel {...props} status={{ phase: "ready" }} />);
    fireEvent.click(screen.getByRole("button", { name: "Videoyu kaydet" }));
    expect(onRecord).toHaveBeenCalledOnce();
  });

  it("locks all morph controls, discards pending numeric edits, and keeps cancel available", () => {
    const onChange = vi.fn();
    const onRecord = vi.fn();
    const onCancelRecording = vi.fn();
    const props = { morph: prepared, playback: { t: 0, playing: false }, objects, onChange, onRecord, onCancelRecording };
    const { rerender } = render(<MorphPanel {...props} status={{ phase: "ready" }} />);
    const duration = screen.getByLabelText("Süre (sn)");
    fireEvent.change(duration, { target: { value: "12" } });

    rerender(<MorphPanel {...props} status={{ phase: "ready" }} recording={{ progress: 0.375 }} />);
    for (const input of [...screen.getAllByRole("slider"), ...screen.getAllByRole("spinbutton")]) {
      expect(input).toBeDisabled();
    }
    for (const button of screen.getAllByRole("button")) {
      if (button.textContent !== "İptal") expect(button).toBeDisabled();
    }
    expect(screen.getByRole("button", { name: "Kaydediliyor… %38" })).toBeDisabled();
    fireEvent.blur(duration);
    expect(duration).toHaveValue(prepared.duration);
    fireEvent.change(screen.getByLabelText("Zaman çizelgesi"), { target: { value: "0.9" } });
    fireEvent.click(screen.getByRole("button", { name: "Önizlemeyi kapat" }));
    expect(onChange).not.toHaveBeenCalled();
    expect(onRecord).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "İptal" }));
    expect(onCancelRecording).toHaveBeenCalledOnce();
  });

  it("unlocks the controls and shows the restored playback after recording", () => {
    const morph = { ...prepared, t: 0.43, playing: true };
    const props = { morph, playback: { t: 0.43, playing: true }, objects, onChange: vi.fn(), onRecord: vi.fn(), onCancelRecording: vi.fn() };
    const { rerender } = render(<MorphPanel {...props} status={{ phase: "ready" }} recording={{ progress: 1 }} />);
    rerender(<MorphPanel {...props} status={{ phase: "ready" }} recording={null} />);
    expect(screen.queryByRole("button", { name: "İptal" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Videoyu kaydet" })).not.toBeDisabled();
    expect(screen.getByRole("button", { name: "Duraklat" })).not.toBeDisabled();
    expect(screen.getByRole("button", { name: "Önizlemeyi kapat" })).not.toBeDisabled();
    for (const input of [...screen.getAllByRole("slider"), ...screen.getAllByRole("spinbutton")]) {
      expect(input).not.toBeDisabled();
    }
    expect(screen.getByLabelText("Morph ilerleme")).toHaveTextContent("43%");
    expect(screen.getByLabelText("Süre (sn)")).toHaveValue(morph.duration);
    expect(screen.getByLabelText("Seed")).toHaveValue(morph.seed);
    fireEvent.click(screen.getByRole("button", { name: "Duraklat" }));
    expect(props.onChange).toHaveBeenCalledWith({ playing: false, t: 0.43 });
  });

  it("announces recording errors while leaving retry available", () => {
    render(<MorphPanel morph={prepared} playback={{ t: 0, playing: false }} status={{ phase: "ready" }} objects={objects} onChange={vi.fn()} onRecord={vi.fn()} recordingError="WebM kaydı desteklenmiyor." />);
    expect(screen.getByRole("alert")).toHaveTextContent("WebM kaydı desteklenmiyor.");
    expect(screen.getByRole("button", { name: "Videoyu kaydet" })).not.toBeDisabled();
  });

  it("picks A/B without moving the inspector selection and excludes mesh objects", () => {
    const onSelect = vi.fn();
    const onMorphSelect = vi.fn();
    render(<ObjectListPanel
      objects={[...objects, { ...objects[0], id: "mesh", name: "mesh", kind: "mesh" }]}
      selectedId="b"
      errors={{}}
      onSelect={onSelect}
      onToggleVisible={vi.fn()}
      onDuplicate={vi.fn()}
      onRemove={vi.fn()}
      morphSourceId="a"
      onMorphSelect={onMorphSelect}
    />);
    expect(screen.getByRole("button", { name: "splat a Morph A" })).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(screen.getByRole("button", { name: "splat b Morph B" }));
    expect(onMorphSelect).toHaveBeenCalledWith("targetId", "b");
    expect(onSelect).not.toHaveBeenCalled();
    expect(screen.queryByRole("button", { name: "mesh Morph A" })).not.toBeInTheDocument();
  });

  it("moves a repeated object to the new slot, toggles a slot off, and resets playback", () => {
    const moved = selectMorphSlot({ ...prepared, playing: true, t: 0.6 }, "sourceId", "b");
    expect(moved).toMatchObject({ sourceId: "b", targetId: null, enabled: false, playing: false, t: 0 });
    expect(selectMorphSlot(moved, "sourceId", "b").sourceId).toBeNull();
  });

  it("locks pair selection, inspector selection, and object actions during recording", () => {
    const props = {
      objects,
      selectedId: "a",
      errors: {},
      onSelect: vi.fn(),
      onToggleVisible: vi.fn(),
      onDuplicate: vi.fn(),
      onRemove: vi.fn(),
      onMorphSelect: vi.fn(),
    };
    const { rerender } = render(<ObjectListPanel {...props} locked />);
    for (const button of screen.getAllByRole("button")) {
      expect(button).toBeDisabled();
      fireEvent.click(button);
    }
    for (const item of screen.getAllByRole("listitem")) {
      expect(item).toHaveAttribute("aria-disabled", "true");
      expect(item).toHaveAttribute("tabindex", "-1");
      fireEvent.click(item);
      fireEvent.keyDown(item, { key: "Enter" });
    }
    for (const callback of [props.onSelect, props.onToggleVisible, props.onDuplicate, props.onRemove, props.onMorphSelect]) {
      expect(callback).not.toHaveBeenCalled();
    }
    rerender(<ObjectListPanel {...props} locked={false} />);
    for (const button of screen.getAllByRole("button")) expect(button).not.toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "splat b Morph B" }));
    expect(props.onMorphSelect).toHaveBeenCalledWith("targetId", "b");
    fireEvent.click(screen.getAllByRole("listitem")[1]);
    expect(props.onSelect).toHaveBeenCalledWith("b");
  });
});
