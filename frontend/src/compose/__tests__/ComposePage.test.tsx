import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Job } from "../../api";
import type { AssetOrientation, SceneDoc } from "../types";
import type { MorphPlayback, MorphState, MorphStatus } from "../morphTypes";
import type { MorphRecordOptions, MorphVideoRecorder, OnMorphRecorder } from "../morphCapture";
import { INITIAL_MORPH_STATE } from "../morphTypes";
import { pickMorphSettings } from "../morphSettings";

interface ViewportProps {
  morph: MorphState;
  morphPlayback: MorphPlayback;
  onMorphChange: (patch: Partial<MorphState>) => void;
  onMorphStatus: (status: MorphStatus) => void;
  onMorphRecorder: OnMorphRecorder;
  recording: boolean;
  doc: SceneDoc;
}
const viewport = vi.hoisted(() => ({ props: null as ViewportProps | null }));

vi.mock("../../api", () => ({ getJobStatus: vi.fn() }));
vi.mock("../composeApi", () => ({
  assetFileUrl: (id: string) => `/assets/${id}`,
  backendUrl: (path: string) => `http://backend${path}`,
  createScene: vi.fn(),
  exportScene: vi.fn(),
  getAssetOrientation: vi.fn(),
  getScene: vi.fn(),
  listAssets: vi.fn(),
  listScenes: vi.fn(),
  saveScene: vi.fn(),
  uploadAsset: vi.fn(),
}));
vi.mock("../ComposeViewport", () => ({ ComposeViewport: (props: ViewportProps) => {
  viewport.props = props;
  return <div data-testid="viewport" />;
} }));

import { getJobStatus } from "../../api";
import * as composeApi from "../composeApi";
import { ComposePage } from "../ComposePage";

const api = vi.mocked(composeApi);
const jobStatus = vi.mocked(getJobStatus);

const DOC: SceneDoc = {
  version: 1,
  id: "s1",
  name: "Bahçe",
  viewUp: "y",
  objects: [
    {
      id: "base",
      kind: "splat",
      asset: "a1",
      name: "garden",
      role: "base",
      visible: true,
      transform: { position: [0, 0, 0], quaternion: [0, 0, 0, 1], scale: 1 },
    },
  ],
};

const STATUE = {
  id: "o_statue",
  kind: "splat" as const,
  asset: "a_statue",
  name: "statue",
  role: "object" as const,
  visible: true,
  transform: { position: [1, 2, 3] as [number, number, number], quaternion: [0, 0, 0, 1] as [number, number, number, number], scale: 2 },
};

function orientation(patch: Partial<AssetOrientation>): AssetOrientation {
  return { up: [0, 0, 1], tilt_deg: 90, plane_inlier_frac: 0.5, above_below_ratio: 4, measured: true, ...patch };
}

function job(patch: Partial<Job>): Job {
  return {
    id: "j1",
    scene: "compose-s1",
    status: "running",
    smoke_test: false,
    phase: { name: "compose", progress: 0.5, message: "birleştiriliyor", details: {} },
    overall_progress: 0.5,
    created_at: 0,
    started_at: 0,
    finished_at: null,
    error: null,
    ply_dir: null,
    download_url: null,
    ...patch,
  };
}

/** Let pending promises (and zero-delay timers) settle inside act. */
async function flush() {
  await act(async () => {
    for (let i = 0; i < 5; i++) await Promise.resolve();
  });
}

async function openScene() {
  render(<ComposePage active />);
  fireEvent.click(await screen.findByRole("button", { name: "Aç" }));
  await screen.findByTestId("viewport");
}

async function openRecordingScene() {
  api.getScene.mockResolvedValue({ ...DOC, objects: [...DOC.objects, STATUE] });
  const view = render(<ComposePage active />);
  fireEvent.click(await screen.findByRole("button", { name: "Aç" }));
  await screen.findByTestId("viewport");
  fireEvent.click(screen.getByRole("button", { name: "garden Morph A" }));
  fireEvent.click(screen.getByRole("button", { name: "statue Morph B" }));
  fireEvent.click(screen.getByRole("button", { name: "Morph hazırla" }));
  act(() => viewport.props!.onMorphStatus({ phase: "ready", count: 200000 }));
  fireEvent.click(screen.getByRole("button", { name: "Oynat" }));
  viewport.props!.morphPlayback.t = 0.625;
  return view;
}

/** The viewport owns capture; resolve it manually to cover late completion after abort. */
function registerPendingRecorder() {
  let options!: MorphRecordOptions;
  let finish!: (blob: Blob | null) => void;
  let fail!: (error: Error) => void;
  const record = vi.fn<MorphVideoRecorder>((next) => {
    options = next;
    return new Promise((resolve, reject) => { finish = resolve; fail = reject; });
  });
  act(() => viewport.props!.onMorphRecorder(record));
  return { record, get options() { return options; }, finish: (blob: Blob | null) => finish(blob), fail: (error: Error) => fail(error) };
}

function mockVideoDownload() {
  const createObjectURL = vi.fn(() => "blob:recorded-morph");
  const revokeObjectURL = vi.fn();
  vi.stubGlobal("URL", class extends URL {
    static createObjectURL = createObjectURL;
    static revokeObjectURL = revokeObjectURL;
  });
  const downloads: { href: string; filename: string }[] = [];
  const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
    downloads.push({ href: this.href, filename: this.download });
  });
  return { createObjectURL, revokeObjectURL, click, downloads };
}

function makeDirty() {
  fireEvent.change(screen.getByDisplayValue("+Y"), { target: { value: "-y" } });
  expect(screen.getByText(/Bahçe •/)).toBeInTheDocument();
}

beforeEach(() => {
  vi.clearAllMocks();
  viewport.props = null;
  api.listScenes.mockResolvedValue([{ id: "s1", name: "Bahçe", updated_ts: 0, object_count: 1 }]);
  api.listAssets.mockResolvedValue([]);
  api.getScene.mockResolvedValue(DOC);
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("ComposePage", () => {
  it("saves every morph setting, restores it on open, and keeps playback transient", async () => {
    await openRecordingScene();
    api.saveScene.mockImplementation(async (saved) => saved);
    act(() => viewport.props!.onMorphChange({
      mode: "cloud", duration: 9.5, dissolve: 0.3, wave: 0.4, arc: 0.2,
      targetBlend: 0.8, seed: 1234, autoAlign: true,
    }));
    const expected = pickMorphSettings(viewport.props!.morph);
    fireEvent.click(screen.getByRole("button", { name: "Kaydet" }));
    await screen.findByText("Kaydedildi");
    const saved = api.saveScene.mock.calls[0][0];
    expect(saved.morph).toEqual(expected);
    expect(Object.keys(saved.morph!)).toHaveLength(10);
    expect(saved.morph).not.toHaveProperty("playing");
    act(() => viewport.props!.onMorphStatus({ phase: "ready" }));
    act(() => viewport.props!.onMorphChange({ t: 0.7, playing: true }));
    expect(screen.getByRole("button", { name: "Kaydet" })).toBeDisabled();
    api.getScene.mockResolvedValue(saved);
    fireEvent.click(screen.getByRole("button", { name: "← Sahneler" }));
    fireEvent.click(await screen.findByRole("button", { name: "Aç" }));
    await screen.findByTestId("viewport");
    expect(viewport.props!.morph).toEqual({ ...INITIAL_MORPH_STATE, ...expected });
    expect(viewport.props!.morphPlayback).toEqual({ t: 0, playing: false });
    expect(screen.getByRole("button", { name: "Kaydet" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Morph hazırla" })).toBeEnabled();
  });

  it("exports the saved morph metadata and leaves later edits dirty during a pending save", async () => {
    await openRecordingScene();
    let finish!: (doc: SceneDoc) => void;
    api.saveScene.mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
    api.exportScene.mockResolvedValue({ job_id: "j1" });
    jobStatus.mockResolvedValue(job({ status: "completed", overall_progress: 1 }));
    fireEvent.click(screen.getByRole("button", { name: "Export" }));
    expect(api.saveScene).toHaveBeenCalledOnce();
    const snapshot = api.saveScene.mock.calls[0][0];
    expect(snapshot.morph).toMatchObject({ sourceId: "base", targetId: "o_statue", duration: 6 });
    act(() => viewport.props!.onMorphChange({ duration: 12 }));
    await act(async () => finish(snapshot));
    expect(api.exportScene).toHaveBeenCalledWith("s1");
    expect(viewport.props!.doc.morph!.duration).toBe(12);
    expect(screen.getByRole("button", { name: "Kaydet" })).toBeEnabled();
  });

  it("toggles Space only in the focused viewport and ignores repeats, modifiers and preparation", async () => {
    await openRecordingScene();
    const region = screen.getByRole("region", { name: "Sahne görünümü" });
    const duration = screen.getByLabelText("Süre (sn)");
    duration.focus();
    fireEvent.keyDown(duration, { key: " " });
    fireEvent.keyDown(window, { key: " " });
    expect(viewport.props!.morph.playing).toBe(true);
    fireEvent.pointerDown(screen.getByTestId("viewport"));
    expect(region).toHaveFocus();
    fireEvent.keyDown(region, { key: " ", repeat: true });
    fireEvent.keyDown(region, { key: " ", ctrlKey: true });
    expect(viewport.props!.morph.playing).toBe(true);
    expect(fireEvent.keyDown(region, { key: " " })).toBe(false);
    expect(viewport.props!.morph).toMatchObject({ playing: false, t: 0.625 });
    fireEvent.keyDown(region, { key: " " });
    expect(viewport.props!.morph.playing).toBe(true);
    act(() => viewport.props!.onMorphStatus({ phase: "loading" }));
    fireEvent.keyDown(region, { key: " " });
    expect(viewport.props!.morph.playing).toBe(true);
    act(() => viewport.props!.onMorphChange({ playing: false, t: 1 }));
    act(() => viewport.props!.onMorphStatus({ phase: "ready" }));
    fireEvent.keyDown(region, { key: " " });
    expect(viewport.props!.morph).toMatchObject({ playing: true, t: 0 });
  });

  it("invalidates recording readiness immediately when auto alignment changes", async () => {
    await openRecordingScene();
    const capture = registerPendingRecorder();
    expect(screen.getByRole("button", { name: "Videoyu kaydet" })).toBeEnabled();
    fireEvent.click(screen.getByRole("checkbox", { name: "Otomatik hizala" }));
    expect(viewport.props!.morph).toMatchObject({ autoAlign: true, playing: false, t: 0 });
    const record = screen.getByRole("button", { name: "Videoyu kaydet" });
    expect(record).toBeDisabled();
    fireEvent.click(record);
    expect(capture.record).not.toHaveBeenCalled();
  });

  it("invalidates ready geometry before the renderer commits and ignores stale worker status", async () => {
    await openRecordingScene();
    const capture = registerPendingRecorder();
    const oldStatus = viewport.props!.onMorphStatus;
    fireEvent.click(screen.getByText("statue", { selector: ".compose-object-name" }));
    const x = screen.getByLabelText("Konum X");
    fireEvent.change(x, { target: { value: "5" } });
    fireEvent.keyDown(x, { key: "Enter" });
    expect(screen.getByRole("button", { name: "Videoyu kaydet" })).toBeDisabled();
    act(() => oldStatus({ phase: "ready", count: 200000 }));
    const button = screen.getByRole("button", { name: "Videoyu kaydet" });
    expect(button).toBeDisabled();
    fireEvent.click(button);
    expect(capture.record).not.toHaveBeenCalled();
    act(() => viewport.props!.onMorphStatus({ phase: "ready", count: 200000 }));
    expect(button).toBeEnabled();
  });

  it("records the configured duration, shows progress, downloads the seed name, and restores live playback", async () => {
    await openRecordingScene();
    expect(screen.getByRole("button", { name: "Videoyu kaydet" })).toBeDisabled();
    const capture = registerPendingRecorder();
    const download = mockVideoDownload();
    vi.useFakeTimers();
    fireEvent.click(screen.getByRole("button", { name: "Videoyu kaydet" }));
    expect(capture.record).toHaveBeenCalledOnce();
    expect(capture.options.duration).toBe(6);
    expect(capture.options.signal.aborted).toBe(false);
    expect(viewport.props?.recording).toBe(true);
    expect(viewport.props?.morph).toMatchObject({ t: 0.625, playing: false });
    act(() => {
      viewport.props!.morphPlayback.t = 0.9;
      capture.options.onProgress(0.375);
    });
    expect(screen.getByRole("button", { name: "Kaydediliyor… %38" })).toBeDisabled();

    const blob = new Blob(["video"], { type: "video/webm" });
    await act(async () => capture.finish(blob));
    expect(download.createObjectURL).toHaveBeenCalledWith(blob);
    expect(download.downloads).toEqual([{ href: "blob:recorded-morph", filename: "splat-morph-42.webm" }]);
    expect(viewport.props?.recording).toBe(false);
    expect(viewport.props?.morph).toMatchObject({ t: 0.625, playing: true });
    expect(viewport.props?.morphPlayback).toEqual({ t: 0.625, playing: true });
    expect(screen.getByRole("button", { name: "Videoyu kaydet" })).not.toBeDisabled();
    expect(screen.getByRole("button", { name: "Duraklat" })).not.toBeDisabled();
    expect(document.querySelector("a[download]")).toBeNull();
    await act(async () => vi.advanceTimersByTimeAsync(1000));
    expect(download.revokeObjectURL).toHaveBeenCalledWith("blob:recorded-morph");
  }, 15_000); // Includes opening/preparing the editor and the full download lifecycle.

  it("locks scene, pair, and inspector controls and shortcuts until recording cancellation finishes", async () => {
    await openRecordingScene();
    fireEvent.click(screen.getByText("statue", { selector: ".compose-object-name" }));
    makeDirty();
    const before = viewport.props!.doc;
    const capture = registerPendingRecorder();
    const download = mockVideoDownload();
    fireEvent.click(screen.getByRole("button", { name: "Videoyu kaydet" }));

    for (const name of ["← Sahneler", "Kaydet", "Export", "Taşı (W)", "Döndür (E)", "Ölçekle (R)", "garden Morph A", "statue Morph B", "statue gizle", "statue kopyala", "statue sil"]) {
      expect(screen.getByRole("button", { name })).toBeDisabled();
      fireEvent.click(screen.getByRole("button", { name }));
    }
    expect(screen.getByDisplayValue("−Y (COLMAP)")).toBeDisabled();
    expect(screen.getByDisplayValue("statue")).toBeDisabled();
    expect(screen.getByLabelText("Konum X")).toBeDisabled();
    expect(screen.getByLabelText("Süre (sn)")).toBeDisabled();
    fireEvent.keyDown(window, { key: "Delete" });
    fireEvent.keyDown(window, { key: "d", ctrlKey: true });
    fireEvent.keyDown(window, { key: "s", ctrlKey: true });
    fireEvent.keyDown(window, { key: "e" });
    expect(viewport.props!.doc).toBe(before);
    expect(viewport.props?.morph).toMatchObject({ sourceId: "base", targetId: "o_statue" });
    expect(screen.getByRole("button", { name: "Taşı (W)" })).toHaveAttribute("aria-pressed", "true");
    expect(api.saveScene).not.toHaveBeenCalled();
    expect(api.exportScene).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "İptal" }));
    expect(capture.options.signal.aborted).toBe(true);
    expect(screen.getByRole("button", { name: "statue Morph B" })).toBeDisabled();
    // A recorder may return buffered data after stop; cancellation must suppress it.
    await act(async () => capture.finish(new Blob(["partial video"])));
    expect(download.createObjectURL).not.toHaveBeenCalled();
    expect(download.click).not.toHaveBeenCalled();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(viewport.props?.morphPlayback).toEqual({ t: 0.625, playing: true });
    expect(screen.getByRole("button", { name: "statue Morph B" })).not.toBeDisabled();
    expect(screen.getByLabelText("Konum X")).not.toBeDisabled();
    expect(screen.getByRole("button", { name: "Kaydet" })).not.toBeDisabled();
  });

  it("reports capture failure and restores the previous paused time without a download", async () => {
    await openRecordingScene();
    fireEvent.click(screen.getByRole("button", { name: "Duraklat" }));
    const capture = registerPendingRecorder();
    const download = mockVideoDownload();
    fireEvent.click(screen.getByRole("button", { name: "Videoyu kaydet" }));
    viewport.props!.morphPlayback.t = 1;
    await act(async () => capture.fail(new Error("WebM codec failed")));
    expect(screen.getByRole("alert")).toHaveTextContent("WebM codec failed");
    expect(viewport.props?.morph).toMatchObject({ t: 0.625, playing: false });
    expect(viewport.props?.morphPlayback).toEqual({ t: 0.625, playing: false });
    expect(screen.getByRole("button", { name: "Videoyu kaydet" })).not.toBeDisabled();
    expect(download.createObjectURL).not.toHaveBeenCalled();
  });

  it("cancels recording when inactive and restores the previous time without resuming playback", async () => {
    const view = await openRecordingScene();
    const capture = registerPendingRecorder();
    const download = mockVideoDownload();
    fireEvent.click(screen.getByRole("button", { name: "Videoyu kaydet" }));
    viewport.props!.morphPlayback.t = 0.9;
    view.rerender(<ComposePage active={false} />);
    expect(capture.options.signal.aborted).toBe(true);
    await act(async () => capture.finish(new Blob(["partial video"])));
    expect(viewport.props?.recording).toBe(false);
    expect(viewport.props?.morph).toMatchObject({ t: 0.625, playing: false });
    expect(viewport.props?.morphPlayback).toEqual({ t: 0.625, playing: false });
    expect(download.createObjectURL).not.toHaveBeenCalled();
    view.rerender(<ComposePage active />);
    expect(screen.getByRole("button", { name: "Oynat" })).not.toBeDisabled();
    expect(screen.getByRole("button", { name: "Videoyu kaydet" })).not.toBeDisabled();
  });

  it("aborts on unmount and ignores late progress and recorded data", async () => {
    const view = await openRecordingScene();
    const capture = registerPendingRecorder();
    const download = mockVideoDownload();
    fireEvent.click(screen.getByRole("button", { name: "Videoyu kaydet" }));
    view.unmount();
    expect(capture.options.signal.aborted).toBe(true);
    await act(async () => {
      capture.options.onProgress(1);
      capture.finish(new Blob(["late video"]));
    });
    expect(download.createObjectURL).not.toHaveBeenCalled();
    expect(download.click).not.toHaveBeenCalled();
    expect(screen.queryByTestId("viewport")).not.toBeInTheDocument();
  });

  it("persists morph selection and clears the saved pair when an endpoint is deleted", async () => {
    api.getScene.mockResolvedValue({ ...DOC, objects: [...DOC.objects, STATUE] });
    await openScene();
    fireEvent.click(screen.getByRole("button", { name: "garden Morph A" }));
    fireEvent.click(screen.getByRole("button", { name: "statue Morph B" }));
    fireEvent.click(screen.getByRole("button", { name: "Morph hazırla" }));
    expect(viewport.props?.morph).toMatchObject({ sourceId: "base", targetId: "o_statue", enabled: true });
    expect(screen.getByRole("button", { name: "Kaydet" })).not.toBeDisabled();
    expect(viewport.props?.doc.morph).toMatchObject({ sourceId: "base", targetId: "o_statue" });
    act(() => viewport.props?.onMorphStatus({ phase: "ready", count: 200000 }));
    fireEvent.click(screen.getByRole("button", { name: "Oynat" }));
    expect(viewport.props?.morph.playing).toBe(true);
    fireEvent.click(screen.getByText("statue", { selector: ".compose-object-name" }));
    fireEvent.keyDown(window, { key: "Delete" });
    expect(viewport.props?.morph).toMatchObject({ sourceId: null, targetId: null, enabled: false, playing: false, t: 0 });
    expect(viewport.props?.doc.morph).toBeNull();
    expect(viewport.props?.morphPlayback).toEqual({ t: 0, playing: false });
  });

  it("pauses morph at the live time when the editor becomes inactive and clears it on scene close", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    api.getScene.mockResolvedValue({ ...DOC, objects: [...DOC.objects, STATUE] });
    const { rerender } = render(<ComposePage active />);
    fireEvent.click(await screen.findByRole("button", { name: "Aç" }));
    await screen.findByTestId("viewport");
    fireEvent.click(screen.getByRole("button", { name: "garden Morph A" }));
    fireEvent.click(screen.getByRole("button", { name: "statue Morph B" }));
    fireEvent.click(screen.getByRole("button", { name: "Morph hazırla" }));
    act(() => viewport.props?.onMorphStatus({ phase: "ready" }));
    fireEvent.click(screen.getByRole("button", { name: "Oynat" }));
    viewport.props!.morphPlayback.t = 0.625;
    rerender(<ComposePage active={false} />);
    expect(viewport.props?.morph).toMatchObject({ playing: false, t: 0.625 });
    expect(viewport.props?.morphPlayback.playing).toBe(false);
    rerender(<ComposePage active />);
    fireEvent.click(screen.getByRole("button", { name: "← Sahneler" }));
    fireEvent.click(await screen.findByRole("button", { name: "Aç" }));
    await screen.findByTestId("viewport");
    expect(viewport.props?.morph).toMatchObject({ sourceId: null, targetId: null, enabled: false, playing: false, t: 0 });
  });

  it("saves once on Ctrl+S even when pressed twice quickly", async () => {
    let resolveSave: (doc: SceneDoc) => void = () => {};
    api.saveScene.mockImplementation(() => new Promise((r) => (resolveSave = r)));
    await openScene();
    makeDirty();

    fireEvent.keyDown(window, { key: "s", ctrlKey: true });
    fireEvent.keyDown(window, { key: "s", ctrlKey: true });
    await waitFor(() => expect(api.saveScene).toHaveBeenCalledTimes(1));
    expect(api.saveScene.mock.calls[0][0].viewUp).toBe("-y");

    await act(async () => resolveSave(DOC));
    expect(await screen.findByText("Kaydedildi")).toBeInTheDocument();
    expect(screen.queryByText(/Bahçe •/)).not.toBeInTheDocument();

    // Not dirty any more: Ctrl+S is a no-op.
    fireEvent.keyDown(window, { key: "s", ctrlKey: true });
    await new Promise((r) => setTimeout(r, 20));
    expect(api.saveScene).toHaveBeenCalledTimes(1);
  });

  it("keeps polling the export job after a failed poll and then shows the download link", async () => {
    await openScene();
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    api.exportScene.mockResolvedValue({ job_id: "j1" });
    jobStatus
      .mockResolvedValueOnce(job({}))
      .mockRejectedValueOnce(new Error("Failed to fetch"))
      .mockResolvedValueOnce(job({ status: "completed", overall_progress: 1, download_url: "/download/j1" }));

    fireEvent.click(screen.getByRole("button", { name: "Export" }));
    await flush();
    expect(screen.getByText(/50% · birleştiriliyor/)).toBeInTheDocument();

    // First poll fails…
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000);
    });
    expect(jobStatus).toHaveBeenCalledTimes(2);
    // …then it retries (with backoff) instead of stopping.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000);
    });
    expect(jobStatus).toHaveBeenCalledTimes(3);
    const link = screen.getByRole("link", { name: ".zip indir" });
    expect(link).toHaveAttribute("href", "http://backend/download/j1");
    expect(screen.getByRole("button", { name: "Export" })).not.toBeDisabled();
  });

  it("marks the export failed after repeated poll failures and re-enables Export", async () => {
    await openScene();
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    api.exportScene.mockResolvedValue({ job_id: "j1" });
    jobStatus.mockResolvedValueOnce(job({})).mockRejectedValue(new Error("Failed to fetch"));

    fireEvent.click(screen.getByRole("button", { name: "Export" }));
    await flush();
    // Each retry is re-armed after a render, so advance one backoff step per act.
    for (let i = 0; i < 6; i++) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(20_000);
      });
    }
    expect(jobStatus).toHaveBeenCalledTimes(6);
    expect(screen.getByText(/Export başarısız: .*Failed to fetch/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Export" })).not.toBeDisabled();
  });

  it("asks for confirmation before closing a dirty scene", async () => {
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    await openScene();
    makeDirty();

    fireEvent.click(screen.getByRole("button", { name: "← Sahneler" }));
    expect(confirm).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId("viewport")).toBeInTheDocument();

    confirm.mockReturnValue(true);
    fireEvent.click(screen.getByRole("button", { name: "← Sahneler" }));
    expect(confirm).toHaveBeenCalledTimes(2);
    expect(await screen.findByRole("heading", { name: "Sahneler" })).toBeInTheDocument();
  });

  it("'Otomatik (zemin)' uses the base asset's measured floor up", async () => {
    api.getAssetOrientation.mockResolvedValue(orientation({ up: [0, 0, 2] }));
    api.saveScene.mockImplementation(async (d) => d);
    await openScene();

    fireEvent.change(screen.getByDisplayValue("+Y"), { target: { value: "auto" } });
    expect(await screen.findByDisplayValue("Otomatik (zemin)")).toBeInTheDocument();
    expect(api.getAssetOrientation).toHaveBeenCalledWith("a1");
    expect(screen.getByText(/Bahçe •/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Kaydet" }));
    await waitFor(() => expect(api.saveScene).toHaveBeenCalledTimes(1));
    expect(api.saveScene.mock.calls[0][0].up).toEqual([0, 0, 1]);

    // Back to a fixed axis clears the vector.
    fireEvent.change(screen.getByDisplayValue("Otomatik (zemin)"), { target: { value: "-y" } });
    expect(screen.getByDisplayValue("−Y (COLMAP)")).toBeInTheDocument();
  });

  it("'Otomatik (zemin)' reports an unmeasured floor and keeps ±Y", async () => {
    api.getAssetOrientation.mockResolvedValue(orientation({ up: [0, 1, 0], measured: false }));
    await openScene();

    fireEvent.change(screen.getByDisplayValue("+Y"), { target: { value: "auto" } });
    expect(await screen.findByText("Zemin düzlemi bulunamadı; +Y/−Y seçin")).toBeInTheDocument();
    expect(screen.getByDisplayValue("+Y")).toBeInTheDocument();
    expect(screen.queryByText(/Bahçe •/)).not.toBeInTheDocument();
  });

  it("Dikleştir rotates a splat object so its measured up matches the scene up", async () => {
    api.getScene.mockResolvedValue({ ...DOC, up: [0, 1, 0], objects: [...DOC.objects, STATUE] });
    api.getAssetOrientation.mockResolvedValue(orientation({ up: [0, 0, 1] }));
    api.saveScene.mockImplementation(async (d) => d);
    await openScene();
    fireEvent.click(screen.getByText("statue"));

    fireEvent.click(screen.getByRole("button", { name: "Dikleştir" }));
    await waitFor(() => expect(screen.getByText(/Bahçe •/)).toBeInTheDocument());
    expect(api.getAssetOrientation).toHaveBeenCalledWith("a_statue");

    fireEvent.click(screen.getByRole("button", { name: "Kaydet" }));
    await waitFor(() => expect(api.saveScene).toHaveBeenCalledTimes(1));
    const saved = api.saveScene.mock.calls[0][0].objects.find((o) => o.id === "o_statue")!;
    expect(saved.transform.position).toEqual([1, 2, 3]);
    expect(saved.transform.scale).toBe(2);
    // +Z (object up) → +Y (scene up): −90° about X.
    const [x, y, z, w] = saved.transform.quaternion;
    expect(x).toBeCloseTo(-Math.SQRT1_2, 6);
    expect(y).toBeCloseTo(0, 6);
    expect(z).toBeCloseTo(0, 6);
    expect(w).toBeCloseTo(Math.SQRT1_2, 6);
  });

  it("Dikleştir reports an object without a floor plane", async () => {
    api.getScene.mockResolvedValue({ ...DOC, objects: [...DOC.objects, STATUE] });
    api.getAssetOrientation.mockResolvedValue(orientation({ measured: false }));
    await openScene();
    fireEvent.click(screen.getByText("statue"));
    fireEvent.click(screen.getByRole("button", { name: "Dikleştir" }));
    expect(await screen.findByText("Bu objede zemin düzlemi bulunamadı")).toBeInTheDocument();
    expect(screen.queryByText(/Bahçe •/)).not.toBeInTheDocument();
  });

  it("Dikleştir applies to the latest transform when the object moved during the request", async () => {
    let resolve: (o: AssetOrientation) => void = () => {};
    api.getScene.mockResolvedValue({ ...DOC, up: [0, 1, 0], objects: [...DOC.objects, STATUE] });
    api.getAssetOrientation.mockImplementation(() => new Promise((r) => (resolve = r)));
    api.saveScene.mockImplementation(async (d) => d);
    await openScene();
    fireEvent.click(screen.getByText("statue"));

    fireEvent.click(screen.getByRole("button", { name: "Dikleştir" }));
    // Busy: the button is disabled while the request runs.
    expect(screen.getByRole("button", { name: "Dikleştir" })).toBeDisabled();
    // The user moves the object meanwhile.
    const x = screen.getByLabelText("Konum X");
    fireEvent.change(x, { target: { value: "5" } });
    fireEvent.keyDown(x, { key: "Enter" });

    await act(async () => resolve(orientation({ up: [0, 0, 1] })));
    await waitFor(() => expect(screen.getByRole("button", { name: "Dikleştir" })).not.toBeDisabled());
    expect(screen.queryByText("Obje yönü hesaplanıyor…")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Kaydet" }));
    await waitFor(() => expect(api.saveScene).toHaveBeenCalledTimes(1));
    const saved = api.saveScene.mock.calls[0][0].objects.find((o) => o.id === "o_statue")!;
    // Both edits kept: the move and the straightening (no registry bounds → pivot = origin).
    expect(saved.transform.position).toEqual([5, 2, 3]);
    expect(saved.transform.quaternion[0]).toBeCloseTo(-Math.SQRT1_2, 6);
    expect(saved.transform.quaternion[3]).toBeCloseTo(Math.SQRT1_2, 6);
  });

  it("Delete removes the selected object but Backspace does not (no undo yet)", async () => {
    api.getScene.mockResolvedValue({ ...DOC, objects: [...DOC.objects, STATUE] });
    await openScene();
    fireEvent.click(screen.getByText("statue"));

    fireEvent.keyDown(window, { key: "Backspace" });
    expect(screen.getByText("statue")).toBeInTheDocument();

    fireEvent.keyDown(window, { key: "Delete" });
    expect(screen.queryByText("statue")).not.toBeInTheDocument();
  });

  it("Dikleştir clears its status when the object was deleted meanwhile", async () => {
    let resolve: (o: AssetOrientation) => void = () => {};
    api.getScene.mockResolvedValue({ ...DOC, objects: [...DOC.objects, STATUE] });
    api.getAssetOrientation.mockImplementation(() => new Promise((r) => (resolve = r)));
    await openScene();
    fireEvent.click(screen.getByText("statue"));
    fireEvent.click(screen.getByRole("button", { name: "Dikleştir" }));
    expect(screen.getByText("Obje yönü hesaplanıyor…")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "statue sil" }));
    await act(async () => resolve(orientation({ up: [0, 0, 1] })));
    await waitFor(() => expect(screen.queryByText("Obje yönü hesaplanıyor…")).not.toBeInTheDocument());
    expect(screen.queryByText("statue")).not.toBeInTheDocument();
  });
});
