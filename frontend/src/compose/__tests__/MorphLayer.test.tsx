import { act, render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ComponentProps } from "react";
import type { PreparedMorph } from "../morphData";
import type { MorphRecordingOptions } from "../morphRecording";
import type { MorphVideoRecorder } from "../morphCapture";
import type { MorphWorkerRequest, MorphWorkerResponse } from "../morph.worker";
import { INITIAL_MORPH_STATE } from "../morphTypes";
import type { SceneObject } from "../types";

const mocks = vi.hoisted(() => ({
  frame: null as ((state: unknown, delta: number) => void) | null,
  scene: { children: [] as unknown[], add: vi.fn() },
  gl: { capabilities: { maxTextureSize: 4096 }, domElement: {} },
  camera: {},
  acquire: vi.fn(),
  release: vi.fn(),
  extract: vi.fn(),
  createGpu: vi.fn(),
  createCapture: vi.fn(),
  nextPaint: vi.fn(),
  recordVideo: vi.fn(),
}));

vi.mock("@react-three/fiber", () => ({
  useFrame: (callback: (state: unknown, delta: number) => void) => { mocks.frame = callback; },
  useThree: (selector: (state: unknown) => unknown) => selector({ scene: mocks.scene, gl: mocks.gl, camera: mocks.camera }),
}));
vi.mock("@sparkjsdev/spark", () => ({ SparkRenderer: class SparkRenderer {} }));
vi.mock("../splatCache", () => ({ packedSplatsCache: { acquire: mocks.acquire, release: mocks.release } }));
vi.mock("../morphSource", () => ({ extractMorphSource: mocks.extract }));
vi.mock("../morphGpu", () => ({ createMorphGpu: mocks.createGpu }));
vi.mock("../morphCapture", () => ({ createMorphCapture: mocks.createCapture, nextMorphPaint: mocks.nextPaint }));
vi.mock("../morphRecording", () => ({ recordMorphVideo: mocks.recordVideo }));

import { MorphLayer } from "../MorphLayer";
import { SparkRenderer } from "@sparkjsdev/spark";

const instances: FakeWorker[] = [];
class FakeWorker {
  onmessage: ((event: MessageEvent<MorphWorkerResponse>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  onmessageerror: (() => void) | null = null;
  postMessage = vi.fn<(request: MorphWorkerRequest, transfer: Transferable[]) => void>();
  terminate = vi.fn();
  constructor() { instances.push(this); }
  reply(data: MorphWorkerResponse) { this.onmessage?.({ data } as MessageEvent<MorphWorkerResponse>); }
}

const data: PreparedMorph = {
  a: new Float32Array(32), b: new Float32Array(32), count: 2, sceneRadius: 1,
  meanTravel: 0.24, matchingTravel: 0.001, alignmentMs: 27,
};
const source: SceneObject = {
  id: "source", kind: "splat", asset: "a", name: "A", role: "object", visible: true,
  transform: { position: [0, 0, 0], quaternion: [0, 0, 0, 1], scale: 1 },
};
const target: SceneObject = { ...source, id: "target", asset: "b", name: "B" };

function props(): ComponentProps<typeof MorphLayer> {
  const playback = { t: 0, playing: false };
  return {
    source, target, sourceUrl: "/a.ply", targetUrl: "/b.ply",
    morph: { ...INITIAL_MORPH_STATE, sourceId: source.id, targetId: target.id, enabled: true },
    playback,
    onChange: vi.fn((patch) => {
      if (patch.t !== undefined) playback.t = patch.t;
      if (patch.playing !== undefined) playback.playing = patch.playing;
    }),
    onStatus: vi.fn(), onDisplay: vi.fn(),
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

async function settle() {
  await act(async () => {
    for (let i = 0; i < 8; i++) await Promise.resolve();
  });
}

async function ready() {
  await settle();
  instances.at(-1)!.reply({ id: 1, ok: true, result: data });
  await settle();
  return mocks.createGpu.mock.results.at(-1)!.value as {
    object: { visible: boolean; generatorError?: Error }; update: ReturnType<typeof vi.fn>; dispose: ReturnType<typeof vi.fn>;
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  instances.length = 0;
  mocks.frame = null;
  mocks.scene.children = [];
  mocks.acquire.mockImplementation(() => ({ initialized: Promise.resolve(), numSplats: 2 }));
  mocks.extract.mockImplementation(async () => new Float32Array(32));
  mocks.createGpu.mockImplementation(() => ({ object: { visible: false }, update: vi.fn(), dispose: vi.fn() }));
  mocks.nextPaint.mockResolvedValue(undefined);
  vi.stubGlobal("Worker", FakeWorker);
});

describe("MorphLayer recording integration", () => {
  function recordingProps() {
    const p = props();
    const onRecorder = vi.fn();
    mocks.scene.children = [new SparkRenderer({ renderer: mocks.gl as never })];
    return { ...p, onRecorder };
  }

  it("rejects recording before preparation and while alignment is rebuilding", async () => {
    let p = recordingProps();
    const { rerender } = render(<MorphLayer {...p} />);
    await settle();
    const recorder = p.onRecorder.mock.calls[0][0] as MorphVideoRecorder;
    const options = { duration: 6, signal: new AbortController().signal, onProgress: vi.fn() };
    await expect(recorder(options)).rejects.toThrow("Morph kayda hazır değil.");
    const resource = await ready();
    p = { ...p, morph: { ...p.morph, autoAlign: true } };
    rerender(<MorphLayer {...p} />);
    await settle();
    expect(resource.dispose).toHaveBeenCalledOnce();
    expect(p.onStatus).toHaveBeenLastCalledWith({ phase: "loading" });
    await expect(recorder(options)).rejects.toThrow("Morph kayda hazır değil.");
    expect(mocks.recordVideo).not.toHaveBeenCalled();
    expect(mocks.createCapture).not.toHaveBeenCalled();
  });

  it.each(["cloud", "shape"] as const)("records %s with current controls before handoff and captures only after rendering", async (mode) => {
    const events: string[] = [];
    const sort = deferred<void>();
    const capture = {
      render: vi.fn(async (signal: AbortSignal, request: () => void, waitForCapture: () => Promise<void> = async () => {}) => {
        events.push("prepare");
        await sort.promise;
        await waitForCapture();
        signal.throwIfAborted();
        events.push("render");
        request();
      }),
      dispose: vi.fn(async () => {}),
    };
    mocks.createCapture.mockReturnValue(capture);
    mocks.nextPaint.mockImplementation(async () => { events.push("paint"); });
    const request = vi.fn(() => { events.push("capture"); });
    const blob = new Blob(["video"]);
    mocks.recordVideo.mockImplementation(async (options: MorphRecordingOptions) => {
      await options.renderFrame(0.5, options.signal, request, async () => { events.push("cadence"); });
      return blob;
    });
    const p = recordingProps();
    p.morph = { ...p.morph, mode, wave: 0.6, arc: 0.4 };
    vi.mocked(p.onDisplay).mockImplementation(() => { events.push("display"); });
    const { unmount } = render(<MorphLayer {...p} />);
    const resource = await ready();
    resource.update.mockImplementation(() => { events.push("update"); });
    p.playback.playing = true;
    const recorder = p.onRecorder.mock.calls[0][0] as MorphVideoRecorder;
    const result = recorder({ duration: 6, signal: new AbortController().signal, onProgress: vi.fn() });
    await settle();
    expect(events).toEqual(["update", "display", "paint", "paint", "prepare"]);
    expect(p.playback).toEqual({ t: 0.5, playing: false });
    expect(resource.object.visible).toBe(true);
    expect(resource.update).toHaveBeenLastCalledWith(0.5, 1, 1, mode, 0.6, 0.4);
    act(() => mocks.frame?.({}, 1));
    expect(resource.update).toHaveBeenCalledTimes(1);
    expect(request).not.toHaveBeenCalled();
    expect(mocks.createCapture).toHaveBeenCalledWith(mocks.scene.children[0], mocks.gl, mocks.scene, mocks.camera);
    sort.resolve();
    await expect(result).resolves.toBe(blob);
    expect(events).toEqual(["update", "display", "paint", "paint", "prepare", "cadence", "render", "capture"]);
    expect(capture.dispose).toHaveBeenCalledTimes(1);
    unmount();
    expect(p.onRecorder).toHaveBeenLastCalledWith(null, recorder);
  });

  it("keeps GPU inputs alive on unmount until the in-flight sort and capture cleanup settle", async () => {
    const sort = deferred<void>();
    const request = vi.fn();
    const capture = {
      render: vi.fn(async (signal: AbortSignal, captureFrame: () => void, waitForCapture: () => Promise<void> = async () => {}) => {
        await sort.promise;
        await waitForCapture();
        signal.throwIfAborted();
        captureFrame();
      }),
      dispose: vi.fn(async () => { await sort.promise; }),
    };
    mocks.createCapture.mockReturnValue(capture);
    mocks.recordVideo.mockImplementation((options: MorphRecordingOptions) => new Promise<Blob | null>((resolve, reject) => {
      options.signal.addEventListener("abort", () => resolve(null), { once: true });
      void options.renderFrame(0.5, options.signal, request, async () => {}).catch((error: unknown) => {
        if (options.signal.aborted) resolve(null);
        else reject(error);
      });
    }));
    const p = recordingProps();
    const { unmount } = render(<MorphLayer {...p} />);
    const resource = await ready();
    const recorder = p.onRecorder.mock.calls[0][0] as MorphVideoRecorder;
    const result = recorder({ duration: 6, signal: new AbortController().signal, onProgress: vi.fn() });
    await settle();
    unmount();
    await settle();
    expect(capture.dispose).toHaveBeenCalledTimes(1);
    expect(resource.dispose).not.toHaveBeenCalled();
    expect(request).not.toHaveBeenCalled();
    sort.resolve();
    await expect(result).resolves.toBeNull();
    await settle();
    expect(resource.dispose).toHaveBeenCalledTimes(1);
    expect(request).not.toHaveBeenCalled();
  });

  it("reports GPU generation failure before requesting a capture and releases recording ownership", async () => {
    const capture = {
      render: vi.fn(async (_signal: AbortSignal, request: () => void, waitForCapture: () => Promise<void> = async () => {}) => { await waitForCapture(); request(); }),
      dispose: vi.fn(async () => {}),
    };
    mocks.createCapture.mockReturnValue(capture);
    const request = vi.fn();
    mocks.recordVideo.mockImplementation(async (options: MorphRecordingOptions) => {
      await options.renderFrame(0.5, options.signal, request, async () => {});
      return new Blob();
    });
    const p = recordingProps();
    const { unmount } = render(<MorphLayer {...p} />);
    const resource = await ready();
    resource.object.generatorError = new Error("Shader failed");
    const recorder = p.onRecorder.mock.calls[0][0] as MorphVideoRecorder;
    await expect(recorder({ duration: 6, signal: new AbortController().signal, onProgress: vi.fn() })).rejects.toThrow("Shader failed");
    expect(request).not.toHaveBeenCalled();
    expect(capture.dispose).toHaveBeenCalledTimes(1);
    unmount();
    expect(resource.dispose).toHaveBeenCalledTimes(1);
  });

  it("releases recording ownership even if restoring Spark after the recording fails", async () => {
    const capture = {
      render: vi.fn(async (_signal: AbortSignal, request: () => void, waitForCapture: () => Promise<void> = async () => {}) => { await waitForCapture(); request(); }),
      dispose: vi.fn(async () => { throw new Error("Context lost"); }),
    };
    mocks.createCapture.mockReturnValue(capture);
    mocks.recordVideo.mockImplementation(async (options: MorphRecordingOptions) => {
      await options.renderFrame(0.5, options.signal, vi.fn(), async () => {});
      return new Blob();
    });
    const p = recordingProps();
    const { unmount } = render(<MorphLayer {...p} />);
    const resource = await ready();
    const recorder = p.onRecorder.mock.calls[0][0] as MorphVideoRecorder;
    await expect(recorder({ duration: 6, signal: new AbortController().signal, onProgress: vi.fn() })).rejects.toThrow("Context lost");
    unmount();
    expect(resource.dispose).toHaveBeenCalledTimes(1);
  });
});

afterEach(() => vi.unstubAllGlobals());

describe("MorphLayer lifecycle", () => {
  it.each(["ready", "error"] as const)("reports a pending preparation's %s status through its original callback", async (phase) => {
    let p = props();
    const initialStatus = p.onStatus;
    const { rerender } = render(<MorphLayer {...p} />);
    await settle();
    const nextStatus = vi.fn();
    // A root can replace its status closure while the same worker is running.
    // That must not assign this worker's result to the newer closure's key.
    p = { ...p, onStatus: nextStatus };
    rerender(<MorphLayer {...p} />);
    expect(instances).toHaveLength(1);
    instances[0].reply(phase === "ready"
      ? { id: 1, ok: true, result: data }
      : { id: 1, ok: false, error: "Preparation failed" });
    await settle();
    expect(initialStatus).toHaveBeenLastCalledWith(expect.objectContaining({ phase }));
    expect(nextStatus).not.toHaveBeenCalled();
  });

  it("holds source references through worker completion and disposes the temporary GPU object on close", async () => {
    const p = props();
    const { unmount } = render(<MorphLayer {...p} />);
    await settle();
    expect(p.onStatus).toHaveBeenLastCalledWith({ phase: "loading" });
    expect(mocks.release).not.toHaveBeenCalled();
    expect(instances[0].postMessage.mock.calls[0][0].seed).toBe(42);
    expect(instances[0].postMessage.mock.calls[0][0].mode).toBe("shape");
    expect(instances[0].postMessage.mock.calls[0][0].autoAlign).toBe(false);
    const resource = await ready();
    expect(mocks.release.mock.calls).toEqual([["/a.ply"], ["/b.ply"]]);
    expect(p.onStatus).toHaveBeenLastCalledWith(expect.objectContaining({ phase: "ready", count: 2, meanTravel: 0.24, alignmentMs: 27 }));
    expect(mocks.scene.add).toHaveBeenCalledWith(resource.object);
    expect(instances[0].terminate).toHaveBeenCalledTimes(1);
    unmount();
    expect(resource.dispose).toHaveBeenCalledTimes(1);
    expect(p.onDisplay).toHaveBeenLastCalledWith(null);
    expect(mocks.release).toHaveBeenCalledTimes(2);
  });

  it("keeps exact originals at both endpoints and shows particles for partial target blend", async () => {
    let p = props();
    const { rerender } = render(<MorphLayer {...p} />);
    const resource = await ready();
    act(() => mocks.frame?.({}, 0));
    expect(resource.object.visible).toBe(false);
    expect(p.onDisplay).toHaveBeenLastCalledWith({ sourceId: "source", targetId: "target", endpoint: "source" });

    p = { ...p, morph: { ...p.morph, t: 0.5 } };
    rerender(<MorphLayer {...p} />);
    act(() => mocks.frame?.({}, 0));
    expect(resource.object.visible).toBe(true);
    expect(p.onDisplay).toHaveBeenLastCalledWith({ sourceId: "source", targetId: "target", endpoint: "particles" });
    p = { ...p, morph: { ...p.morph, t: 1 } };
    rerender(<MorphLayer {...p} />);
    act(() => mocks.frame?.({}, 0));
    expect(resource.object.visible).toBe(false);
    expect(p.onDisplay).toHaveBeenLastCalledWith({ sourceId: "source", targetId: "target", endpoint: "target" });

    p = { ...p, morph: { ...p.morph, targetBlend: 0.7 } };
    rerender(<MorphLayer {...p} />);
    act(() => mocks.frame?.({}, 0));
    expect(resource.object.visible).toBe(true);
    expect(resource.update).toHaveBeenLastCalledWith(1, 1, 0.7, "shape", 0, 0);
    expect(p.onDisplay).toHaveBeenLastCalledWith({ sourceId: "source", targetId: "target", endpoint: "particles" });
    expect(mocks.createGpu).toHaveBeenCalledTimes(1);
  });

  it("uses the full configured duration, preserves progress on knob edits, and emits completion only once", async () => {
    let p = props();
    const { rerender } = render(<MorphLayer {...p} />);
    await ready();
    p = { ...p, morph: { ...p.morph, playing: true } };
    rerender(<MorphLayer {...p} />);
    act(() => { for (let i = 0; i < 180; i++) mocks.frame?.({}, 1 / 60); });
    expect(p.playback.t).toBeCloseTo(0.5, 8);
    expect(p.playback.playing).toBe(true);
    p = { ...p, morph: { ...p.morph, dissolve: 0.3 } };
    rerender(<MorphLayer {...p} />);
    expect(p.playback.t).toBeCloseTo(0.5, 8);
    act(() => { for (let i = 0; i < 181; i++) mocks.frame?.({}, 1 / 60); });
    expect(p.playback).toEqual({ t: 1, playing: false });
    const completionCalls = vi.mocked(p.onChange).mock.calls.filter(([patch]) => patch.t === 1);
    expect(completionCalls).toEqual([[{ t: 1, playing: false }]]);
    act(() => mocks.frame?.({}, 1 / 60));
    expect(vi.mocked(p.onChange).mock.calls.filter(([patch]) => patch.t === 1)).toHaveLength(1);
  });

  it.each(["seed", "target", "mode", "autoAlign"])("terminates an old worker and ignores its late result when the %s changes", async (change) => {
    let p = props();
    const { rerender } = render(<MorphLayer {...p} />);
    await settle();
    const previousWorker = instances[0];
    p = change === "seed"
      ? { ...p, morph: { ...p.morph, seed: 123 } }
      : change === "mode"
      ? { ...p, morph: { ...p.morph, mode: "cloud", t: 0, playing: false } }
      : change === "autoAlign"
      ? { ...p, morph: { ...p.morph, autoAlign: true, t: 0, playing: false } }
      : { ...p, target: { ...p.target, id: "replacement", asset: "c" }, targetUrl: "/c.ply", morph: { ...p.morph, targetId: "replacement" } };
    rerender(<MorphLayer {...p} />);
    await settle();
    expect(previousWorker.terminate).toHaveBeenCalled();
    expect(instances).toHaveLength(2);
    expect(instances[1].postMessage.mock.calls[0][0].seed).toBe(change === "seed" ? 123 : 42);
    expect(instances[1].postMessage.mock.calls[0][0].mode).toBe(change === "mode" ? "cloud" : "shape");
    expect(instances[1].postMessage.mock.calls[0][0].autoAlign).toBe(change === "autoAlign");
    previousWorker.reply({ id: 1, ok: true, result: data });
    await settle();
    expect(mocks.createGpu).not.toHaveBeenCalled();
    expect(vi.mocked(p.onStatus).mock.calls.some(([status]) => status.phase === "error")).toBe(false);
    await ready();
    expect(mocks.createGpu).toHaveBeenCalledTimes(1);
    expect(mocks.release).toHaveBeenCalledTimes(4);
  });

  it("updates wave and arc during playback without rebuilding correspondence", async () => {
    let p = props();
    const { rerender } = render(<MorphLayer {...p} />);
    const resource = await ready();
    p.playback.t = 0.4;
    p.playback.playing = true;
    p = { ...p, morph: { ...p.morph, wave: 0.8, arc: 0.3 } };
    rerender(<MorphLayer {...p} />);
    act(() => mocks.frame?.({}, 0));
    expect(instances).toHaveLength(1);
    expect(mocks.createGpu).toHaveBeenCalledTimes(1);
    expect(p.playback).toEqual({ t: 0.4, playing: true });
    expect(resource.update).toHaveBeenLastCalledWith(0.4, 1, 1, "shape", 0.8, 0.3);
  });

  it("ignores a rename, rebuilds a moved endpoint, and discards the previous display", async () => {
    let p = props();
    const { rerender } = render(<MorphLayer {...p} />);
    const first = await ready();
    p.playback.t = 0.5;
    p.playback.playing = true;
    act(() => mocks.frame?.({}, 0));
    p = { ...p, source: { ...p.source, name: "Renamed" } };
    rerender(<MorphLayer {...p} />);
    expect(instances).toHaveLength(1);
    p = { ...p, source: { ...p.source, transform: { ...p.source.transform, position: [2, 0, 0] } } };
    rerender(<MorphLayer {...p} />);
    await settle();
    expect(first.dispose).toHaveBeenCalledTimes(1);
    expect(p.onDisplay).toHaveBeenLastCalledWith(null);
    expect(p.playback).toEqual({ t: 0.5, playing: false });
    expect(instances).toHaveLength(2);
    await ready();
  });

  it("does not create a worker or GPU object after cancellation while source decoding is pending", async () => {
    const load = deferred<void>();
    mocks.acquire.mockReturnValue({ initialized: load.promise, numSplats: 2 });
    const p = props();
    const { unmount } = render(<MorphLayer {...p} />);
    unmount();
    expect(mocks.release).not.toHaveBeenCalled();
    load.resolve();
    await settle();
    expect(instances).toHaveLength(0);
    expect(mocks.createGpu).not.toHaveBeenCalled();
    expect(mocks.release).toHaveBeenCalledTimes(2);
    expect(vi.mocked(p.onStatus).mock.calls.some(([status]) => status.phase === "error")).toBe(false);
  });

  it("waits for both decodes before releasing references when one asset fails", async () => {
    const failed = deferred<void>();
    const pending = deferred<void>();
    mocks.acquire.mockReturnValueOnce({ initialized: failed.promise, numSplats: 2 }).mockReturnValueOnce({ initialized: pending.promise, numSplats: 2 });
    const p = props();
    render(<MorphLayer {...p} />);
    failed.reject(new Error("Bad asset"));
    await settle();
    expect(mocks.release).not.toHaveBeenCalled();
    pending.resolve();
    await settle();
    expect(mocks.release).toHaveBeenCalledTimes(2);
    expect(p.onStatus).toHaveBeenLastCalledWith({ phase: "error", message: "Bad asset" });
    expect(p.playback.playing).toBe(false);
    expect(instances).toHaveLength(0);
  });

  it("reports worker failures, releases both assets, and never adds a partial GPU preview", async () => {
    const p = props();
    render(<MorphLayer {...p} />);
    await settle();
    instances[0].reply({ id: 1, ok: false, error: "No visible splats" });
    await settle();
    expect(p.onStatus).toHaveBeenLastCalledWith({ phase: "error", message: "No visible splats" });
    expect(p.onChange).toHaveBeenLastCalledWith({ playing: false });
    expect(mocks.release).toHaveBeenCalledTimes(2);
    expect(mocks.createGpu).not.toHaveBeenCalled();
    expect(instances[0].terminate).toHaveBeenCalledTimes(1);
  });

  it("terminates a worker immediately when posting its input fails", async () => {
    class FailingWorker extends FakeWorker {
      constructor() {
        super();
        this.postMessage.mockImplementation(() => { throw new DOMException("Could not transfer", "DataCloneError"); });
      }
    }
    vi.stubGlobal("Worker", FailingWorker);
    const p = props();
    render(<MorphLayer {...p} />);
    await settle();
    expect(p.onStatus).toHaveBeenLastCalledWith({ phase: "error", message: expect.stringContaining("Could not transfer") });
    expect(instances[0].terminate).toHaveBeenCalledTimes(1);
    expect(mocks.release).toHaveBeenCalledTimes(2);
    expect(mocks.createGpu).not.toHaveBeenCalled();
  });

  it("restores original display and pauses after a GPU generation error without reporting every frame", async () => {
    const p = props();
    const { unmount } = render(<MorphLayer {...p} />);
    const resource = await ready();
    p.playback.t = 0.4;
    p.playback.playing = true;
    act(() => mocks.frame?.({}, 0));
    expect(resource.object.visible).toBe(true);
    resource.object.generatorError = new Error("Shader failed");
    act(() => mocks.frame?.({}, 1 / 60));
    expect(resource.object.visible).toBe(false);
    expect(p.onDisplay).toHaveBeenLastCalledWith(null);
    expect(p.onStatus).toHaveBeenLastCalledWith({ phase: "error", message: "Shader failed" });
    expect(p.playback).toEqual({ t: 0.4, playing: false });
    const calls = vi.mocked(p.onStatus).mock.calls.length;
    act(() => mocks.frame?.({}, 1 / 60));
    expect(p.onStatus).toHaveBeenCalledTimes(calls);
    unmount();
    expect(resource.dispose).toHaveBeenCalledTimes(1);
  });
});
