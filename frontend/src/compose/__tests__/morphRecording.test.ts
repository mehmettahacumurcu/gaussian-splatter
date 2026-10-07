import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createFrameSchedule, DEFAULT_RECORDING_FPS, recordMorphVideo, selectRecordingMimeType } from "../morphRecording";

describe("recording frame schedule", () => {
  it.each([[6, 30], [0.5, 30], [1.25, 24], [0.73, 30]])("includes exact endpoints for %s seconds at %s fps", (duration, fps) => {
    const frames = createFrameSchedule(duration, fps);
    const intervals = Math.round(duration * fps);
    expect(frames).toHaveLength(intervals + 1);
    expect(frames[0]).toBe(0);
    expect(frames.at(-1)).toBe(1);
    frames.forEach((t, index) => expect(t).toBe(index / intervals));
  });

  it("defaults to 30 fps and rejects impossible schedules", () => {
    expect(DEFAULT_RECORDING_FPS).toBe(30);
    expect(createFrameSchedule(1)).toHaveLength(31);
    for (const duration of [0, -1, NaN, Infinity, 0.0001]) expect(() => createFrameSchedule(duration)).toThrow(RangeError);
    for (const fps of [0, -1, NaN, Infinity]) expect(() => createFrameSchedule(1, fps)).toThrow(RangeError);
  });
});

describe("recording codec selection", () => {
  it("prefers VP9, falls back to VP8 then plain WebM, and rejects unsupported browsers", () => {
    const all = vi.fn(() => true);
    expect(selectRecordingMimeType(all)).toBe("video/webm;codecs=vp9");
    expect(all).toHaveBeenCalledTimes(1);
    expect(selectRecordingMimeType((mime) => mime !== "video/webm;codecs=vp9")).toBe("video/webm;codecs=vp8");
    expect(selectRecordingMimeType((mime) => mime === "video/webm")).toBe("video/webm");
    expect(selectRecordingMimeType(() => false)).toBeNull();
  });
});

describe("manual canvas recording", () => {
  let operations: string[];
  let recorders: MockRecorder[];
  let track: { requestFrame: ReturnType<typeof vi.fn>; stop: ReturnType<typeof vi.fn> };
  let stream: MediaStream;
  let canvas: HTMLCanvasElement;
  let controller: AbortController;
  let captureStream: ReturnType<typeof vi.fn>;

  class MockRecorder extends EventTarget {
    static isTypeSupported = vi.fn(() => true);
    static onConstruct: ((recorder: MockRecorder) => void) | undefined;
    state: RecordingState = "inactive";
    mimeType: string;
    start = vi.fn(() => {
      operations.push("start");
      this.state = "recording";
    });
    stop = vi.fn(() => {
      operations.push("stop");
      this.state = "inactive";
      const event = new Event("dataavailable");
      Object.defineProperty(event, "data", { value: new Blob(["video"]) });
      this.dispatchEvent(event);
      this.dispatchEvent(new Event("stop"));
    });
    constructor(_stream: MediaStream, options: MediaRecorderOptions) {
      super();
      this.mimeType = options.mimeType!;
      recorders.push(this);
      MockRecorder.onConstruct?.(this);
    }
  }

  beforeEach(() => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "performance"] });
    operations = [];
    recorders = [];
    controller = new AbortController();
    track = { requestFrame: vi.fn(() => operations.push("capture")), stop: vi.fn() };
    stream = { getVideoTracks: () => [track], getTracks: () => [track] } as unknown as MediaStream;
    captureStream = vi.fn(() => stream);
    canvas = { width: 960, height: 640, captureStream } as unknown as HTMLCanvasElement;
    MockRecorder.isTypeSupported.mockReset().mockReturnValue(true);
    MockRecorder.onConstruct = undefined;
    vi.stubGlobal("MediaRecorder", MockRecorder);
    vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => setTimeout(() => callback(performance.now()), 16));
    vi.stubGlobal("cancelAnimationFrame", (id: number) => clearTimeout(id));
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  const renderImmediately = async (_t: number, _signal: AbortSignal, capture: () => void, waitForCapture: () => Promise<void>) => {
    await waitForCapture();
    capture();
  };

  it("renders each fixed sample before capture, delays recorder start, paces frames, and returns a WebM", async () => {
    const times: number[] = [];
    const progress = vi.fn();
    const result = recordMorphVideo({
      canvas, duration: 0.5, fps: 4, signal: controller.signal, onProgress: progress,
      renderFrame: async (t, _signal, capture, waitForCapture) => {
        await waitForCapture();
        operations.push(`render:${t}`);
        if (t === 0) {
          expect(recorders).toHaveLength(0);
          expect(captureStream).not.toHaveBeenCalled();
        }
        capture();
        times.push(performance.now());
      },
    });
    await vi.runAllTimersAsync();
    const blob = await result;
    expect(captureStream).toHaveBeenCalledWith(0);
    expect(recorders).toHaveLength(1);
    expect(operations).toEqual(["render:0", "start", "capture", "render:0.5", "capture", "render:1", "capture", "stop"]);
    expect(times).toEqual([0, 250, 500]);
    expect(performance.now()).toBeGreaterThanOrEqual(766);
    expect(blob).toBeInstanceOf(Blob);
    expect(blob!.size).toBe(5);
    expect(blob!.type).toBe("video/webm;codecs=vp9");
    expect(progress.mock.calls.map(([value]) => value)).toEqual([0, 1 / 3, 2 / 3, 1]);
    expect(track.stop).toHaveBeenCalled();
    expect(vi.getTimerCount()).toBe(0);
  });

  it("waits for render completion before advancing even when rendering is slower than fps", async () => {
    const samples: number[] = [];
    let release: () => void = () => {};
    const result = recordMorphVideo({
      canvas, duration: 0.5, fps: 2, signal: controller.signal,
      renderFrame: async (t, _signal, capture, waitForCapture) => {
        samples.push(t);
        if (t === 0) await new Promise<void>((resolve) => { release = resolve; });
        await waitForCapture();
        capture();
      },
    });
    await vi.advanceTimersByTimeAsync(2000);
    expect(samples).toEqual([0]);
    expect(captureStream).not.toHaveBeenCalled();
    expect(track.requestFrame).not.toHaveBeenCalled();
    release();
    await vi.runAllTimersAsync();
    expect(await result).toBeInstanceOf(Blob);
    expect(samples).toEqual([0, 1]);
  });

  it("overlaps preparation with frame pacing so a fast sort does not lower the recording fps", async () => {
    const preparedAt: number[] = [];
    const capturedAt: number[] = [];
    const result = recordMorphVideo({
      canvas, duration: 0.5, fps: 4, signal: controller.signal,
      renderFrame: async (_t, _signal, capture, waitForCapture) => {
        await new Promise((resolve) => setTimeout(resolve, 100));
        preparedAt.push(performance.now());
        await waitForCapture();
        capture();
        capturedAt.push(performance.now());
      },
    });
    await vi.runAllTimersAsync();
    expect(await result).toBeInstanceOf(Blob);
    expect(preparedAt).toEqual([100, 200, 450]);
    expect(capturedAt).toEqual([100, 350, 600]);
  });

  it("cancels immediately during a pending render, stops recorder/tracks, and returns no blob", async () => {
    let lateCapture: (() => void) | undefined;
    let finishRender: () => void = () => {};
    const result = recordMorphVideo({
      canvas, duration: 0.5, fps: 2, signal: controller.signal,
      renderFrame: async (t, _signal, capture) => {
        if (t === 0) capture();
        else {
          lateCapture = capture;
          await new Promise<void>((resolve) => { finishRender = resolve; });
        }
      },
    });
    await vi.advanceTimersByTimeAsync(500);
    expect(lateCapture).toBeDefined();
    controller.abort();
    expect(recorders[0].stop).toHaveBeenCalledTimes(1);
    expect(track.stop).toHaveBeenCalled();
    expect(await result).toBeNull();
    expect(() => lateCapture!()).toThrow();
    finishRender();
    await vi.runAllTimersAsync();
    expect(track.requestFrame).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("cancels before first frame without starting MediaRecorder", async () => {
    const result = recordMorphVideo({
      canvas, duration: 1, signal: controller.signal,
      renderFrame: () => new Promise(() => {}),
    });
    controller.abort();
    expect(await result).toBeNull();
    expect(recorders).toHaveLength(0);
    expect(captureStream).not.toHaveBeenCalled();
    expect(track.stop).not.toHaveBeenCalled();
  });

  it("cancels while pacing frames and clears the pending timer", async () => {
    const result = recordMorphVideo({ canvas, duration: 1, signal: controller.signal, renderFrame: renderImmediately });
    await vi.advanceTimersByTimeAsync(1);
    controller.abort();
    expect(await result).toBeNull();
    expect(track.requestFrame).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("cancels during the final paint wait without returning the buffered video", async () => {
    const result = recordMorphVideo({ canvas, duration: 0.5, fps: 2, signal: controller.signal, renderFrame: renderImmediately });
    await vi.advanceTimersByTimeAsync(1000);
    expect(track.requestFrame).toHaveBeenCalledTimes(2);
    expect(recorders[0].stop).not.toHaveBeenCalled();
    controller.abort();
    expect(await result).toBeNull();
    expect(recorders[0].stop).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("rejects a viewport resize before capturing its inconsistent frame", async () => {
    const result = recordMorphVideo({
      canvas, duration: 0.5, fps: 2, signal: controller.signal,
      renderFrame: async (t, _signal, capture, waitForCapture) => {
        if (t === 1) canvas.width += 1;
        await waitForCapture();
        capture();
      },
    });
    const rejection = expect(result).rejects.toThrow("boyutu değişti");
    await vi.runAllTimersAsync();
    await rejection;
    expect(track.requestFrame).toHaveBeenCalledTimes(1);
    expect(recorders[0].stop).toHaveBeenCalledTimes(1);
    expect(track.stop).toHaveBeenCalled();
  });

  it("returns null without allocating resources when already cancelled", async () => {
    controller.abort();
    expect(await recordMorphVideo({ canvas, duration: 1, signal: controller.signal, renderFrame: renderImmediately })).toBeNull();
    expect(captureStream).not.toHaveBeenCalled();
  });

  it("cleans all stream tracks when requestFrame or recorder construction is unavailable", async () => {
    const extraTrack = { stop: vi.fn() };
    stream.getTracks = () => [track, extraTrack] as unknown as MediaStreamTrack[];
    track.requestFrame = undefined as unknown as typeof track.requestFrame;
    await expect(recordMorphVideo({ canvas, duration: 1, signal: controller.signal, renderFrame: renderImmediately })).rejects.toThrow("requestFrame");
    expect(track.stop).toHaveBeenCalled();
    expect(extraTrack.stop).toHaveBeenCalled();

    track.requestFrame = vi.fn();
    vi.stubGlobal("MediaRecorder", class {
      static isTypeSupported = () => true;
      constructor() { throw new Error("constructor failed"); }
    });
    await expect(recordMorphVideo({ canvas, duration: 1, signal: controller.signal, renderFrame: renderImmediately })).rejects.toThrow("constructor failed");
    expect(track.stop).toHaveBeenCalledTimes(2);
    expect(extraTrack.stop).toHaveBeenCalledTimes(2);
  });

  it("rejects unsupported browser features before allocating a stream", async () => {
    MockRecorder.isTypeSupported.mockReturnValue(false);
    await expect(recordMorphVideo({ canvas, duration: 1, signal: controller.signal, renderFrame: renderImmediately })).rejects.toThrow("WebM");
    expect(captureStream).not.toHaveBeenCalled();
    vi.stubGlobal("MediaRecorder", undefined);
    await expect(recordMorphVideo({ canvas, duration: 1, signal: controller.signal, renderFrame: renderImmediately })).rejects.toThrow("MediaRecorder");
    expect(captureStream).not.toHaveBeenCalled();
  });

  it("cleans up synchronous start/stop failures without unhandled promises", async () => {
    MockRecorder.onConstruct = (recorder) => { recorder.start.mockImplementation(() => { throw new Error("start failed"); }); };
    const startFailure = recordMorphVideo({
      canvas, duration: 0.5, fps: 2, signal: controller.signal,
      renderFrame: renderImmediately,
    });
    await expect(startFailure).rejects.toThrow("start failed");
    expect(track.stop).toHaveBeenCalled();

    MockRecorder.onConstruct = (recorder) => { recorder.stop.mockImplementation(() => { throw new Error("stop failed"); }); };
    const stopFailure = recordMorphVideo({ canvas, duration: 0.5, fps: 2, signal: controller.signal, renderFrame: renderImmediately });
    const rejection = expect(stopFailure).rejects.toThrow("stop failed");
    await vi.runAllTimersAsync();
    await rejection;
    expect(track.stop).toHaveBeenCalledTimes(2);
  });

  it("rejects asynchronous recorder errors and interrupts an uncooperative render", async () => {
    const result = recordMorphVideo({
      canvas, duration: 1, signal: controller.signal,
      renderFrame: async (t, _signal, capture) => {
        if (t === 0) capture();
        else await new Promise(() => {});
      },
    });
    const failure = new Event("error");
    Object.defineProperty(failure, "error", { value: new DOMException("encoder failed", "EncodingError") });
    recorders[0].dispatchEvent(failure);
    await expect(result).rejects.toThrow("encoder failed");
    expect(track.stop).toHaveBeenCalled();
    expect(vi.getTimerCount()).toBe(0);
  });

  it("rejects an unexpected recorder stop instead of returning a partial recording", async () => {
    const result = recordMorphVideo({
      canvas, duration: 1, signal: controller.signal,
      renderFrame: async (t, _signal, capture) => {
        if (t === 0) capture();
        else await new Promise(() => {});
      },
    });
    recorders[0].dispatchEvent(new Event("stop"));
    await expect(result).rejects.toThrow("beklenmedik");
    expect(track.stop).toHaveBeenCalled();
  });

  it("rejects an empty encoder result", async () => {
    MockRecorder.onConstruct = (recorder) => {
      recorder.stop.mockImplementation(() => {
        recorder.state = "inactive";
        recorder.dispatchEvent(new Event("stop"));
      });
    };
    const result = recordMorphVideo({ canvas, duration: 0.5, fps: 2, signal: controller.signal, renderFrame: renderImmediately });
    const rejection = expect(result).rejects.toThrow("boş");
    await vi.runAllTimersAsync();
    await rejection;
    expect(track.stop).toHaveBeenCalled();
  });

  it("rejects render errors or a missing capture and cleans up", async () => {
    await expect(recordMorphVideo({
      canvas, duration: 1, signal: controller.signal,
      renderFrame: async () => { throw new Error("render failed"); },
    })).rejects.toThrow("render failed");
    await expect(recordMorphVideo({ canvas, duration: 1, signal: controller.signal, renderFrame: async () => {} })).rejects.toThrow("did not capture");
    expect(track.stop).not.toHaveBeenCalled();
  });
});
