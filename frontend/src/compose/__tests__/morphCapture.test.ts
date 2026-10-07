import { afterEach, describe, expect, it, vi } from "vitest";
import type { SparkRenderer } from "@sparkjsdev/spark";
import { Camera, Scene, type WebGLRenderer } from "three";
import { createMorphCapture, nextMorphPaint } from "../morphCapture";

function deferred() {
  let resolve!: () => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<void>((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

function fixture() {
  const events: string[] = [];
  const previous = { sortRadial: false };
  const accumulators = [{ refCount: 1 }];
  const view = {
    display: null as { accumulator: { refCount: number } } | null,
    prepare: vi.fn(async () => {}),
    dispose: vi.fn(() => {
      events.push("dispose");
      if (view.display) spark.releaseAccumulator(view.display.accumulator);
      view.display = null;
    }),
  };
  const spark = {
    viewpoint: previous as unknown,
    autoUpdate: true,
    defaultView: {
      autoUpdate: true,
      setAutoUpdate: vi.fn((value: boolean) => { spark.defaultView.autoUpdate = value; }),
    },
    active: accumulators[0],
    newViewpoint: vi.fn(() => view),
    updateInternal: vi.fn(() => {
      events.push("update");
      spark.releaseAccumulator(spark.active);
      spark.active = { refCount: 1 };
      accumulators.push(spark.active);
      return true;
    }),
    releaseAccumulator: vi.fn((accumulator: { refCount: number }) => { accumulator.refCount--; }),
    prepareViewpoint: vi.fn(() => { events.push("bind"); }),
  };
  let prepareGate: Promise<void> | undefined;
  view.prepare.mockImplementation(async () => {
    events.push("prepare");
    const accumulator = spark.active;
    // Model Spark 0.1.10's separate temporary prepare hold and display hold.
    if (accumulator !== view.display?.accumulator) accumulator.refCount++;
    if (prepareGate) await prepareGate;
    if (accumulator !== view.display?.accumulator) {
      accumulator.refCount++;
      if (view.display) spark.releaseAccumulator(view.display.accumulator);
      view.display = { accumulator };
    }
    events.push("sorted");
  });
  const scene = new Scene();
  const camera = new Camera();
  const gl = { render: vi.fn(() => { events.push("render"); }) };
  const capture = vi.fn(() => { events.push("capture"); });
  const owner = () => createMorphCapture(spark as unknown as SparkRenderer, gl as unknown as WebGLRenderer, scene, camera);
  return { spark, previous, view, gl, scene, camera, capture, events, accumulators, owner, gate: (promise: Promise<void>) => { prepareGate = promise; } };
}

afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });

describe("Spark morph capture", () => {
  it("awaits the dedicated viewpoint sort, then renders and requests capture in that order", async () => {
    const f = fixture();
    const gate = deferred();
    f.gate(gate.promise);
    const capture = f.owner();
    const result = capture.render(new AbortController().signal, f.capture);
    expect(f.spark.newViewpoint).toHaveBeenCalledWith(expect.objectContaining({ autoUpdate: false, sortRadial: false }));
    expect(f.spark.viewpoint).toBe(f.view);
    expect(f.spark.autoUpdate).toBe(false);
    expect(f.spark.defaultView.autoUpdate).toBe(false);
    expect(f.gl.render).not.toHaveBeenCalled();
    expect(f.capture).not.toHaveBeenCalled();
    gate.resolve();
    await result;
    expect(f.view.prepare).toHaveBeenCalledWith({ scene: f.scene, camera: f.camera, update: false });
    expect(f.events).toEqual(["update", "prepare", "sorted", "bind", "render", "capture"]);
    expect(f.gl.render).toHaveBeenCalledWith(f.scene, f.camera);
    await capture.dispose();
    expect(f.spark.viewpoint).toBe(f.previous);
    expect(f.spark.autoUpdate).toBe(true);
    expect(f.spark.defaultView.autoUpdate).toBe(true);
    expect(f.spark.prepareViewpoint).toHaveBeenLastCalledWith(f.previous);
    expect(f.view.dispose).toHaveBeenCalledTimes(1);
  });

  it("balances prepare's temporary accumulator hold over a long sequence", async () => {
    const f = fixture();
    const capture = f.owner();
    const signal = new AbortController().signal;
    for (let frame = 0; frame < 180; frame++) {
      await capture.render(signal, f.capture);
      expect(f.accumulators.filter((accumulator) => accumulator.refCount > 0)).toHaveLength(1);
      expect(f.spark.active.refCount).toBe(2); // active + current display
    }
    await capture.dispose();
    expect(f.spark.active.refCount).toBe(1);
    expect(f.accumulators.slice(0, -1).every((accumulator) => accumulator.refCount === 0)).toBe(true);
  });

  it("finishes sorting before waiting for cadence and defers the final draw until cadence permits it", async () => {
    const f = fixture();
    const sort = deferred();
    const cadence = deferred();
    const reachedCadence = deferred();
    f.gate(sort.promise);
    const capture = f.owner();
    const waitForCapture = vi.fn(async () => {
      f.events.push("cadence");
      reachedCadence.resolve();
      await cadence.promise;
    });
    const result = capture.render(new AbortController().signal, f.capture, waitForCapture);
    expect(waitForCapture).not.toHaveBeenCalled();
    sort.resolve();
    await reachedCadence.promise;
    expect(f.events).toEqual(["update", "prepare", "sorted", "cadence"]);
    expect(f.gl.render).not.toHaveBeenCalled();
    expect(f.capture).not.toHaveBeenCalled();
    cadence.resolve();
    await result;
    expect(f.events).toEqual(["update", "prepare", "sorted", "cadence", "bind", "render", "capture"]);
    await capture.dispose();
  });

  it("does not bind or render a disposed viewpoint after cancellation during the cadence wait", async () => {
    const f = fixture();
    const cadence = deferred();
    const reachedCadence = deferred();
    const capture = f.owner();
    const controller = new AbortController();
    const result = capture.render(controller.signal, f.capture, async () => {
      reachedCadence.resolve();
      await cadence.promise;
    });
    await reachedCadence.promise;
    controller.abort();
    await capture.dispose();
    expect(f.view.dispose).toHaveBeenCalledTimes(1);
    expect(f.spark.viewpoint).toBe(f.previous);
    f.spark.prepareViewpoint.mockClear();
    cadence.resolve();
    await expect(result).rejects.toMatchObject({ name: "AbortError" });
    expect(f.spark.prepareViewpoint).not.toHaveBeenCalled();
    expect(f.gl.render).not.toHaveBeenCalled();
    expect(f.capture).not.toHaveBeenCalled();
    expect(f.spark.active.refCount).toBe(1);
  });

  it("does not release an unborrowed accumulator when a frame reuses the same display", async () => {
    const f = fixture();
    const capture = f.owner();
    const signal = new AbortController().signal;
    await capture.render(signal, f.capture);
    f.spark.updateInternal.mockReturnValue(true);
    f.spark.releaseAccumulator.mockClear();
    await capture.render(signal, f.capture);
    expect(f.spark.releaseAccumulator).not.toHaveBeenCalled();
    expect(f.spark.active.refCount).toBe(2);
    await capture.dispose();
    expect(f.spark.active.refCount).toBe(1);
  });

  it.each(["cancel", "failure"])("waits for pending preparation before restoring and disposing after %s", async (reason) => {
    const f = fixture();
    const gate = deferred();
    f.gate(gate.promise);
    const controller = new AbortController();
    const capture = f.owner();
    const render = capture.render(controller.signal, f.capture);
    const observed = render.catch((error: unknown) => error);
    if (reason === "cancel") controller.abort();
    const disposed = capture.dispose();
    await Promise.resolve();
    expect(f.spark.viewpoint).toBe(f.view);
    expect(f.view.dispose).not.toHaveBeenCalled();
    expect(f.spark.active.refCount).toBe(2);
    if (reason === "failure") gate.reject(new Error("Sort failed"));
    else gate.resolve();
    const error = await observed;
    await disposed;
    expect(error).toMatchObject(reason === "cancel" ? { name: "AbortError" } : { message: "Sort failed" });
    expect(f.gl.render).not.toHaveBeenCalled();
    expect(f.capture).not.toHaveBeenCalled();
    expect(f.spark.viewpoint).toBe(f.previous);
    expect(f.spark.autoUpdate).toBe(true);
    expect(f.spark.defaultView.autoUpdate).toBe(true);
    expect(f.view.dispose).toHaveBeenCalledTimes(1);
    expect(f.spark.active.refCount).toBe(1);
  });

  it("preserves pre-existing disabled update flags", async () => {
    const f = fixture();
    f.spark.autoUpdate = false;
    f.spark.defaultView.autoUpdate = false;
    const capture = f.owner();
    await capture.render(new AbortController().signal, f.capture);
    await capture.dispose();
    expect(f.spark.autoUpdate).toBe(false);
    expect(f.spark.defaultView.autoUpdate).toBe(false);
  });

  it("disposes the owned viewpoint even if rebinding the previous viewpoint fails", async () => {
    const f = fixture();
    const capture = f.owner();
    await capture.render(new AbortController().signal, f.capture);
    f.spark.prepareViewpoint.mockImplementation(() => { throw new Error("Context lost"); });
    await expect(capture.dispose()).rejects.toThrow("Context lost");
    expect(f.spark.viewpoint).toBe(f.previous);
    expect(f.spark.autoUpdate).toBe(true);
    expect(f.spark.defaultView.autoUpdate).toBe(true);
    expect(f.view.dispose).toHaveBeenCalledTimes(1);
    expect(f.spark.active.refCount).toBe(1);
  });

  it("cancels while waiting for a busy accumulator without starting another preparation", async () => {
    const f = fixture();
    f.spark.updateInternal.mockReturnValue(false);
    const request = vi.fn(() => 7);
    const cancel = vi.fn();
    vi.stubGlobal("requestAnimationFrame", request);
    vi.stubGlobal("cancelAnimationFrame", cancel);
    const capture = f.owner();
    const controller = new AbortController();
    const result = capture.render(controller.signal, f.capture);
    controller.abort();
    await expect(result).rejects.toMatchObject({ name: "AbortError" });
    await capture.dispose();
    expect(request).toHaveBeenCalledTimes(1);
    expect(cancel).toHaveBeenCalledWith(7);
    expect(f.view.prepare).not.toHaveBeenCalled();
    expect(f.gl.render).not.toHaveBeenCalled();
    expect(f.spark.viewpoint).toBe(f.previous);
  });
});

describe("morph paint boundary", () => {
  it("resolves only at the requested animation frame and removes its abort listener", async () => {
    let callback!: FrameRequestCallback;
    vi.stubGlobal("requestAnimationFrame", vi.fn((next: FrameRequestCallback) => { callback = next; return 3; }));
    const cancel = vi.fn();
    vi.stubGlobal("cancelAnimationFrame", cancel);
    const controller = new AbortController();
    const result = nextMorphPaint(controller.signal);
    callback(100);
    await result;
    controller.abort();
    expect(cancel).not.toHaveBeenCalled();
  });

  it("rejects an already aborted signal without requesting a frame", async () => {
    const request = vi.fn();
    vi.stubGlobal("requestAnimationFrame", request);
    const controller = new AbortController();
    controller.abort();
    await expect(nextMorphPaint(controller.signal)).rejects.toMatchObject({ name: "AbortError" });
    expect(request).not.toHaveBeenCalled();
  });
});
