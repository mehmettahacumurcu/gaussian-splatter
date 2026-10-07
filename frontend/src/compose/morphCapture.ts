import type { SparkRenderer } from "@sparkjsdev/spark";
import type { Camera, Scene, WebGLRenderer } from "three";

export interface MorphRecordOptions {
  duration: number;
  signal: AbortSignal;
  onProgress: (progress: number) => void;
}

export type MorphVideoRecorder = (options: MorphRecordOptions) => Promise<Blob | null>;
export type OnMorphRecorder = (recorder: MorphVideoRecorder | null, released?: MorphVideoRecorder) => void;

/** Abortable paint boundary, including when a tab is suspended. */
export function nextMorphPaint(signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    signal.throwIfAborted();
    const cancel = () => {
      cancelAnimationFrame(id);
      reject(signal.reason);
    };
    const id = requestAnimationFrame(() => {
      signal.removeEventListener("abort", cancel);
      resolve();
    });
    signal.addEventListener("abort", cancel, { once: true });
  });
}

/**
 * Spark 0.1.10's public prepare() awaits GPU readback AND worker sorting.
 * A dedicated viewpoint avoids racing the live viewpoint's in-flight sort.
 * No private Spark fields or fixed guesses about sort completion are needed.
 */
export function createMorphCapture(spark: SparkRenderer, gl: WebGLRenderer, scene: Scene, camera: Camera) {
  const previous = spark.viewpoint;
  const autoUpdate = spark.autoUpdate;
  const viewAutoUpdate = spark.defaultView.autoUpdate;
  const view = spark.newViewpoint({ autoUpdate: false, sortRadial: previous.sortRadial });
  spark.autoUpdate = false;
  spark.defaultView.setAutoUpdate(false);
  spark.viewpoint = view;
  let pending: Promise<void> = Promise.resolve();

  return {
    async render(signal: AbortSignal, capture: () => void, waitForCapture: () => Promise<void> = async () => {}) {
      signal.throwIfAborted();
      scene.updateMatrixWorld(true);
      camera.updateMatrixWorld(true);
      while (!spark.updateInternal({ scene, viewToWorld: camera.matrixWorld })) {
        // Accumulators still held by an earlier live sort can briefly be busy.
        await nextMorphPaint(signal);
      }
      const accumulator = spark.active;
      const borrowed = accumulator !== view.display?.accumulator;
      // 0.1.10 prepare() adds a temporary accumulator reference but does not
      // release it (updateDisplay separately owns the persistent display ref).
      // Balance that public ref via releaseAccumulator or a long video exhausts
      // Spark's accumulator pool. Explicit update above makes the borrowed
      // accumulator known, including on cancellation or sort failure.
      pending = view.prepare({ scene, camera, update: false }).finally(() => {
        if (borrowed) spark.releaseAccumulator(accumulator);
      });
      // prepare has no abort API. Keep ownership until it settles, even on cancel.
      await pending;
      // Sorting overlaps the frame interval; only the final draw waits for cadence.
      await waitForCapture();
      signal.throwIfAborted();
      spark.prepareViewpoint(view);
      gl.render(scene, camera);
      // Request in the same turn as the draw: the canvas does not preserve its buffer.
      capture();
    },
    async dispose() {
      try { await pending; } catch { /* the recording reports the original failure */ }
      try {
        spark.viewpoint = previous;
        spark.autoUpdate = autoUpdate;
        spark.defaultView.setAutoUpdate(viewAutoUpdate);
        spark.prepareViewpoint(previous);
      } finally {
        view.dispose();
      }
    },
  };
}
