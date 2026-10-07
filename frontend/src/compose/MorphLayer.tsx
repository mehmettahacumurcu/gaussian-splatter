import { useEffect, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { SparkRenderer } from "@sparkjsdev/spark";
import type { PreparedMorph } from "./morphData";
import { createMorphGpu, type MorphGpu } from "./morphGpu";
import { extractMorphSource } from "./morphSource";
import type { MorphWorkerRequest, MorphWorkerResponse } from "./morph.worker";
import type { MorphPlayback, MorphState, MorphStatus } from "./morphTypes";
import { packedSplatsCache } from "./splatCache";
import type { SceneObject } from "./types";

export interface MorphDisplay {
  sourceId: string;
  targetId: string;
  endpoint: "source" | "target" | "particles";
}

interface Props {
  source: SceneObject;
  target: SceneObject;
  sourceUrl: string;
  targetUrl: string;
  morph: MorphState;
  playback: MorphPlayback;
  onChange: (patch: Partial<MorphState>) => void;
  onStatus: (status: MorphStatus) => void;
  onDisplay: (display: MorphDisplay | null) => void;
}

function prepareInWorker(a: Float32Array, b: Float32Array, seed: number, signal: AbortSignal) {
  return new Promise<PreparedMorph>((resolve, reject) => {
    signal.throwIfAborted();
    const worker = new Worker(new URL("./morph.worker.ts", import.meta.url), { type: "module" });
    const finish = () => {
      signal.removeEventListener("abort", cancel);
      worker.terminate();
    };
    const cancel = () => {
      finish();
      reject(new DOMException("Morph cancelled", "AbortError"));
    };
    signal.addEventListener("abort", cancel, { once: true });
    worker.onmessage = ({ data }: MessageEvent<MorphWorkerResponse>) => {
      finish();
      if (data.ok) resolve(data.result);
      else reject(new Error(data.error));
    };
    worker.onerror = (event) => {
      finish();
      reject(new Error(event.message || "Morph worker failed"));
    };
    worker.onmessageerror = () => {
      finish();
      reject(new Error("Morph worker result could not be read"));
    };
    const request: MorphWorkerRequest = { id: 1, a, b, seed };
    try {
      worker.postMessage(request, [a.buffer, b.buffer]);
    } catch (error) {
      finish();
      reject(error);
    }
  });
}

/** Owns the temporary preview only. The scene document and source data stay intact. */
export function MorphLayer(props: Props) {
  const { source, target, sourceUrl, targetUrl, morph, playback, onDisplay } = props;
  const scene = useThree((state) => state.scene);
  const gl = useThree((state) => state.gl);
  const latest = useRef(props);
  latest.current = props;
  const gpu = useRef<MorphGpu | null>(null);
  const display = useRef<MorphDisplay["endpoint"] | null>(null);
  // A rename must not rebuild 200k pairs. Geometry/appearance edits must.
  const sourceKey = JSON.stringify([source.id, source.asset, source.visible, source.transform, source.crop, source.color]);
  const targetKey = JSON.stringify([target.id, target.asset, target.visible, target.transform, target.crop, target.color]);

  useEffect(() => {
    playback.t = morph.t;
    playback.playing = morph.playing;
  }, [morph.t, morph.playing, playback]);

  useEffect(() => {
    const abort = new AbortController();
    let resource: MorphGpu | null = null;
    const start = performance.now();
    latest.current.onChange({ t: playback.t, playing: false });
    latest.current.onStatus({ phase: "loading" });
    const packedA = packedSplatsCache.acquire(sourceUrl);
    const packedB = packedSplatsCache.acquire(targetUrl);
    void (async () => {
      try {
        const initialized = await Promise.allSettled([packedA.initialized, packedB.initialized]);
        for (const result of initialized) if (result.status === "rejected") throw result.reason;
        abort.signal.throwIfAborted();
        if (Math.max(packedA.numSplats, packedB.numSplats) > 1_000_000) {
          throw new Error("Morph v1 supports up to 1,000,000 splats per asset. Start with about 200,000.");
        }
        const extracted = await Promise.allSettled([
          extractMorphSource(packedA, source, abort.signal),
          extractMorphSource(packedB, target, abort.signal),
        ]);
        const [a, b] = extracted.map((result) => {
          if (result.status === "rejected") throw result.reason;
          return result.value;
        });
        const data = await prepareInWorker(a, b, morph.seed, abort.signal);
        abort.signal.throwIfAborted();
        resource = createMorphGpu(data, gl.capabilities.maxTextureSize);
        resource.object.visible = false;
        scene.add(resource.object);
        gpu.current = resource;
        latest.current.onStatus({ phase: "ready", count: data.count, precomputeMs: performance.now() - start });
      } catch (error) {
        if (!abort.signal.aborted) {
          latest.current.onStatus({ phase: "error", message: error instanceof Error ? error.message : String(error) });
          latest.current.onChange({ playing: false });
        }
      } finally {
        // Keep source ownership until asynchronous decode/extraction has ended.
        packedSplatsCache.release(sourceUrl);
        packedSplatsCache.release(targetUrl);
      }
    })();
    return () => {
      abort.abort();
      if (gpu.current === resource) gpu.current = null;
      const spark = scene.children.find((child): child is SparkRenderer => child instanceof SparkRenderer);
      resource?.dispose(spark);
      display.current = null;
      onDisplay(null);
    };
  }, [sourceKey, targetKey, sourceUrl, targetUrl, morph.seed, scene, gl, playback, onDisplay]);

  useFrame((_, delta) => {
    const resource = gpu.current;
    if (!resource) return;
    const current = latest.current;
    if (resource.object.generatorError) {
      const error = resource.object.generatorError;
      resource.object.visible = false;
      playback.playing = false;
      gpu.current = null;
      display.current = null;
      onDisplay(null);
      current.onChange({ t: playback.t, playing: false });
      current.onStatus({ phase: "error", message: error instanceof Error ? error.message : String(error) });
      return;
    }
    if (playback.playing) {
      playback.t = Math.min(1, playback.t + Math.min(delta, 0.1) / current.morph.duration);
      if (playback.t >= 1) {
        playback.playing = false;
        current.onChange({ t: 1, playing: false });
      }
    }
    const progress = playback.t * current.morph.targetBlend;
    const endpoint = progress <= 0 ? "source" : progress >= 1 ? "target" : "particles";
    resource.object.visible = endpoint === "particles";
    resource.update(playback.t, current.morph.dissolve, current.morph.targetBlend);
    if (display.current !== endpoint) {
      display.current = endpoint;
      onDisplay({ sourceId: source.id, targetId: target.id, endpoint });
    }
  });
  return null;
}
