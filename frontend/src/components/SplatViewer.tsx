/**
 * 4D Gaussian Splat viewer (v3.7.9 — gray-screen fix + stale request drop).
 *
 * STRATEGY:
 *   1. Mount: hemen frame 0 fetch + render → UI ~1-2 sn'de açılır
 *   2. Arka planda 3 paralel prefetch (currentFrame'den genişleyerek)
 *   3. currentFrame değişince:
 *        - "pending frame" güncellenir
 *        - Tek swap loop'u var, latest pending'e swap eder
 *        - Stale (eski) requests DROP → play'de queue yığılmaz
 *   4. Swap pattern: ADD-before-REMOVE
 *        - Önce yeni scene eklenir (briefly 2 scene)
 *        - Sonra eski scene silinir
 *        - Gray screen YOK (eski scene yeni gelene kadar görünür)
 *
 * @mkkellogg/gaussian-splats-3d kütüphanesinin scene visibility toggle
 * çalışmadığı için her frame değişiminde scene swap yapıyoruz.
 */
import { useEffect, useRef } from "react";
// @ts-expect-error — kütüphanede type tanımı yok
import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";
import { frameUrl } from "../api";

interface Props {
  jobId: string;
  numFrames: number;
  currentFrame: number;
  onLoadProgress?: (loaded: number, total: number) => void;
  onReady?: () => void;
  onError?: (msg: string) => void;
  singleFrameMode?: boolean; // compat — şu an tek mod (cached swap)
}

export function SplatViewer({
  jobId,
  numFrames,
  currentFrame,
  onLoadProgress,
  onReady,
  onError,
}: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const viewerRef = useRef<any>(null);
  const mountedRef = useRef(true);

  // Scene state
  const loadedFrameRef = useRef<number>(-1); // şu an viewer'da gösterilen
  const pendingFrameRef = useRef<number>(0); // istenen son frame (latest)
  const swappingRef = useRef<boolean>(false);

  // Cache
  const blobCacheRef = useRef<(Blob | null)[]>([]);
  const fetchInFlightRef = useRef<(Promise<Blob> | null)[]>([]);
  const loadedCountRef = useRef(0);

  useEffect(() => {
    mountedRef.current = true;
    loadedFrameRef.current = -1;
    pendingFrameRef.current = currentFrame;
    swappingRef.current = false;
    blobCacheRef.current = new Array(numFrames).fill(null);
    fetchInFlightRef.current = new Array(numFrames).fill(null);
    loadedCountRef.current = 0;

    if (!containerRef.current) return;

    const viewer = new GaussianSplats3D.Viewer({
      rootElement: containerRef.current,
      selfDrivenMode: true,
      useBuiltInControls: true,
      cameraUp: [0, -1, 0],
      initialCameraPosition: [0, 0, 5],
      initialCameraLookAt: [0, 0, 0],
      sphericalHarmonicsDegree: 3,
      sharedMemoryForWorkers: false,
    });
    viewerRef.current = viewer;

    /**
     * Bir frame'i fetch + cache et. Duplicate fetch önler.
     */
    const fetchFrame = async (i: number): Promise<Blob> => {
      const cached = blobCacheRef.current[i];
      if (cached) return cached;

      const inFlight = fetchInFlightRef.current[i];
      if (inFlight) return inFlight;

      const promise = (async () => {
        const resp = await fetch(frameUrl(jobId, i));
        if (!resp.ok) throw new Error(`HTTP ${resp.status} frame ${i}`);
        const blob = await resp.blob();
        blobCacheRef.current[i] = blob;
        loadedCountRef.current++;
        onLoadProgress?.(loadedCountRef.current, numFrames);
        return blob;
      })();
      fetchInFlightRef.current[i] = promise;
      try {
        return await promise;
      } finally {
        fetchInFlightRef.current[i] = null;
      }
    };

    /**
     * Single swap loop — pending frame değiştikçe latest'e swap eder.
     * Aynı anda sadece 1 instance çalışır.
     * Pattern: ADD-before-REMOVE → gray screen yok.
     */
    const runSwapLoop = async () => {
      if (swappingRef.current) return; // başka loop çalışıyor
      swappingRef.current = true;
      try {
        while (mountedRef.current && pendingFrameRef.current !== loadedFrameRef.current) {
          const target = pendingFrameRef.current;
          const blob = await fetchFrame(target);
          if (!mountedRef.current) return;

          // Pending değişti mi (kullanıcı scrub etti) — restart
          if (pendingFrameRef.current !== target) continue;

          // ADD NEW first (eski scene görünmeye devam eder)
          const url = URL.createObjectURL(blob);
          try {
            await viewer.addSplatScene(url, {
              splatAlphaRemovalThreshold: 5,
              showLoadingUI: false,
              progressiveLoad: false,
              format: (GaussianSplats3D as any).SceneFormat?.Ply ?? 0,
            });
          } finally {
            URL.revokeObjectURL(url);
          }
          if (!mountedRef.current) return;

          // REMOVE OLD (yeni scene index 1'de, eski 0'da → 0'ı sil)
          if (loadedFrameRef.current >= 0 && typeof viewer.removeSplatScene === "function") {
            try {
              await viewer.removeSplatScene(0);
            } catch (e) {
              console.warn("[SplatViewer] removeSplatScene:", e);
            }
          }

          loadedFrameRef.current = target;
        }
      } catch (e) {
        console.error("[SplatViewer] swap loop error:", e);
      } finally {
        swappingRef.current = false;
      }
    };

    /**
     * currentFrame değişiminde çağrılır. Pending'i set + loop'u kick.
     * Stale requests doğal olarak drop olur (loop sadece LATEST'e gider).
     */
    const requestFrame = (frameIdx: number) => {
      pendingFrameRef.current = frameIdx;
      if (!swappingRef.current) {
        runSwapLoop();
      }
      // Loop zaten çalışıyorsa, pendingFrameRef güncellendiği için
      // bir sonraki iter'de o frame'e gidecek.
    };

    (viewer as any)._requestFrame = requestFrame;

    /**
     * Background prefetch — currentFrame'den dışarı doğru.
     */
    const backgroundPrefetch = async () => {
      const CONCURRENCY = 3;
      const order: number[] = [];
      for (let d = 0; d < numFrames; d++) {
        const next = currentFrame + d;
        const prevIdx = currentFrame - d - 1;
        if (next < numFrames && !order.includes(next)) order.push(next);
        if (prevIdx >= 0 && !order.includes(prevIdx)) order.push(prevIdx);
      }
      let idx = 0;
      const workers: Promise<void>[] = [];
      for (let w = 0; w < CONCURRENCY; w++) {
        workers.push((async () => {
          while (idx < order.length && mountedRef.current) {
            const i = order[idx++];
            if (blobCacheRef.current[i]) continue;
            try {
              await fetchFrame(i);
            } catch (e) {
              console.warn(`[SplatViewer] prefetch frame ${i} failed:`, e);
            }
          }
        })());
      }
      await Promise.all(workers);
      if (mountedRef.current) {
        console.log(`[SplatViewer] All ${numFrames} frames cached`);
      }
    };

    const init = async () => {
      try {
        // İlk frame — blocking ki UI hızlıca açılsın
        pendingFrameRef.current = currentFrame;
        await runSwapLoop();
        if (!mountedRef.current) return;
        viewer.start();
        onReady?.();
        console.log(`[SplatViewer] Initial frame ${currentFrame} ready`);

        // Arka planda prefetch
        backgroundPrefetch();
      } catch (err) {
        console.error("[SplatViewer] init error:", err);
        onError?.(String(err));
      }
    };

    init();

    return () => {
      mountedRef.current = false;
      try {
        viewer.stop?.();
        viewer.dispose?.();
      } catch (e) {
        console.warn("[SplatViewer] dispose:", e);
      }
      viewerRef.current = null;
      blobCacheRef.current = [];
      fetchInFlightRef.current = [];
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, numFrames]);

  // currentFrame değişince swap iste (stale requests doğal drop)
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    const reqFn = (viewer as any)._requestFrame;
    if (typeof reqFn === "function") {
      reqFn(currentFrame);
    }
  }, [currentFrame]);

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        height: "100%",
        position: "relative",
        backgroundColor: "#1a1a1a",
      }}
    />
  );
}
