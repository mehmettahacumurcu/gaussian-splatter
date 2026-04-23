/**
 * 4D Gaussian Splat viewer.
 *
 * İşleyiş:
 *   - Mount olduğunda @mkkellogg/gaussian-splats-3d Viewer'ı oluşturur
 *   - Tüm N frame'i (scene olarak) önceden yükler
 *   - currentFrame değiştiğinde sadece ilgili scene'i görünür yapar
 *     (visibility toggle → smooth scrubbing, remove/add yerine)
 *   - Mouse ile orbit / pan / zoom (three.js OrbitControls içeride)
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
  const scenesReadyRef = useRef(false);
  const mountedRef = useRef(true);

  // ---- mount: viewer oluştur + tüm scene'leri yükle ----
  useEffect(() => {
    mountedRef.current = true;
    if (!containerRef.current) return;

    const viewer = new GaussianSplats3D.Viewer({
      rootElement: containerRef.current,
      selfDrivenMode: true,
      useBuiltInControls: true,
      cameraUp: [0, -1, 0],
      initialCameraPosition: [0, 0, 5],
      initialCameraLookAt: [0, 0, 0],
      sphericalHarmonicsDegree: 2,
      sharedMemoryForWorkers: false, // Tauri/Electron ortamında güvenli default
    });
    viewerRef.current = viewer;

    const loadAll = async () => {
      try {
        for (let i = 0; i < numFrames; i++) {
          if (!mountedRef.current) return;
          await viewer.addSplatScene(frameUrl(jobId, i), {
            splatAlphaRemovalThreshold: 5,
            showLoadingUI: false, // kendi progress display'imiz var
            progressiveLoad: false,
            // URL'de uzanti yok (/splat/.../frame/0) - format'i acikca soyle
            format: (GaussianSplats3D as any).SceneFormat?.Ply ?? 0,
          });
          onLoadProgress?.(i + 1, numFrames);
        }
        if (!mountedRef.current) return;
        scenesReadyRef.current = true;
        // Sadece currentFrame'i görünür yap, diğerlerini gizle
        applyVisibility(viewer, numFrames, currentFrame);
        viewer.start();
        onReady?.();
      } catch (err) {
        console.error("[SplatViewer] load error:", err);
        onError?.(String(err));
      }
    };
    loadAll();

    return () => {
      mountedRef.current = false;
      scenesReadyRef.current = false;
      try {
        // viewer.dispose() kendi canvas/DOM'unu sokumler.
        // Manuel removeChild StrictMode double-mount'ta "node not a child" verir.
        viewer.stop?.();
        viewer.dispose?.();
      } catch (e) {
        console.warn("[SplatViewer] dispose:", e);
      }
      viewerRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, numFrames]);

  // ---- currentFrame değişince visibility güncelle ----
  useEffect(() => {
    if (!scenesReadyRef.current || !viewerRef.current) return;
    applyVisibility(viewerRef.current, numFrames, currentFrame);
  }, [currentFrame, numFrames]);

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

/**
 * GaussianSplats3D'de her scene bir "splat buffer" tutar.
 * Viewer.splatMesh iç yapısında sceneVisibility ayarlanabilir.
 * Farklı sürümlerde farklı isimler var → birkaç yaklaşımı sırayla dener.
 */
function applyVisibility(viewer: any, total: number, active: number) {
  try {
    const mesh = viewer.splatMesh ?? viewer.getSplatMesh?.();
    if (mesh && typeof mesh.getSceneCount === "function") {
      const count = mesh.getSceneCount();
      for (let i = 0; i < count; i++) {
        const visible = i === active;
        // Yaklaşım 1: native API
        if (typeof mesh.setSceneVisibility === "function") {
          mesh.setSceneVisibility(i, visible);
          continue;
        }
        // Yaklaşım 2: scene objesine doğrudan yaz
        const scene = mesh.scenes?.[i];
        if (scene) {
          scene.visible = visible;
        }
      }
      return;
    }
    // Yaklaşım 3: viewer seviyesinde scene listesi
    const scenes = viewer.splatScenes ?? viewer.scenes ?? [];
    scenes.forEach((s: any, i: number) => {
      if (s) s.visible = i === active;
    });
  } catch (e) {
    console.warn("[SplatViewer] visibility error:", e);
  }
  // total parametresi şu an kullanılmıyor ama imzaya dahil
  void total;
}
