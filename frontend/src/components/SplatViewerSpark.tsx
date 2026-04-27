/**
 * 4DGS Viewer — Spark.js (World Labs) tabanlı.
 *
 * Mevcut SplatViewer'ın (mkkellogg) frame-swap pattern'inin alternatifi.
 * Spark.js Three.js üzerinde 4DGS-native renderer; tüm 90 PLY frame
 * tek seferde yüklenir, frame değişimi GPU'da visibility toggle ile
 * yapılır → "Gracia tarzı" smooth playback.
 *
 * STRATEJI:
 *   1. Mount'ta Three.js scene + WebGL renderer + Spark renderer kur
 *   2. Tüm 90 PLY paralel fetch → 90 SplatMesh (sahneye eklenir, hidden)
 *   3. Frame slider değişince: aktif SplatMesh visible=true, diğerleri false
 *   4. Active mesh + komşu ±1 mesh visible (smooth GPU compose, swap maliyeti yok)
 *   5. RAM budget: 90 frame × ~10 MB = ~900 MB GPU (8 GB karta sığar)
 *
 * BİZİM mkkellogg viewer'dan farkı:
 *   - mkkellogg: her frame için addSplatScene + removeSplatScene → 50-150 ms
 *   - Spark    : visible flag GPU side switch → ~1 ms
 *
 * SINIRLAMA:
 *   - 90 SplatMesh'i sort etmek (GPU'da) belli bir FPS düşüşü yaratabilir
 *   - Performance düşükse aktif±1 ile sınırlandır
 */
import { useEffect, useRef } from "react";
import * as THREE from "three";
// @ts-expect-error — Spark types henüz tam değil
import { SplatMesh, SparkRenderer } from "@sparkjsdev/spark";
import { frameUrl } from "../api";

interface Props {
  jobId: string;
  numFrames: number;
  currentFrame: number;
  onLoadProgress?: (loaded: number, total: number) => void;
  onReady?: () => void;
  onError?: (msg: string) => void;
}

export function SplatViewerSpark({
  jobId,
  numFrames,
  currentFrame,
  onLoadProgress,
  onReady,
  onError,
}: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Refs (animation loop dışından erişim için)
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sparkRef = useRef<any>(null);
  const meshesRef = useRef<any[]>([]);
  const currentFrameRef = useRef<number>(0);
  const mountedRef = useRef<boolean>(true);
  const animFrameIdRef = useRef<number | null>(null);

  useEffect(() => {
    mountedRef.current = true;
    if (!containerRef.current) return;
    const container = containerRef.current;

    // --- Three.js scene + camera + renderer ---
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(
      60, // fov
      container.clientWidth / container.clientHeight,
      0.01,
      1000
    );
    camera.position.set(0, 0, 5);
    camera.up.set(0, -1, 0); // banana orbital convention
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      preserveDrawingBuffer: false,
    });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // --- Spark renderer (Three.js render pipeline'a entegre) ---
    const spark = new SparkRenderer({ renderer });
    sparkRef.current = spark;

    // --- Basic mouse controls (orbit-style) ---
    let isDragging = false;
    let lastX = 0;
    let lastY = 0;
    const sphericalCamera = {
      radius: 5,
      theta: 0,
      phi: Math.PI / 2,
    };
    const updateCameraPosition = () => {
      camera.position.x =
        sphericalCamera.radius *
        Math.sin(sphericalCamera.phi) *
        Math.cos(sphericalCamera.theta);
      camera.position.y =
        sphericalCamera.radius * Math.cos(sphericalCamera.phi);
      camera.position.z =
        sphericalCamera.radius *
        Math.sin(sphericalCamera.phi) *
        Math.sin(sphericalCamera.theta);
      camera.lookAt(0, 0, 0);
    };
    updateCameraPosition();

    const onMouseDown = (e: MouseEvent) => {
      isDragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
    };
    const onMouseMove = (e: MouseEvent) => {
      if (!isDragging) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      sphericalCamera.theta -= dx * 0.005;
      sphericalCamera.phi = Math.max(
        0.1,
        Math.min(Math.PI - 0.1, sphericalCamera.phi - dy * 0.005)
      );
      updateCameraPosition();
    };
    const onMouseUp = () => {
      isDragging = false;
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      sphericalCamera.radius = Math.max(
        0.5,
        Math.min(50, sphericalCamera.radius + e.deltaY * 0.01)
      );
      updateCameraPosition();
    };
    renderer.domElement.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    renderer.domElement.addEventListener("wheel", onWheel, { passive: false });

    // --- Resize handler ---
    const onResize = () => {
      if (!container || !cameraRef.current || !rendererRef.current) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      cameraRef.current.aspect = w / h;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(w, h);
    };
    window.addEventListener("resize", onResize);

    // --- Animation loop ---
    const animate = () => {
      if (!mountedRef.current) return;
      animFrameIdRef.current = requestAnimationFrame(animate);

      // Visibility update (her frame'de cheap)
      const targetIdx = currentFrameRef.current;
      meshesRef.current.forEach((mesh, i) => {
        if (mesh) {
          // Sadece aktif frame visible. Komşular gizli.
          // (İleride: blend desteği için ±1 visible + opacity manipule)
          mesh.visible = i === targetIdx;
        }
      });

      renderer.render(scene, camera);
    };
    animate();

    // --- Tüm PLY'leri paralel yükle ---
    const loadAll = async () => {
      console.log(
        `[SplatViewerSpark] ${numFrames} frame paralel yükleniyor...`
      );
      let loadedCount = 0;
      const meshes: any[] = new Array(numFrames).fill(null);

      // Promise.all ile paralel
      const promises = Array.from({ length: numFrames }, async (_, i) => {
        try {
          const url = frameUrl(jobId, i);
          const mesh = new SplatMesh({ url });
          // Spark SplatMesh load promise'ı internal — onLoad callback'i veya
          // .ready promise'ı varsa bekle. Yoksa fire-and-forget olur.
          // 0.1.x: SplatMesh constructor sync ama internally async fetch
          // 2.0+: explicit await SplatMesh.load(url)
          if (mesh.initialized && typeof mesh.initialized.then === "function") {
            await mesh.initialized;
          } else if (typeof (SplatMesh as any).load === "function") {
            // 2.0 API
            // pass — already loaded with constructor URL
          }
          if (!mountedRef.current) return;
          mesh.visible = i === currentFrameRef.current;
          mesh.frustumCulled = false;
          scene.add(mesh);
          meshes[i] = mesh;
          loadedCount++;
          onLoadProgress?.(loadedCount, numFrames);
        } catch (e) {
          console.warn(`[SplatViewerSpark] frame ${i} yükleme hatası:`, e);
        }
      });

      await Promise.all(promises);
      meshesRef.current = meshes.filter((m) => m !== null);
      if (mountedRef.current) {
        console.log(
          `[SplatViewerSpark] ${meshesRef.current.length}/${numFrames} frame yüklendi`
        );
        onReady?.();
      }
    };

    loadAll().catch((e) => {
      console.error("[SplatViewerSpark] init error:", e);
      onError?.(String(e));
    });

    // --- Cleanup ---
    return () => {
      mountedRef.current = false;
      if (animFrameIdRef.current !== null) {
        cancelAnimationFrame(animFrameIdRef.current);
      }
      // Mesh'leri dispose et
      meshesRef.current.forEach((mesh) => {
        if (mesh) {
          scene.remove(mesh);
          if (mesh.dispose) mesh.dispose();
        }
      });
      meshesRef.current = [];
      // Renderer cleanup
      try {
        renderer.dispose();
      } catch (e) {
        /* noop */
      }
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
      // Listeners
      renderer.domElement.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
      renderer.domElement.removeEventListener("wheel", onWheel);
      window.removeEventListener("resize", onResize);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, numFrames]);

  // currentFrame değişince ref'i güncelle (animate loop kullansın)
  useEffect(() => {
    currentFrameRef.current = currentFrame;
  }, [currentFrame]);

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        height: "100%",
        position: "relative",
        backgroundColor: "#1a1a1a",
        overflow: "hidden",
      }}
    />
  );
}
