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
// @ts-ignore — Spark types henüz tam değil
import { SplatMesh, SparkRenderer } from "@sparkjsdev/spark";
import { frameUrl, type PerfStats } from "../api";

interface Props {
  jobId: string;
  numFrames: number;
  currentFrame: number;
  onLoadProgress?: (loaded: number, total: number) => void;
  onReady?: () => void;
  onError?: (msg: string) => void;
  onPerfTick?: (stats: PerfStats) => void;
}

export function SplatViewerSpark({
  jobId,
  numFrames,
  currentFrame,
  onLoadProgress,
  onReady,
  onError,
  onPerfTick,
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
  // v4.1 — Frame interpolation: smooth float time + auto-play detection
  const smoothTimeRef = useRef<number>(0);
  const lastFrameChangeAtRef = useRef<number>(0);  // performance.now() of last currentFrame update
  const lastFrameValueRef = useRef<number>(0);     // previous currentFrame for delta detection
  const detectedFpsRef = useRef<number>(10);       // adaptive: parent'in update hızını ölç
  const lastAnimateAtRef = useRef<number>(0);
  // v4.2 — WASD free-fly camera (FPS-style)
  const yawRef = useRef<number>(0);                // mouse look horizontal angle
  const pitchRef = useRef<number>(0);              // mouse look vertical angle (clamped)
  const keysRef = useRef<Set<string>>(new Set());  // pressed keys (lowercase)
  const moveSpeedRef = useRef<number>(2.0);        // unit/sec base speed (wheel adjusts)

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

    // --- v4.2: Free-fly camera (FPS-style) — WASD + mouse look + wheel speed ---
    //
    // Controls:
    //   Mouse drag (sol klik) → look around (yaw + pitch)
    //   W / A / S / D         → forward / strafe-left / back / strafe-right
    //   Q / E or Space / Ctrl → up / down (world axis)
    //   Shift                 → sprint (3× speed)
    //   Mouse wheel           → speed control (move faster/slower)
    //
    // Note: banana scene uses camera.up = (0, -1, 0). For statik room sahneler
    // user may need invert button. Bu state diff implementation'da yok henuz.

    // Initial camera position — scene icine bakacak sekilde
    camera.position.set(0, 0, 5);
    yawRef.current = 0;
    pitchRef.current = 0;

    const updateCameraOrientation = () => {
      // Yaw + pitch → forward vector
      // camera.up = (0, -1, 0) (banana). Real "up" world = camera.up.negate() = (0, 1, 0)
      // Spherical → cartesian: forward unit vector
      const cy = Math.cos(yawRef.current);
      const sy = Math.sin(yawRef.current);
      const cp = Math.cos(pitchRef.current);
      const sp = Math.sin(pitchRef.current);
      // Banana convention: -Y is "up", so pitch increases downward in world space
      const forward = new THREE.Vector3(sy * cp, -sp, -cy * cp);
      const lookTarget = camera.position.clone().add(forward);
      camera.lookAt(lookTarget);
    };
    updateCameraOrientation();

    let isDragging = false;
    let lastX = 0;
    let lastY = 0;

    const onMouseDown = (e: MouseEvent) => {
      isDragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
      // Pointer lock daha iyi UX olur ama basit tutmak icin direkt drag tracking
    };
    const onMouseMove = (e: MouseEvent) => {
      if (!isDragging) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      const sensitivity = 0.003;
      yawRef.current -= dx * sensitivity;
      pitchRef.current = Math.max(-1.4, Math.min(1.4, pitchRef.current - dy * sensitivity));
      updateCameraOrientation();
    };
    const onMouseUp = () => {
      isDragging = false;
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      // v4.3: Wheel default = forward/back (orbit-zoom tarzi, FPS hibrit)
      //   Shift+wheel = speed control (FPS standardi)
      if (e.shiftKey) {
        // Speed adjust
        const factor = e.deltaY > 0 ? 0.9 : 1.1;
        moveSpeedRef.current = Math.max(0.05, Math.min(50, moveSpeedRef.current * factor));
      } else {
        // Forward/back hareket — wheel down = backward (mantikli)
        const forward = new THREE.Vector3();
        camera.getWorldDirection(forward);
        const distance = e.deltaY * 0.01 * moveSpeedRef.current;
        camera.position.add(forward.multiplyScalar(-distance));
        updateCameraOrientation();
      }
    };

    // Keyboard listeners — global window'a (canvas focus gerekli degil)
    const onKeyDown = (e: KeyboardEvent) => {
      keysRef.current.add(e.key.toLowerCase());
      // Browser default'lari engelle (space scroll, vs)
      if ([" ", "w", "a", "s", "d", "q", "e"].includes(e.key.toLowerCase())) {
        e.preventDefault();
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      keysRef.current.delete(e.key.toLowerCase());
    };

    // v4.3: Bug fix — focus kaybinda (Alt+Tab, browser switch, viewer remount) tuslarin
    // "stuck" kalmasini engelle. Window blur olunca tum tuslari serbest birak.
    const onBlur = () => {
      keysRef.current.clear();
      isDragging = false;
    };

    renderer.domElement.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    renderer.domElement.addEventListener("wheel", onWheel, { passive: false });
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", onBlur);

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

    // --- Animation loop with smooth interpolation (v4.1) ---
    //
    // Strategy:
    //   - Parent yalnizca discrete currentFrame (integer) gonderir
    //   - Burda kendi smooth float time ilerletilir
    //   - Eger parent currentFrame son ~300 ms icinde degistiyse → auto-play algilandi,
    //     smooth time her tick advance edilir (detected fps × dt)
    //   - Aksi halde (slider drag, idle) → smooth time = currentFrame (snap)
    //   - Render: 2 komsu frame visible, opacity lerp (1-α, α)
    //
    const setMeshOpacity = (mesh: any, value: number) => {
      // Spark SplatMesh material'inde opacity hem standart property'de hem
      // custom uniform'da olabilir. Ikisini de set et — non-existent olan no-op.
      try {
        if (mesh.material) {
          mesh.material.transparent = true;
          if ("opacity" in mesh.material) (mesh.material as any).opacity = value;
          const u = (mesh.material as any).uniforms;
          if (u && u.opacity && "value" in u.opacity) u.opacity.value = value;
          if (u && u.uOpacity && "value" in u.uOpacity) u.uOpacity.value = value;
        }
        if ("opacity" in mesh) (mesh as any).opacity = value;
      } catch (_) {
        /* fallback: only visible flag */
      }
    };

    const animate = () => {
      if (!mountedRef.current) return;
      animFrameIdRef.current = requestAnimationFrame(animate);

      const now = performance.now();
      const dtSec = lastAnimateAtRef.current > 0
        ? Math.min((now - lastAnimateAtRef.current) / 1000, 0.1)
        : 0;
      lastAnimateAtRef.current = now;

      // Auto-play detection: son 300 ms icinde parent currentFrame degisti mi?
      const elapsedSinceParentUpdate = now - lastFrameChangeAtRef.current;
      const isAutoPlay = elapsedSinceParentUpdate < 300 && elapsedSinceParentUpdate > 1;

      let t: number;
      if (isAutoPlay && numFrames > 1) {
        // Smooth advance — detected fps × dt
        smoothTimeRef.current += dtSec * detectedFpsRef.current;
        // Wrap [0, numFrames)
        smoothTimeRef.current = ((smoothTimeRef.current % numFrames) + numFrames) % numFrames;
        // Sync drift correction: smooth time parent currentFrame'den ±1 frame fazla uzaklasmasin
        const drift = smoothTimeRef.current - currentFrameRef.current;
        if (drift > 1.5 || drift < -1.5) {
          // Reset to current
          smoothTimeRef.current = currentFrameRef.current;
        }
        t = smoothTimeRef.current;
      } else {
        // Static: snap to currentFrame
        smoothTimeRef.current = currentFrameRef.current;
        t = smoothTimeRef.current;
      }

      // Dual-frame blend
      const f0 = Math.floor(t) % numFrames;
      const f1 = (f0 + 1) % numFrames;
      const alpha = t - Math.floor(t);  // 0..1

      meshesRef.current.forEach((mesh, i) => {
        if (!mesh) return;
        if (i === f0) {
          mesh.visible = true;
          setMeshOpacity(mesh, alpha < 0.001 ? 1.0 : 1.0 - alpha);
        } else if (i === f1 && alpha > 0.001) {
          mesh.visible = true;
          setMeshOpacity(mesh, alpha);
        } else {
          mesh.visible = false;
          setMeshOpacity(mesh, 1.0); // reset for when reused
        }
      });

      // --- v4.2: WASD camera movement update (try/catch for safety) ---
      try {
        const keys = keysRef.current;
        if (keys.size > 0 && dtSec > 0) {
          const sprint = keys.has("shift") ? 3.0 : 1.0;
          const speed = moveSpeedRef.current * sprint * dtSec;

          // Forward direction (where camera looks)
          const forward = new THREE.Vector3();
          camera.getWorldDirection(forward);

          // Right direction (forward × world up)
          const right = new THREE.Vector3().crossVectors(forward, camera.up).normalize();

          const move = new THREE.Vector3();
          if (keys.has("w") || keys.has("arrowup")) move.add(forward);
          if (keys.has("s") || keys.has("arrowdown")) move.sub(forward);
          if (keys.has("d") || keys.has("arrowright")) move.add(right);
          if (keys.has("a") || keys.has("arrowleft")) move.sub(right);
          // Q/E or Space/Ctrl — world up/down (banana camera.up=(0,-1,0))
          const worldUp = camera.up.clone().negate();
          if (keys.has("e") || keys.has(" ")) move.add(worldUp);
          if (keys.has("q") || keys.has("control")) move.sub(worldUp);

          if (move.lengthSq() > 0) {
            move.normalize().multiplyScalar(speed);
            camera.position.add(move);
            updateCameraOrientation();
          }
        }
      } catch (err) {
        console.warn("[SplatViewerSpark] WASD movement error:", err);
        keysRef.current.clear();  // safety: stuck tuslari serbest birak
      }

      try {
        renderer.render(scene, camera);
      } catch (err) {
        console.warn("[SplatViewerSpark] render error:", err);
      }

      // Emit perf stats (FPS via dt; gauss count = only visible meshes —
      // 4D loads every timestamp as its own SplatMesh, summing them all
      // would report ~N_frames × N_per_frame which is misleading).
      if (onPerfTick && dtSec > 0) {
        const fps = 1 / dtSec;
        let gaussCount = 0;
        for (const m of meshesRef.current) {
          const mesh = m as { visible?: boolean; numSplats?: number; splatCount?: number } | null;
          if (!mesh || mesh.visible === false) continue;
          const n = mesh.numSplats ?? mesh.splatCount ?? 0;
          if (typeof n === "number") gaussCount += n;
        }
        onPerfTick({ fps, gaussCount });
      }
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
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", onBlur);
      window.removeEventListener("resize", onResize);
      // v4.3: cleanup'da key state'i de temizle
      keysRef.current.clear();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, numFrames]);

  // currentFrame değişince ref'i güncelle + auto-play detection (v4.1)
  useEffect(() => {
    const now = performance.now();
    const prev = lastFrameValueRef.current;
    const elapsed = now - lastFrameChangeAtRef.current;
    // Eger parent her N ms'de bir frame ilerletiyorsa, fps = 1000 / N
    if (elapsed > 0 && elapsed < 1000 && currentFrame !== prev) {
      const delta = Math.abs(currentFrame - prev);
      // Wrap-around detection: numFrames-1 → 0 jumps, ignore
      if (delta < 5) {
        const detectedFps = (1000 / elapsed) * Math.max(1, delta);
        // Smooth detected fps with EMA
        detectedFpsRef.current = detectedFpsRef.current * 0.7 + detectedFps * 0.3;
      }
    }
    lastFrameValueRef.current = currentFrame;
    lastFrameChangeAtRef.current = now;
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
