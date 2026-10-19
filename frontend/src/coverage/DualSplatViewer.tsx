/**
 * Two splats side by side under one shared camera: the trained splat on the
 * left, its coverage heatmap on the right. Moving in either pane moves both.
 *
 * Plain three.js + Spark (not react-three-fiber): two WebGL canvases render
 * the same THREE.PerspectiveCamera every frame, which keeps them in lockstep
 * without syncing state through React.
 */
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { PointerControls, SparkControls, SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import type { CoverageView } from "./coverageApi";

interface Props {
  normalUrl: string;
  heatUrl: string;
  /** World up of the capture (from the cameras). */
  up: [number, number, number];
  /** The camera jumps here whenever this object changes. */
  view: CoverageView | null;
}

type LoadState = "loading" | "ready" | "error";

// COLMAP/OpenCV camera (x right, y down, z forward) -> three.js camera (y up, looks down -z).
const FLIP_YZ = new THREE.Matrix4().makeScale(1, -1, -1);

export function applyView(camera: THREE.PerspectiveCamera, view: CoverageView) {
  const m = new THREE.Matrix4().fromArray(view.c2w).transpose().multiply(FLIP_YZ);
  m.decompose(camera.position, camera.quaternion, new THREE.Vector3());
  camera.fov = view.fov_y;
  camera.updateProjectionMatrix();
}

export function DualSplatViewer({ normalUrl, heatUrl, up, view }: Props) {
  const leftRef = useRef<HTMLCanvasElement>(null);
  const rightRef = useRef<HTMLCanvasElement>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const [state, setState] = useState<Record<"normal" | "heat", LoadState>>({ normal: "loading", heat: "loading" });

  useEffect(() => {
    const left = leftRef.current;
    const right = rightRef.current;
    if (!left || !right) return;
    const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
    camera.up.set(...up);
    cameraRef.current = camera;
    setState({ normal: "loading", heat: "loading" });

    let disposed = false;
    const panes = ([[left, normalUrl, "normal"], [right, heatUrl, "heat"]] as const).map(([canvas, url, key]) => {
      const renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
      const scene = new THREE.Scene();
      scene.add(new SparkRenderer({ renderer }));
      const mesh = new SplatMesh({ url });
      scene.add(mesh);
      mesh.initialized.then(
        () => !disposed && setState((s) => ({ ...s, [key]: "ready" })),
        () => !disposed && setState((s) => ({ ...s, [key]: "error" })),
      );
      return { canvas, renderer, scene, mesh };
    });
    // Keyboard (WASD/arrows, document-wide) + mouse on the left; mouse on the right.
    const controlsLeft = new SparkControls({ canvas: left });
    const controlsRight = new PointerControls({ canvas: right });

    const resize = () => {
      const w = left.clientWidth;
      const h = left.clientHeight;
      if (!w || !h) return;
      for (const p of panes) p.renderer.setSize(p.canvas.clientWidth, p.canvas.clientHeight, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(left);
    resize();

    let last = performance.now();
    panes[0].renderer.setAnimationLoop((now: number) => {
      const dt = Math.min((now - last) / 1000, 0.1);
      last = now;
      controlsLeft.update(camera);
      controlsRight.update(dt, camera);
      for (const p of panes) p.renderer.render(p.scene, camera);
    });

    return () => {
      disposed = true;
      observer.disconnect();
      panes[0].renderer.setAnimationLoop(null);
      for (const p of panes) {
        p.mesh.dispose();
        p.renderer.dispose();
      }
      cameraRef.current = null;
    };
  }, [normalUrl, heatUrl, up]);

  useEffect(() => {
    if (view && cameraRef.current) applyView(cameraRef.current, view);
  }, [view, normalUrl, heatUrl]);

  const badge = (s: LoadState) => (s === "loading" ? " · yükleniyor…" : s === "error" ? " · yüklenemedi" : "");
  return (
    <div className="coverage-dual">
      <div className="coverage-pane">
        <canvas ref={leftRef} tabIndex={0} />
        <div className="coverage-label">Splat{badge(state.normal)}</div>
      </div>
      <div className="coverage-pane">
        <canvas ref={rightRef} tabIndex={0} />
        <div className="coverage-label">Kapsama ısı haritası{badge(state.heat)}</div>
        <div className="coverage-legend">
          Kaç kamera gördü
          <div className="coverage-ramp" />
          <div className="coverage-ramp-labels"><span>0</span><span>~6</span><span>40+</span></div>
        </div>
      </div>
    </div>
  );
}
