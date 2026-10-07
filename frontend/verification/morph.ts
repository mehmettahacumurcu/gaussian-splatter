import { PerspectiveCamera, Scene, WebGLRenderer } from "three";
import { SparkRenderer } from "@sparkjsdev/spark";
import { MORPH_GLSL, interpolateMorphParticle } from "../src/compose/morphMath";
import { MORPH_STRIDE, type PreparedMorph } from "../src/compose/morphData";
import { createMorphGpu } from "../src/compose/morphGpu";
import { createMorphCapture } from "../src/compose/morphCapture";
import { recordMorphVideo } from "../src/compose/morphRecording";
import type { MorphMode } from "../src/compose/morphTypes";
import type { MorphWorkerResponse } from "../src/compose/morph.worker";

const result = document.querySelector<HTMLPreElement>("#result")!;
const button = document.querySelector<HTMLButtonElement>("#run")!;
const log = (message: string) => { result.textContent += `${message}\n`; };
const assert = (condition: unknown, message: string) => { if (!condition) throw new Error(message); };

// Transform feedback reads the shader's actual float outputs without packing
// through a color framebuffer or using a second hand-written interpolation.
function verifyShader() {
  const canvas = document.createElement("canvas");
  const gl = canvas.getContext("webgl2")!;
  assert(gl, "WebGL2 unavailable");
  const compile = (type: number, source: string) => {
    const shader = gl.createShader(type)!;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    assert(gl.getShaderParameter(shader, gl.COMPILE_STATUS), gl.getShaderInfoLog(shader) ?? "Shader failed");
    return shader;
  };
  const vertex = compile(gl.VERTEX_SHADER, `#version 300 es
precision highp float;
uniform vec4 a[4], b[4];
uniform float t, dissolve, blend, radius, mode, wave, arc;
out vec3 center, scales;
out vec4 quaternion, rgba;
${MORPH_GLSL}
void main() {
  interpolateMorph(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3],
    t, dissolve, blend, radius, mode, wave, arc, center, scales, quaternion, rgba);
  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}`);
  const fragment = compile(gl.FRAGMENT_SHADER, `#version 300 es
precision highp float;
out vec4 color;
void main() { color = vec4(1.0); }`);
  const program = gl.createProgram()!;
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.transformFeedbackVaryings(program, ["center", "scales", "quaternion", "rgba"], gl.INTERLEAVED_ATTRIBS);
  gl.linkProgram(program);
  assert(gl.getProgramParameter(program, gl.LINK_STATUS), gl.getProgramInfoLog(program) ?? "Link failed");
  gl.useProgram(program);
  const buffer = gl.createBuffer()!;
  gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, buffer);
  gl.bufferData(gl.TRANSFORM_FEEDBACK_BUFFER, 14 * 4, gl.STREAM_READ);
  gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, buffer);
  gl.enable(gl.RASTERIZER_DISCARD);
  const values = new Float32Array(14);
  let cases = 0, maxError = 0;
  for (const mode of ["cloud", "shape"] as const) {
    for (const projection of [0, 0.37, 1]) for (const wave of [0, 0.6, 1]) {
      for (const arc of [0, 1]) for (const t of [0, 0.1, 0.25, 0.5, 0.75, 0.99, 1]) {
        for (const blend of [0.5, 1]) for (const degenerate of [false, true]) {
          const a = new Float32Array([1, 2, 3, 0.9, 0, 0.2, 2, 0.317, 0, 0, 0, 1, 0.2, 0.4, 0.6, projection]);
          const b = new Float32Array([8, 9, 10, 0, 0.3, 0.02, 3, 0.317, 0, 1, 0, 0, 0.7, 0.3, 0.1, projection]);
          if (degenerate) { b.set(a.subarray(0, 3)); a.fill(0, 8, 12); b.set([0, 0, 0, -1], 8); }
          gl.uniform4fv(gl.getUniformLocation(program, "a[0]"), a);
          gl.uniform4fv(gl.getUniformLocation(program, "b[0]"), b);
          for (const [name, value] of Object.entries({ t, dissolve: 1, blend, radius: 10, mode: mode === "shape" ? 1 : 0, wave, arc })) {
            gl.uniform1f(gl.getUniformLocation(program, name), value);
          }
          gl.beginTransformFeedback(gl.POINTS);
          gl.drawArrays(gl.POINTS, 0, 1);
          gl.endTransformFeedback();
          gl.getBufferSubData(gl.TRANSFORM_FEEDBACK_BUFFER, 0, values);
          const cpu = interpolateMorphParticle(a, b, 0, t, 1, blend, 10, mode, wave, arc);
          const expected = [...cpu.slice(0, 3), ...cpu.slice(4, 7), ...cpu.slice(8, 12), ...cpu.slice(12, 15), cpu[3]];
          for (let i = 0; i < values.length; i++) {
            const error = Math.abs(values[i] - expected[i]);
            maxError = Math.max(maxError, error);
            assert(Number.isFinite(error) && error < 0.002, `${mode} parity failure t=${t} component=${i}: ${values[i]} vs ${expected[i]}`);
            if (t === 0 || (t === 1 && blend === 1)) {
              assert(error === 0, `${mode} endpoint is not exact: component ${i}`);
            }
          }
          cases++;
        }
      }
    }
  }
  assert(gl.getError() === gl.NO_ERROR, "WebGL error during parity check");
  gl.deleteBuffer(buffer);
  gl.deleteProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  gl.getExtension("WEBGL_lose_context")?.loseContext();
  log(`PASS GLSL/CPU: ${cases} cases; max absolute error ${maxError.toExponential(4)}`);
}

function sphere(count: number, target: boolean) {
  const attributes = new Float32Array(count * MORPH_STRIDE);
  for (let i = 0; i < count; i++) {
    const y = 1 - 2 * (i + 0.5) / count;
    const angle = i * Math.PI * (3 - Math.sqrt(5));
    const radius = Math.sqrt(1 - y * y);
    attributes.set([
      radius * Math.cos(angle) * (target ? 1.08 : 1), y, radius * Math.sin(angle), 0.8,
      0.025, 0.025, 0.025, 0, 0, 0, 0, 1,
      target ? 1 : 0.85, target ? 0.4 : 0.05, 0.015, 0,
    ], i * MORPH_STRIDE);
  }
  return attributes;
}

async function prepare(mode: MorphMode): Promise<PreparedMorph> {
  const worker = new Worker(new URL("../src/compose/morph.worker.ts", import.meta.url), { type: "module" });
  try {
    return await new Promise((resolve, reject) => {
      worker.onmessage = ({ data }: MessageEvent<MorphWorkerResponse>) => data.ok ? resolve(data.result) : reject(new Error(data.error));
      worker.onerror = (event) => reject(new Error(event.message));
      const a = sphere(10_000, false), b = sphere(12_000, true);
      worker.postMessage({ id: 1, a, b, seed: 42, mode }, [a.buffer, b.buffer]);
    });
  } finally { worker.terminate(); }
}

async function verifyRenderer(mode: MorphMode) {
  const gl = new WebGLRenderer({ antialias: false });
  gl.setSize(480, 360);
  document.querySelector("#renders")!.append(gl.domElement);
  const scene = new Scene();
  const camera = new PerspectiveCamera(50, 480 / 360, 0.01, 100);
  camera.position.set(0, 0, 4);
  const spark = new SparkRenderer({ renderer: gl });
  scene.add(spark);
  const data = await prepare(mode);
  const gpu = createMorphGpu(data, gl.capabilities.maxTextureSize);
  scene.add(gpu.object);
  const capture = createMorphCapture(spark, gl, scene, camera);
  let frames = 0;
  const sums: number[] = [];
  try {
    const blob = await recordMorphVideo({
      canvas: gl.domElement, duration: 0.5, signal: new AbortController().signal,
      renderFrame: async (t, signal, request, wait) => {
        gpu.update(t, 1, 1, mode, 0.65, 0.75);
        await capture.render(signal, () => {
          if (gpu.object.generatorError) throw gpu.object.generatorError;
          const context = gl.getContext();
          const pixels = new Uint8Array(480 * 360 * 4);
          context.readPixels(0, 0, 480, 360, context.RGBA, context.UNSIGNED_BYTE, pixels);
          let sum = 0;
          for (let i = 0; i < pixels.length; i += 4) sum += pixels[i] + pixels[i + 1] + pixels[i + 2];
          assert(sum > 10000, `${mode} rendered an empty frame at ${t}`);
          assert(context.getError() === context.NO_ERROR, `${mode} WebGL error`);
          sums.push(sum);
          request(); frames++;
        }, wait);
      },
    });
    assert(blob && blob.size > 0 && frames === 16, `${mode} recording failed`);
    assert(new Set(sums).size > 3, `${mode} frames did not change`);
    const video = document.createElement("video");
    video.controls = true;
    video.muted = true;
    video.src = URL.createObjectURL(blob!);
    document.querySelector("#renders")!.append(video);
    await new Promise<void>((resolve, reject) => {
      video.onloadeddata = () => resolve();
      video.onerror = () => reject(new Error(`${mode} WebM cannot be decoded`));
    });
    log(`PASS ${mode}: worker + dyno + ${frames} nonempty frames; ${blob!.size} byte WebM decoded (${video.videoWidth}x${video.videoHeight})`);
  } finally {
    await capture.dispose();
    gpu.dispose(spark);
    gl.dispose();
    gl.forceContextLoss();
  }
}

button.addEventListener("click", async () => {
  button.disabled = true;
  result.textContent = "Running…\n";
  try {
    verifyShader();
    await verifyRenderer("shape");
    await verifyRenderer("cloud");
    log("PASS all browser checks");
  } catch (error) { log(`FAIL ${error instanceof Error ? error.stack : error}`); }
});
