import { DataTexture, FloatType, NearestFilter, RGBAFormat } from "three";
import { SplatGenerator, dyno, type SparkRenderer } from "@sparkjsdev/spark";
import type { PreparedMorph } from "./morphData";
import { disposeMorphGenerator } from "./morphDispose";
import { MORPH_GLSL } from "./morphMath";
import type { MorphMode } from "./morphTypes";

/** Four RGBA32F texels per splat. Width fits WebGL2's minimum texture limit. */
const TEXTURE_WIDTH = 1024;

function attributeTexture(attributes: Float32Array, count: number) {
  const height = Math.ceil(count / (TEXTURE_WIDTH / 4));
  const padded = new Float32Array(TEXTURE_WIDTH * height * 4);
  padded.set(attributes);
  const texture = new DataTexture(padded, TEXTURE_WIDTH, height, RGBAFormat, FloatType);
  texture.minFilter = texture.magFilter = NearestFilter;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  return texture;
}

/** Spark 0.1.10 generators emit world-space Gsplats, like its snowBox example. */
export function createMorphGpu(data: PreparedMorph, maxTextureSize: number) {
  if (TEXTURE_WIDTH > maxTextureSize || Math.ceil(data.count / 256) > maxTextureSize) {
    throw new Error("Morph exceeds this GPU's texture size limit. Use smaller splat assets.");
  }
  const a = attributeTexture(data.a, data.count);
  const b = attributeTexture(data.b, data.count);
  const time = dyno.dynoFloat(0);
  const dissolve = dyno.dynoFloat(1);
  const targetBlend = dyno.dynoFloat(1);
  const mode = dyno.dynoFloat(0);
  const wave = dyno.dynoFloat(0);
  const arc = dyno.dynoFloat(0);
  const radius = dyno.dynoFloat(data.sceneRadius);
  const textureA = dyno.dynoSampler2D(a);
  const textureB = dyno.dynoSampler2D(b);
  const generator = dyno.dynoBlock({ index: "int" }, { gsplat: dyno.Gsplat }, ({ index }) => {
    const node = new dyno.Dyno({
      inTypes: { index: "int", a: "sampler2D", b: "sampler2D", t: "float", dissolve: "float", blend: "float", radius: "float", mode: "float", wave: "float", arc: "float" },
      outTypes: { gsplat: dyno.Gsplat },
      inputs: { index, a: textureA, b: textureB, t: time, dissolve, blend: targetBlend, radius, mode, wave, arc },
      globals: () => [dyno.defineGsplat, MORPH_GLSL],
      statements: ({ inputs: i, outputs: o }) => [
        `ivec2 uv = ivec2((${i.index} % 256) * 4, ${i.index} / 256);`,
        `interpolateMorph(
          texelFetch(${i.a}, uv, 0), texelFetch(${i.a}, uv + ivec2(1, 0), 0),
          texelFetch(${i.a}, uv + ivec2(2, 0), 0), texelFetch(${i.a}, uv + ivec2(3, 0), 0),
          texelFetch(${i.b}, uv, 0), texelFetch(${i.b}, uv + ivec2(1, 0), 0),
          texelFetch(${i.b}, uv + ivec2(2, 0), 0), texelFetch(${i.b}, uv + ivec2(3, 0), 0),
          ${i.t}, ${i.dissolve}, ${i.blend}, ${i.radius}, ${i.mode}, ${i.wave}, ${i.arc},
          ${o.gsplat}.center, ${o.gsplat}.scales, ${o.gsplat}.quaternion, ${o.gsplat}.rgba);`,
        `${o.gsplat}.flags = GSPLAT_FLAG_ACTIVE;`,
        `${o.gsplat}.index = ${i.index};`,
      ],
    });
    return { gsplat: node.outputs.gsplat };
  });
  const object = new SplatGenerator({ numSplats: data.count, generator });
  object.name = "Splat morph preview";
  object.raycast = () => {};
  return {
    object,
    update(t: number, amount: number, blend: number, selectedMode: MorphMode = "cloud", waveAmount = 0, arcAmount = 0) {
      const modeValue = selectedMode === "shape" ? 1 : 0;
      if (time.value === t && dissolve.value === amount && targetBlend.value === blend
        && mode.value === modeValue && wave.value === waveAmount && arc.value === arcAmount) return;
      time.value = t;
      dissolve.value = amount;
      targetBlend.value = blend;
      mode.value = modeValue;
      wave.value = waveAmount;
      arc.value = arcAmount;
      object.updateVersion();
    },
    dispose(spark?: SparkRenderer) {
      object.removeFromParent();
      disposeMorphGenerator(spark, generator, [a, b]);
    },
  };
}

export type MorphGpu = ReturnType<typeof createMorphGpu>;
