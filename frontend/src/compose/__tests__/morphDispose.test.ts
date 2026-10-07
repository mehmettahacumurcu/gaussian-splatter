import { DataTexture, FloatType, RGBAFormat } from "three";
import { PackedSplats, SplatModifier, dyno, type SparkRenderer } from "@sparkjsdev/spark";
import { describe, expect, it, vi } from "vitest";
import { disposeMorphGenerator } from "../morphDispose";

function sourceGraph(texture: DataTexture) {
  const sampler = dyno.dynoSampler2D(texture);
  return dyno.dynoBlock({ index: "int" }, { gsplat: dyno.Gsplat }, ({ index }) => {
    const node = new dyno.Dyno({
      inTypes: { index: "int", source: "sampler2D" },
      outTypes: { gsplat: dyno.Gsplat },
      inputs: { index, source: sampler },
      globals: () => [dyno.defineGsplat],
      statements: ({ inputs: i, outputs: o }) => [
        `${o.gsplat} = Gsplat(vec3(0), GSPLAT_FLAG_ACTIVE, vec3(0.1), ${i.index}, vec4(0, 0, 0, 1), texelFetch(${i.source}, ivec2(0), 0));`,
      ],
    });
    return { gsplat: node.outputs.gsplat };
  });
}

describe("morph GPU disposal against Spark 0.1.10", () => {
  it("evicts the renderer wrapper and frees cached shader references without touching other generators", () => {
    const texture = new DataTexture(new Float32Array(16), 2, 2, RGBAFormat, FloatType);
    const source = sourceGraph(texture);
    const modifier = new SplatModifier(
      dyno.dynoBlock({ gsplat: dyno.Gsplat }, { gsplat: dyno.Gsplat }, ({ gsplat }) => ({ gsplat })),
    );
    const wrapped = modifier.apply(source);
    const unrelated = sourceGraph(texture);
    const unrelatedWrapped = modifier.apply(unrelated);
    const packed = new PackedSplats();
    const { program, material } = packed.prepareProgramMaterial(wrapped);
    const { program: otherProgram } = packed.prepareProgramMaterial(unrelatedWrapped);
    program.update();
    expect(program.shader).toContain("texelFetch");
    expect(Object.values(program.uniforms).some((uniform) => uniform.value === texture)).toBe(true);
    const disposeMaterial = vi.spyOn(material, "dispose");
    const disposeTexture = vi.spyOn(texture, "dispose");

    disposeMorphGenerator({ modifier } as unknown as SparkRenderer, source, [texture]);

    expect(disposeMaterial).toHaveBeenCalledOnce();
    expect(disposeTexture).toHaveBeenCalledOnce();
    expect(texture.image.data?.byteLength).toBe(0);
    expect(modifier.cache.has(source)).toBe(false);
    expect(PackedSplats.generatorProgram.has(wrapped)).toBe(false);
    expect(program.updaters).toHaveLength(0);
    expect(program.uniforms).toEqual({});
    expect(material.uniforms).toEqual({});
    expect(modifier.cache.get(unrelated)).toBe(unrelatedWrapped);
    expect(PackedSplats.generatorProgram.get(unrelatedWrapped)).toBe(otherProgram);

    disposeMorphGenerator({ modifier } as unknown as SparkRenderer, unrelated, []);
    packed.dispose();
  });

  it("releases textures before the generator has rendered, without a renderer", () => {
    const texture = new DataTexture(new Float32Array(16), 2, 2, RGBAFormat, FloatType);
    const source = sourceGraph(texture);
    disposeMorphGenerator(undefined, source, [texture]);
    expect(texture.image.data?.byteLength).toBe(0);
  });
});
