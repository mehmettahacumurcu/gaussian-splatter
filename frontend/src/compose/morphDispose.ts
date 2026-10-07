import type { DataTexture } from "three";
import { PackedSplats, dyno, type GsplatGenerator, type SparkRenderer } from "@sparkjsdev/spark";

/**
 * Spark 0.1.10 caches the renderer's wrapped generator, not the source graph.
 * Its private program-to-material cache has no eviction API. Dispose the GPU
 * material and sever its graph/uniform references so only small shader metadata
 * remains there, rather than retaining the preview's large attribute buffers.
 */
export function disposeMorphGenerator(
  spark: SparkRenderer | undefined,
  generator: GsplatGenerator,
  textures: readonly DataTexture[],
) {
  // Version-specific access is confined here; there is no public dispose API.
  const modifier = (spark as unknown as {
    modifier?: { cache: Map<GsplatGenerator, GsplatGenerator> };
  } | undefined)?.modifier;
  const wrapped = modifier?.cache.get(generator);
  modifier?.cache.delete(generator);
  const keys = wrapped ? [generator, wrapped] : [generator];
  for (const key of keys) {
    const program = PackedSplats.generatorProgram.get(key);
    if (!program) continue;
    program.prepareMaterial().dispose();
    PackedSplats.generatorProgram.delete(key);
    program.updaters.length = 0;
    for (const name of Object.keys(program.uniforms)) delete program.uniforms[name];
    program.graph = dyno.dynoBlock({}, {}, () => ({}));
  }
  for (const texture of textures) {
    texture.dispose();
    // Dyno sampler closures may outlive the preview in Spark's internal caches.
    texture.image.data = new Float32Array(0);
  }
}
