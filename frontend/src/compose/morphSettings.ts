import type { MorphSettings, SceneObject } from "./types";

/** Copy only the persisted fields, never runtime controls or worker buffers. */
export function pickMorphSettings(state: MorphSettings): MorphSettings {
  const { sourceId, targetId, mode, duration, dissolve, wave, arc, targetBlend, seed, autoAlign } = state;
  return { sourceId, targetId, mode, duration, dissolve, wave, arc, targetBlend, seed, autoAlign };
}

/** Mirrors backend validation, including partial selections and object references. */
export function validateMorphSettings(morph: MorphSettings, objects: readonly SceneObject[]): boolean {
  if (morph.mode !== "cloud" && morph.mode !== "shape") return false;
  if (typeof morph.autoAlign !== "boolean") return false;
  if (!Number.isFinite(morph.duration) || morph.duration < 0.5 || morph.duration > 120) return false;
  if (![morph.dissolve, morph.wave, morph.arc, morph.targetBlend].every(
    (value) => Number.isFinite(value) && value >= 0 && value <= 1,
  )) return false;
  if (!Number.isInteger(morph.seed) || morph.seed < 0 || morph.seed > 4294967295) return false;
  if (morph.sourceId !== null && morph.sourceId === morph.targetId) return false;
  return [morph.sourceId, morph.targetId].every(
    (id) => id === null || objects.some((object) => object.id === id && object.kind === "splat"),
  );
}

export function sameMorphSettings(a: MorphSettings | null | undefined, b: MorphSettings | null): boolean {
  if (!a || !b) return (a ?? null) === b;
  return (Object.keys(pickMorphSettings(b)) as (keyof MorphSettings)[]).every((key) => a[key] === b[key]);
}
