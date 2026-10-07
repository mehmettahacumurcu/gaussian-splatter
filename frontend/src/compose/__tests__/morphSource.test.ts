import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { PackedSplats, SplatMesh } from "@sparkjsdev/spark";
import { Color, Quaternion, Vector3 } from "three";
import { MORPH_STRIDE } from "../morphData";
import { extractMorphSource } from "../morphSource";
import type { SceneObject, Vec3 } from "../types";

// Spark's module initializes its embedded WASM asynchronously. No renderer,
// network asset, or mocked packing/decoding API is needed for these tests.
beforeAll(async () => { await SplatMesh.staticInitialized; });
afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

function object(overrides: Partial<SceneObject> = {}): SceneObject {
  return {
    id: "source", kind: "splat", asset: "source.ply", name: "Source", role: "object", visible: true,
    transform: { position: [0, 0, 0], quaternion: [0, 0, 0, 1], scale: 1 },
    ...overrides,
  };
}

function source(positions: Vec3[], quaternion = new Quaternion(), color = new Color(0.2, 0.4, 0.8)): PackedSplats {
  const packed = new PackedSplats();
  for (const position of positions) {
    packed.pushSplat(new Vector3(...position), new Vector3(0.1, 0.2, 0.3), quaternion, 0.7, color);
  }
  return packed;
}

function decodedRecord(packed: PackedSplats, index = 0): Float32Array {
  // Actual Spark getSplat returns reused scratch objects: copy immediately.
  const splat = packed.getSplat(index);
  return new Float32Array([
    ...splat.center.toArray(), splat.opacity,
    ...splat.scales.toArray(), 0,
    ...splat.quaternion.toArray(),
    ...splat.color.toArray(), 0,
  ]);
}

describe("morph source extraction with Spark PackedSplats", () => {
  it("applies oriented crop in raw object coordinates before world rotation, scale, and translation", async () => {
    const packed = source([[3.5, 0.25, 0], [2, 1, 0], [2, 0, 1]]);
    const halfTurn = Math.SQRT1_2;
    const transformed = object({
      transform: { position: [100, 200, 300], quaternion: [0, 0, halfTurn, halfTurn], scale: 3 },
      crop: { center: [2, 0, 0], halfSize: [0.5, 2, 0.5], quaternion: [0, 0, halfTurn, halfTurn] },
    });
    const result = await extractMorphSource(packed, transformed, new AbortController().signal);
    expect(result.length).toBe(MORPH_STRIDE);
    expect(result[0]).toBeCloseTo(99.25);
    expect(result[1]).toBeCloseTo(210.5);
    expect(result[2]).toBe(300);
  });

  it("includes splats exactly on crop boundaries and handles an entirely empty crop", async () => {
    const packed = source([[1, 0, 0], [-1, 0, 0], [1.25, 0, 0]]);
    const boundary = object({ crop: { center: [0, 0, 0], halfSize: [1, 1, 1], quaternion: [0, 0, 0, 1] } });
    const result = await extractMorphSource(packed, boundary, new AbortController().signal);
    expect(result.length).toBe(2 * MORPH_STRIDE);
    expect([result[0], result[MORPH_STRIDE]]).toEqual([1, -1]);
    const empty = object({ crop: { center: [100, 100, 100], halfSize: [0.1, 0.1, 0.1], quaternion: [0, 0, 0, 1] } });
    expect(await extractMorphSource(packed, empty, new AbortController().signal)).toHaveLength(0);
    expect(await extractMorphSource(new PackedSplats(), object(), new AbortController().signal)).toHaveLength(0);
  });

  it("transforms positions, uniform scales, and rotation in world-times-local order", async () => {
    const localRotation = new Quaternion().setFromAxisAngle(new Vector3(1, 0, 0), Math.PI / 2);
    const packed = source([[1, 2, 3]], localRotation);
    const raw = decodedRecord(packed);
    const result = await extractMorphSource(packed, object({
      transform: { position: [10, 20, 30], quaternion: [0, 0, Math.SQRT1_2, Math.SQRT1_2], scale: 2 },
    }), new AbortController().signal);
    expect(Array.from(result.slice(0, 3))).toEqual([6, 22, 36]);
    expect(result[3]).toBe(raw[3]);
    for (let axis = 0; axis < 3; axis++) expect(result[4 + axis]).toBeCloseTo(raw[4 + axis] * 2, 6);
    // A 90-degree local X rotation followed by a 90-degree world Z rotation
    // has quaternion [0.5, 0.5, 0.5, 0.5]; reversed composition has negative Y.
    for (let component = 8; component < 12; component++) expect(result[component]).toBeCloseTo(0.5, 2);
    expect(Math.hypot(...result.slice(8, 12))).toBeCloseTo(1, 6);
  });

  it("applies exposure and tint before saturation, preserving unclamped DC values", async () => {
    const packed = source([[0, 0, 0]]);
    const raw = decodedRecord(packed);
    const grayscale = await extractMorphSource(packed, object({
      color: { exposure: 1, tint: [0.5, 1, 1.5], saturation: 0 },
    }), new AbortController().signal);
    const expectedLuma = 0.2126 * raw[12] + 0.7152 * raw[13] * 2 + 0.0722 * raw[14] * 3;
    for (let channel = 12; channel < 15; channel++) expect(grayscale[channel]).toBeCloseTo(expectedLuma, 6);
    const bright = await extractMorphSource(packed, object({
      color: { exposure: 2, tint: [1, 1, 1], saturation: 1 },
    }), new AbortController().signal);
    for (let channel = 12; channel < 15; channel++) expect(bright[channel]).toBeCloseTo(raw[channel] * 4, 6);
    expect(bright[14]).toBeGreaterThan(1);
    const identity = await extractMorphSource(packed, object({ color: null }), new AbortController().signal);
    expect(Array.from(identity.slice(12, 15))).toEqual(Array.from(raw.slice(12, 15)));
  });

  it("does not mutate shared packed source data or scene settings", async () => {
    const packed = source([[1, 2, 3], [4, 5, 6]]);
    const originalWords = packed.packedArray!.slice();
    const originalRecords = [decodedRecord(packed, 0), decodedRecord(packed, 1)];
    const settings = object({
      transform: { position: [10, 20, 30], quaternion: [0, 0, Math.SQRT1_2, Math.SQRT1_2], scale: 3 },
      color: { exposure: 2, tint: [0.5, 1, 1], saturation: 0.4 },
    });
    const serializedSettings = JSON.stringify(settings);
    const first = await extractMorphSource(packed, settings, new AbortController().signal);
    const second = await extractMorphSource(packed, settings, new AbortController().signal);
    expect(first).toEqual(second);
    expect(first.buffer).not.toBe(packed.packedArray!.buffer);
    expect(packed.packedArray).toEqual(originalWords);
    expect([decodedRecord(packed, 0), decodedRecord(packed, 1)]).toEqual(originalRecords);
    expect(JSON.stringify(settings)).toBe(serializedSettings);
  });

  it("waits for initialization and rejects pre-aborted extraction without decoding", async () => {
    let finishInitialization!: () => void;
    const initialized = new Promise<void>((resolve) => { finishInitialization = resolve; });
    const packed = new PackedSplats({ construct: () => initialized });
    const getSplat = vi.spyOn(packed, "getSplat");
    const controller = new AbortController();
    const reason = new Error("Cancelled source selection");
    controller.abort(reason);
    const pending = extractMorphSource(packed, object(), controller.signal);
    const rejected = expect(pending).rejects.toBe(reason);
    finishInitialization();
    await rejected;
    expect(getSplat).not.toHaveBeenCalled();
  });

  it("yields between decode batches so cancellation stops further work", async () => {
    const packed = source(Array.from({ length: 5000 }, (_, i): Vec3 => [i % 16, 0, 0]));
    const getSplat = vi.spyOn(packed, "getSplat");
    const controller = new AbortController();
    vi.useFakeTimers();
    const pending = extractMorphSource(packed, object(), controller.signal);
    await Promise.resolve();
    expect(getSplat).toHaveBeenCalledTimes(4096);
    const reason = new Error("Morph pair changed");
    controller.abort(reason);
    const rejected = expect(pending).rejects.toBe(reason);
    await vi.runAllTimersAsync();
    await rejected;
    expect(getSplat).toHaveBeenCalledTimes(4096);
  });
});
