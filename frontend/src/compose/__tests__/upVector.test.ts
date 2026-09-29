import { describe, expect, it } from "vitest";
import { effectiveUp, normalizeUp, upKey } from "../upVector";
import type { SceneDoc } from "../types";

function doc(patch: Partial<SceneDoc>): SceneDoc {
  return { version: 1, id: "s", name: "x", viewUp: "y", objects: [], ...patch };
}

describe("effectiveUp", () => {
  it("falls back to ±Y from viewUp", () => {
    expect(effectiveUp(doc({}))).toEqual([0, 1, 0]);
    expect(effectiveUp(doc({ viewUp: "-y" }))).toEqual([0, -1, 0]);
    expect(effectiveUp(doc({ viewUp: "-y", up: null }))).toEqual([0, -1, 0]);
  });

  it("prefers the explicit up vector over viewUp", () => {
    const up: [number, number, number] = [-0.94, -0.11, -0.32];
    const n = Math.hypot(...up);
    const unit = up.map((v) => v / n) as [number, number, number];
    expect(effectiveUp(doc({ viewUp: "-y", up: unit }))).toEqual(unit);
  });

  it("ignores a degenerate stored up", () => {
    expect(effectiveUp(doc({ viewUp: "-y", up: [0, 0, 0] }))).toEqual([0, -1, 0]);
    expect(effectiveUp(doc({ up: [NaN, 1, 0] }))).toEqual([0, 1, 0]);
  });
});

describe("normalizeUp", () => {
  it("normalises finite non-zero vectors", () => {
    const n = normalizeUp([0, 0, 2]);
    expect(n).toEqual([0, 0, 1]);
    const m = normalizeUp([1, 1, 0])!;
    expect(Math.hypot(...m)).toBeCloseTo(1, 12);
  });

  it("rejects zero, tiny and non-finite vectors", () => {
    expect(normalizeUp([0, 0, 0])).toBeNull();
    expect(normalizeUp([1e-12, 0, 0])).toBeNull();
    expect(normalizeUp([Infinity, 0, 0])).toBeNull();
    expect(normalizeUp([NaN, 1, 0])).toBeNull();
  });
});

describe("upKey", () => {
  it("is stable for equal vectors and differs for different ones", () => {
    expect(upKey([0, 1, 0])).toBe(upKey([0, 1, 0]));
    expect(upKey([0, 1, 0])).not.toBe(upKey([0, -1, 0]));
  });
});
