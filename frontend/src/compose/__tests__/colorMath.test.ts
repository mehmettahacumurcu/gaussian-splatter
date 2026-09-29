import { describe, expect, it } from "vitest";
import { hexToTint, tintToHex } from "../colorMath";

describe("tint hex helpers", () => {
  it("round-trips representable tints", () => {
    expect(tintToHex([1, 0.5, 0])).toBe("#ff8000");
    const back = hexToTint("#ff8000");
    expect(back[0]).toBe(1);
    expect(back[1]).toBeCloseTo(0.5, 2);
    expect(back[2]).toBe(0);
  });
  it("clamps to [0,1] and treats non-finite as 1", () => {
    expect(tintToHex([2, -1, 0])).toBe("#ff0000");
    expect(tintToHex([NaN, Infinity, 0])).toBe("#ffff00");
  });
});
