import type { ColorAdjust, Vec3 } from "./types";

const LUMA: Vec3 = [0.2126, 0.7152, 0.0722]; // Rec.709, same as backend bake

/** Row-major 3×3 M = S · diag(gain) (matches backend bake.color_matrix). */
export function colorMatrix(c: ColorAdjust): number[] {
  const g = 2 ** c.exposure;
  const gain = c.tint.map((t) => t * g);
  const m: number[] = [];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      const s = LUMA[j] + c.saturation * ((i === j ? 1 : 0) - LUMA[j]);
      m.push(s * gain[j]);
    }
  }
  return m;
}

export function applyColor(c: ColorAdjust, rgb: Vec3): Vec3 {
  const m = colorMatrix(c);
  return [0, 1, 2].map((i) => m[3 * i] * rgb[0] + m[3 * i + 1] * rgb[1] + m[3 * i + 2] * rgb[2]) as Vec3;
}

/**
 * The hex form is lossy: 8 bits per channel, and values are clamped to [0, 1]
 * (non-finite components are treated as 1).
 */
export function tintToHex(t: Vec3): string {
  const ch = (v: number) => {
    const x = Number.isFinite(v) ? v : 1;
    return Math.round(Math.min(Math.max(x, 0), 1) * 255).toString(16).padStart(2, "0");
  };
  return `#${t.map(ch).join("")}`;
}

export function hexToTint(hex: string): Vec3 {
  const n = parseInt(hex.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}
