import * as THREE from "three";
import { beforeEach, describe, expect, it, vi } from "vitest";

// Stub Spark: no WASM/WebGL in jsdom, and we only need to observe construction/disposal.
const { constructed, FakePackedSplats } = vi.hoisted(() => {
  const constructed: InstanceType<typeof FakePackedSplats>[] = [];
  class FakePackedSplats {
    options: { url?: string; splatEncoding?: unknown };
    extra: Record<string, unknown> = {};
    dispose = vi.fn();
    constructor(options: { url?: string; splatEncoding?: unknown }) {
      this.options = options;
      constructed.push(this);
    }
  }
  return { constructed, FakePackedSplats };
});
vi.mock("@sparkjsdev/spark", () => ({ PackedSplats: FakePackedSplats, LN_SCALE_MIN: -12, LN_SCALE_MAX: 9 }));

import {
  COMPOSE_SPLAT_ENCODING,
  RefCountedCache,
  disposeSplatMeshOwn,
  packedSplatsCache,
} from "../splatCache";

describe("RefCountedCache", () => {
  it("loads once per key, shares the value and disposes it when the last user releases", () => {
    const load = vi.fn((key: string) => ({ key }));
    const dispose = vi.fn();
    const cache = new RefCountedCache(load, dispose);

    const a1 = cache.acquire("a");
    const a2 = cache.acquire("a");
    const b = cache.acquire("b");
    expect(a1).toBe(a2);
    expect(b).not.toBe(a1);
    expect(load.mock.calls.map((c) => c[0])).toEqual(["a", "b"]);
    expect(cache.refCount("a")).toBe(2);

    cache.release("a");
    expect(dispose).not.toHaveBeenCalled();
    cache.release("a");
    expect(dispose).toHaveBeenCalledTimes(1);
    expect(dispose).toHaveBeenCalledWith(a1);
    expect(cache.refCount("a")).toBe(0);

    // A later acquire loads a fresh value.
    const a3 = cache.acquire("a");
    expect(a3).not.toBe(a1);
    expect(load).toHaveBeenCalledTimes(3);
  });

  it("ignores releases of unknown or already released keys", () => {
    const dispose = vi.fn();
    const cache = new RefCountedCache((k: string) => k, dispose);
    cache.release("nope");
    cache.acquire("a");
    cache.release("a");
    cache.release("a");
    expect(dispose).toHaveBeenCalledTimes(1);
    expect(cache.size).toBe(0);
  });

  it("does not cache a value whose load throws", () => {
    let fail = true;
    const cache = new RefCountedCache((k: string) => {
      if (fail) throw new Error("boom");
      return k;
    }, vi.fn());
    expect(() => cache.acquire("a")).toThrow("boom");
    expect(cache.size).toBe(0);
    fail = false;
    expect(cache.acquire("a")).toBe("a");
    expect(cache.refCount("a")).toBe(1);
  });
});

describe("packedSplatsCache", () => {
  beforeEach(() => {
    constructed.length = 0;
  });

  it("builds one PackedSplats per URL with the composer encoding and frees it (plus SH textures) on last release", () => {
    const p1 = packedSplatsCache.acquire("/assets/x.ply");
    const p2 = packedSplatsCache.acquire("/assets/x.ply");
    expect(p1).toBe(p2);
    expect(constructed).toHaveLength(1);
    expect(constructed[0].options).toEqual({ url: "/assets/x.ply", splatEncoding: COMPOSE_SPLAT_ENCODING });

    const sh1 = new THREE.Texture();
    const sh1Dispose = vi.spyOn(sh1, "dispose");
    constructed[0].extra.sh1Texture = { value: sh1 };

    packedSplatsCache.release("/assets/x.ply");
    expect(constructed[0].dispose).not.toHaveBeenCalled();
    packedSplatsCache.release("/assets/x.ply");
    expect(constructed[0].dispose).toHaveBeenCalledTimes(1);
    expect(sh1Dispose).toHaveBeenCalledTimes(1);
  });

  it("widens the base-colour range so over-bright colours survive until the colour matrix", () => {
    expect(COMPOSE_SPLAT_ENCODING.rgbMin).toBe(-0.5);
    expect(COMPOSE_SPLAT_ENCODING.rgbMax).toBe(1.5);
    // Everything else stays at Spark's defaults.
    expect(COMPOSE_SPLAT_ENCODING).toMatchObject({ lnScaleMin: -12, lnScaleMax: 9, sh1Min: -1, sh1Max: 1 });
  });
});

describe("disposeSplatMeshOwn", () => {
  it("frees the mesh's edit SDF texture but leaves the shared PackedSplats alone", () => {
    const sdfTexture = new THREE.Texture();
    const sdfDispose = vi.spyOn(sdfTexture, "dispose");
    const packedDispose = vi.fn();
    const meshDispose = vi.fn();
    const mesh = {
      packedSplats: { dispose: packedDispose, extra: {} },
      rgbaDisplaceEdits: { sdfTexture },
      dispose: meshDispose,
    };
    disposeSplatMeshOwn(mesh as never);
    expect(sdfDispose).toHaveBeenCalledTimes(1);
    expect(packedDispose).not.toHaveBeenCalled();
    expect(meshDispose).not.toHaveBeenCalled();
  });
});
