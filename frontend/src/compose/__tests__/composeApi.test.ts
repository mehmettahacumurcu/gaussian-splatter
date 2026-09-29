import { afterEach, describe, expect, it, vi } from "vitest";
import { assetFileUrl, createScene, exportScene, getAssetOrientation, saveScene, uploadAsset } from "../composeApi";
import type { SceneDoc } from "../types";

afterEach(() => vi.unstubAllGlobals());

function okJson(body: unknown) {
  return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
}

describe("compose API", () => {
  it("uploads as multipart form data", async () => {
    const fetchMock = vi.fn().mockResolvedValue(okJson({ id: "a_1" }));
    vi.stubGlobal("fetch", fetchMock);
    await uploadAsset(new File(["ply"], "statue.ply"));
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toMatch(/\/compose\/assets$/);
    expect(init.method).toBe("POST");
    expect(init.body).toBeInstanceOf(FormData);
    expect((init.body as FormData).get("file")).toBeInstanceOf(File);
  });

  it("saves scenes with PUT and a JSON body", async () => {
    const doc: SceneDoc = { version: 1, id: "s_1", name: "x", viewUp: "y", objects: [] };
    const fetchMock = vi.fn().mockResolvedValue(okJson(doc));
    vi.stubGlobal("fetch", fetchMock);
    await saveScene(doc);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toMatch(/\/compose\/scenes\/s_1$/);
    expect(init.method).toBe("PUT");
    expect(JSON.parse(init.body)).toEqual(doc);
  });

  it("creates scenes with a JSON {name, base_asset} body", async () => {
    const fetchMock = vi.fn().mockResolvedValue(okJson({}));
    vi.stubGlobal("fetch", fetchMock);
    await createScene("My scene", "scene__garden");
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toMatch(/\/compose\/scenes$/);
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ name: "My scene", base_asset: "scene__garden" });
    expect(new Headers(init.headers).get("Content-Type")).toBe("application/json");
  });

  it("starts an export with a POST", async () => {
    const fetchMock = vi.fn().mockResolvedValue(okJson({ job_id: "j1" }));
    vi.stubGlobal("fetch", fetchMock);
    await expect(exportScene("s_1")).resolves.toEqual({ job_id: "j1" });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toMatch(/\/compose\/scenes\/s_1\/export$/);
    expect(init.method).toBe("POST");
  });

  it("builds asset file urls", () => {
    expect(assetFileUrl("scene__garden")).toMatch(/\/compose\/assets\/scene__garden\/file$/);
  });

  it("fetches an asset's orientation with an encoded id", async () => {
    const body = { up: [0, 0, 1], tilt_deg: 90, plane_inlier_frac: 0.4, above_below_ratio: 3, measured: true };
    const fetchMock = vi.fn().mockResolvedValue(okJson(body));
    vi.stubGlobal("fetch", fetchMock);
    await expect(getAssetOrientation("scène/ü 1")).resolves.toEqual(body);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toMatch(/\/compose\/assets\/sc%C3%A8ne%2F%C3%BC%201\/orientation$/);
    expect(init?.method ?? "GET").toBe("GET");
  });
});
