import { afterEach, describe, expect, it, vi } from "vitest";
import { assetFileUrl, saveScene, uploadAsset } from "../composeApi";
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

  it("builds asset file urls", () => {
    expect(assetFileUrl("scene__garden")).toMatch(/\/compose\/assets\/scene__garden\/file$/);
  });
});
