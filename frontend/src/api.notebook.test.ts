import { afterEach, describe, expect, it, vi } from "vitest";

import { generateStaticNotebook, getStaticNotebookPresets } from "./api";
import type { StaticNotebookRunSpec } from "./notebook/types";

const SPEC: StaticNotebookRunSpec = {
  schema_version: 1,
  input_folder: "captures/room",
  frame_selection: { mode: "smart", fixed_fps: 4 },
  quality: {
    profile: "balanced_l4",
    n_iters: null,
    max_gaussians: null,
    advanced: {},
  },
  publish: { replace_owned_result: true },
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("notebook API", () => {
  it("posts only JSON and parses filename star", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(new Blob(["{}"]), {
        status: 200,
        headers: {
          "Content-Disposition": "attachment; filename*=UTF-8'en'room%20scan.ipynb",
        },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const artifact = await generateStaticNotebook(SPEC);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toMatch(/\/notebooks\/static$/);
    expect(init.method).toBe("POST");
    expect(init.body).toBe(JSON.stringify(SPEC));
    expect(artifact.filename).toBe("room scan.ipynb");
    expect(artifact.blob.size).toBeGreaterThan(0);
  });

  it("gets backend-owned presets", async () => {
    const payload = { schema_version: 1, default_profile: "balanced_l4" };
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(payload), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    await expect(getStaticNotebookPresets()).resolves.toEqual(payload);
    expect(fetchMock.mock.calls[0][0]).toMatch(/\/notebooks\/static\/presets$/);
  });

  it("sanitizes a path-like attachment filename", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response("x", {
          status: 200,
          headers: {
            "Content-Disposition": 'attachment; filename="../evil\\name.ipynb"',
          },
        }),
      ),
    );
    const artifact = await generateStaticNotebook(SPEC);
    expect(artifact.filename).not.toMatch(/[\\/]/);
  });
});
