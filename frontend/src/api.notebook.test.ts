import { afterEach, describe, expect, it, vi } from "vitest";

import { generatePipelineNotebook, generateStaticNotebook, getStaticNotebookPresets } from "./api";
import type { TextToSplatNotebookSpec } from "./notebook/PipelineNotebookPanel";
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
  it("posts text-to-splat through the shared pipeline endpoint as literal JSON", async () => {
    const spec: TextToSplatNotebookSpec = {
      pipeline: "text_to_splat", input_mode: "text", input_path: "", preset: "baseline",
      prompt: 'a "red" car\nİstanbul 🚗; print("literal")', negative_prompt: "", style: "", seed: 42,
      output_dir: null, target_splat_count: null,
      image_model: "sdxl", background_model: "u2net", reconstruction_model: "trellis", gpu_preset: "l4",
      image_steps: 25, image_guidance: 7, image_resolution: 1024, trellis_seed: 42,
      sparse_steps: 12, sparse_cfg: 7.5, slat_steps: 12, slat_cfg: 3,
      mesh_views: 24, mesh_fit_iterations: 1500, mesh_splat_cap: 50000,
    };
    const fetchMock = vi.fn().mockResolvedValue(new Response("{}", { status: 200, headers: {
      "Content-Disposition": 'attachment; filename="car_text_to_splat.ipynb"',
    } }));
    vi.stubGlobal("fetch", fetchMock);
    const artifact = await generatePipelineNotebook(spec);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toMatch(/\/notebooks\/static\/pipeline$/);
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual(spec);
    expect(artifact.filename).toBe("car_text_to_splat.ipynb");
  });

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
