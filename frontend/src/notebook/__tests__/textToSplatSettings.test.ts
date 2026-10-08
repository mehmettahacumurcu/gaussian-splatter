import { describe, expect, it } from "vitest";
import { GPU_PRESETS, MAX_DETAIL, estimateTextRequirements, restoreTextToSplatSettings } from "../textToSplatSettings";

describe("mesh recipe settings", () => {
  it("loads old saved settings and ignores unknown selectors without crashing", () => {
    const legacy = restoreTextToSplatSettings({ gpu_preset: "a100", reconstruction_model: "trellis2", mesh_views: 72 });
    expect(legacy).toMatchObject({ trellis2_pipeline_type: "512", mesh_sh_degree: 2, mesh_render_resolution: 1024, mesh_views: 72 });
    expect(restoreTextToSplatSettings(JSON.parse('{"gpu_preset":"old_gpu","image_model":"old_model","trellis2_pipeline_type":"old","unknown":10}'))).toEqual(GPU_PRESETS.l4.settings);
  });
  it("matches backend admission estimates and avoids a misleading fallback when fitting is too large", () => {
    const small = { ...GPU_PRESETS.a100.settings, image_model: "sdxl" as const, reconstruction_model: "trellis2" as const };
    for (const [pipeline, minimum] of [["512", 28], ["1024_cascade", 40], ["1536_cascade", 60]] as const) {
      expect(estimateTextRequirements({ ...small, trellis2_pipeline_type: pipeline }, "baseline").minimum).toBe(minimum);
    }
    const maximum = { ...small, mesh_splat_cap: 3000000, mesh_render_resolution: 2048, mesh_sh_degree: 3 };
    expect(estimateTextRequirements(maximum, "baseline").minimum).toBe(58);
    expect(estimateTextRequirements({ ...maximum, gpu_preset: "l4" }, "baseline").lower).toBeUndefined();
    expect(GPU_PRESETS.rtx_pro_6000.settings).toMatchObject(MAX_DETAIL);
  });
});
