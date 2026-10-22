import { describe, expect, it } from "vitest";
import { driveRelative, formatCount, formatDuration, formatGb, notebookSpec, parseCount } from "../format";
import type { SpirulaSettings } from "../spirulaApi";

const SETTINGS: SpirulaSettings = {
  preset: "quality", iterations: null, cap_max: 6_000_000, fps: 4, sfm_quality: null, geometry_model: "moge2-vitb",
  generate_depth: true, appearance: false, floater_suppression: "mild", resolution_divisor: 2, allow_partial: false,
};

describe("format helpers", () => {
  it("formats sizes, counts and durations", () => {
    expect(formatGb(1458)).toBe("1.4 GB");
    expect(formatGb(46000)).toBe("45 GB");
    expect(formatCount(3_000_000)).toBe("3M");
    expect(formatCount(2_500_000)).toBe("2.5M");
    expect(formatCount(450_000)).toBe("450K");
    expect(formatDuration(30)).toBe("~1 dk");
    expect(formatDuration(54 * 60)).toBe("~54 dk");
    expect(formatDuration(150 * 60)).toBe("~2 sa 30 dk");
  });

  it("parses splat counts the way people type them", () => {
    expect(parseCount("50m")).toBe(50_000_000);
    expect(parseCount("3.5M")).toBe(3_500_000);
    expect(parseCount("500k")).toBe(500_000);
    expect(parseCount("2500000")).toBe(2_500_000);
    expect(parseCount("lots")).toBeNull();
  });

  it("maps a path in the Drive folder to its MyDrive-relative path", () => {
    expect(driveRelative("G:\\Drive'ım\\GaussianTests\\inputs\\oda.MOV", "G:\\Drive'ım")).toBe("GaussianTests/inputs/oda.MOV");
    expect(driveRelative('"g:\\Drive’ım\\a\\b"', "G:\\Drive'ım\\")).toBe("a/b");
    expect(driveRelative("E:\\captures\\oda.MOV", "G:\\Drive'ım")).toBeNull();
    expect(driveRelative("G:\\Drive'ımX\\a.MOV", "G:\\Drive'ım")).toBeNull();
  });

  it("builds the notebook spec from the same settings", () => {
    const spec = notebookSpec("video", "GaussianTests/oda.MOV", SETTINGS);
    expect(spec).toMatchObject({
      pipeline: "spirula", input_mode: "video", input_path: "GaussianTests/oda.MOV", preset: "quality",
      max_gaussians: 6_000_000, appearance: false, floater_suppression: "mild", resolution_divisor: 2, sfm_quality: null,
    });
    expect(notebookSpec("dataset", "x/ds.zip", SETTINGS).input_mode).toBe("dataset_zip");
    expect(notebookSpec("images", "x/photos", SETTINGS).input_mode).toBe("dataset_folder");
  });
});
