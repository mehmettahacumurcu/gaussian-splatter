import { describe, expect, it } from "vitest";

import { resolveDriveFolder } from "../drivePath";

describe("resolveDriveFolder", () => {
  it.each(["captures/myroom", "MyDrive/captures/myroom"])(
    "normalizes %s",
    (raw) => {
      expect(resolveDriveFolder(raw)).toEqual({
        ok: true,
        canonicalInput: "captures/myroom",
        displayInput: "MyDrive/captures/myroom",
        displayOutput: "MyDrive/captures/myroom_result",
      });
    },
  );

  it.each([
    "",
    "MyDrive",
    "/content/drive/MyDrive/x",
    "C:/x",
    "a\\b",
    "a/../b",
    "a/./b",
    "a//b",
    "a/x_result",
    "a/b/",
  ])("rejects %s", (raw) => {
    expect(resolveDriveFolder(raw).ok).toBe(false);
  });
});
