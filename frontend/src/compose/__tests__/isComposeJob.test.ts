import { describe, expect, it } from "vitest";
import { isComposeJob } from "../isComposeJob";

describe("isComposeJob", () => {
  it("detects compose export jobs by their scene prefix", () => {
    expect(isComposeJob({ scene: "compose-bahce" })).toBe(true);
    expect(isComposeJob({ scene: "garden" })).toBe(false);
    expect(isComposeJob({ scene: "my-compose-scene" })).toBe(false);
  });
});
