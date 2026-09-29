import { describe, expect, it } from "vitest";
import { errorMessage } from "../errorMessage";

describe("errorMessage", () => {
  it("returns the string detail of a FastAPI error", () => {
    const e = new Error('HTTP 409 Conflict: {"detail":"Scene \\"x\\" already exists"}');
    expect(errorMessage(e)).toBe('Scene "x" already exists');
  });

  it("joins 422 validation details with their body location", () => {
    const body = {
      detail: [
        { loc: ["body", "objects", 1, "transform", "scale"], msg: "Input should be greater than 0", type: "greater_than" },
        { loc: ["body", "name"], msg: "Field required", type: "missing" },
      ],
    };
    const e = new Error(`HTTP 422 Unprocessable Entity: ${JSON.stringify(body)}`);
    expect(errorMessage(e)).toBe("objects.1.transform.scale: Input should be greater than 0; name: Field required");
  });

  it("keeps the message when the body is not JSON or has no detail", () => {
    expect(errorMessage(new Error("HTTP 500 Internal Server Error: boom"))).toBe("HTTP 500 Internal Server Error: boom");
    expect(errorMessage(new Error('HTTP 502 Bad Gateway: {"error":"x"}'))).toBe('HTTP 502 Bad Gateway: {"error":"x"}');
    expect(errorMessage(new Error("HTTP 400 Bad: {not json"))).toBe("HTTP 400 Bad: {not json");
  });

  it("handles non-Error values", () => {
    expect(errorMessage("Failed to fetch")).toBe("Failed to fetch");
  });
});
