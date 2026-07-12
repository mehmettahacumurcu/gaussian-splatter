export type DriveFolderResolution =
  | {
      ok: true;
      canonicalInput: string;
      displayInput: string;
      displayOutput: string;
    }
  | { ok: false; error: string };

const CONTROL = /[\u0000-\u001f\u007f]/;
const DRIVE_LETTER = /^[A-Za-z]:/;

export function resolveDriveFolder(raw: string): DriveFolderResolution {
  let value = raw.trim();
  if (value === "MyDrive") {
    return { ok: false, error: "Choose a folder below MyDrive." };
  }
  if (value.startsWith("MyDrive/")) value = value.slice("MyDrive/".length);
  if (!value || value.includes("\\") || CONTROL.test(value)) {
    return { ok: false, error: "Enter a MyDrive-relative folder." };
  }
  if (value.startsWith("/") || value.endsWith("/") || DRIVE_LETTER.test(value)) {
    return { ok: false, error: "Do not enter an absolute or local computer path." };
  }
  const parts = value.split("/");
  if (parts.some((part) => part === "" || part === "." || part === "..")) {
    return { ok: false, error: "Remove empty, dot, or traversal path segments." };
  }
  if (parts.at(-1)?.endsWith("_result")) {
    return { ok: false, error: "Choose the input folder, not a result folder." };
  }
  const canonicalInput = parts.join("/");
  const outputParts = [...parts];
  outputParts[outputParts.length - 1] += "_result";
  return {
    ok: true,
    canonicalInput,
    displayInput: `MyDrive/${canonicalInput}`,
    displayOutput: `MyDrive/${outputParts.join("/")}`,
  };
}
