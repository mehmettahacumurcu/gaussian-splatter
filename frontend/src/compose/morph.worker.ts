import { prepareMorph } from "./morphData";
import type { PreparedMorph } from "./morphData";

export interface MorphWorkerRequest {
  id: number;
  a: Float32Array;
  b: Float32Array;
  seed: number;
}

export type MorphWorkerResponse =
  | { id: number; ok: true; result: PreparedMorph }
  | { id: number; ok: false; error: string };

// Use a local worker surface to avoid adding WebWorker/DOM duplicate lib types.
const workerScope = self as unknown as {
  onmessage: ((event: MessageEvent<MorphWorkerRequest>) => void) | null;
  postMessage(message: MorphWorkerResponse, transfer?: Transferable[]): void;
};

workerScope.onmessage = ({ data }) => {
  try {
    const result = prepareMorph(data.a, data.b, data.seed);
    workerScope.postMessage({ id: data.id, ok: true, result }, [result.a.buffer, result.b.buffer]);
  } catch (error) {
    workerScope.postMessage({
      id: data.id,
      ok: false,
      error: error instanceof Error ? error.message : "Could not prepare the morph.",
    });
  }
};
