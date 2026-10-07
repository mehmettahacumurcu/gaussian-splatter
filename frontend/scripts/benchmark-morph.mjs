// Node 22.18+/24: node --expose-gc scripts/benchmark-morph.mjs
// Times include validation, robust bounds, correspondence, padding and output
// attribute copies. Input generation and travel measurement are excluded.
import { performance } from "node:perf_hooks";
import os from "node:os";
import { prepareMorph } from "../src/compose/morphData.ts";
import { noisySphere, meanNormalizedTravel, nearestNeighborLowerBound } from "../src/compose/__tests__/fixtures/morphSpheres.ts";

console.log(JSON.stringify({ node: process.version, cpu: os.cpus()[0]?.model, fixture: "Independent random sphere surfaces; ±1% radial noise; seeds 12345/67890", units: "Each scene centered and divided by its longest 2nd–98th percentile dimension", timing: "End-to-end prepareMorph; one 10k warmup; median of three timed runs" }));
const warmA = noisySphere(10_000, 12345);
const warmB = noisySphere(10_000, 67890);
for (const mode of ["cloud", "shape"]) prepareMorph(warmA, warmB, 42, mode);
for (const [countA, countB] of [[100_000, 100_000], [1_000_000, 1_000_000], [100_000, 120_000]]) {
  const a = noisySphere(countA, 12345);
  const b = noisySphere(countB, 67890);
  for (const mode of ["cloud", "shape"]) {
    const times = [];
    let travel = 0;
    for (let run = 0; run < 3; run++) {
      global.gc?.();
      const start = performance.now();
      const prepared = prepareMorph(a, b, 42, mode);
      times.push(performance.now() - start);
      if (run === 0) travel = meanNormalizedTravel(prepared, a, b);
    }
    console.log(JSON.stringify({ countA, countB, mode, timesMs: times.map((x) => +x.toFixed(1)), medianMs: +[...times].sort((x, y) => x - y)[1].toFixed(1), meanNormalizedTravel: +travel.toFixed(8) }));
  }
}
const smallA = noisySphere(1500, 12345);
const smallB = noisySphere(2000, 67890);
const lowerBound = nearestNeighborLowerBound(smallA, smallB);
const cloudTravel = meanNormalizedTravel(prepareMorph(smallA, smallB, 42, "cloud"), smallA, smallB);
const shapeTravel = meanNormalizedTravel(prepareMorph(smallA, smallB, 42, "shape"), smallA, smallB);
console.log(JSON.stringify({ countA: 1500, countB: 2000, mortonMeanTravel: cloudTravel, shapeMeanTravel: shapeTravel, nearestNeighborLowerBound: lowerBound, shapeToLowerBoundRatio: shapeTravel / lowerBound }));
