// Node 22.18+/24: node --expose-gc scripts/benchmark-morph.mjs
// Add --alignment to run only the rotated-copy alignment benchmark.
// Times include validation, robust bounds, correspondence, padding and output
// attribute copies. Input generation and travel measurement are excluded.
import { performance } from "node:perf_hooks";
import os from "node:os";
import { asymmetricCloud, rotatedCloud } from "../src/compose/__tests__/fixtures/morphAlignment.ts";
import { ALIGNMENT_SAMPLE_LIMIT } from "../src/compose/morphAlignment.ts";
import { prepareMorph } from "../src/compose/morphData.ts";
import { noisySphere, meanNormalizedTravel, nearestNeighborLowerBound } from "../src/compose/__tests__/fixtures/morphSpheres.ts";

if (!process.argv.includes("--alignment")) {
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
}

console.log(JSON.stringify({
  benchmark: "automatic-alignment", node: process.version, cpu: os.cpus()[0]?.model,
  fixture: "Lopsided anisotropic noisy shell (seed 12345), arbitrary rotation and shuffled copy (seed 617)",
  sampleLimit: ALIGNMENT_SAMPLE_LIMIT,
  units: "Off: longest robust extent; on: twice RMS radius. meanTravel: actual world travel / larger robust bbox diagonal.",
  timing: "End-to-end prepareMorph; alignmentMs subset includes PCA, sample candidate/refinement scoring, and coordinate transforms. One 10k warmup, median of three runs. Loading, worker startup/transfer and GPU upload excluded.",
}));
const alignmentWarmA = asymmetricCloud(10_000);
const alignmentWarmB = rotatedCloud(alignmentWarmA, 617, 1, 0, true);
for (const enabled of [false, true]) prepareMorph(alignmentWarmA, alignmentWarmB, 42, "shape", enabled);
const median = (values) => +[...values].sort((a, b) => a - b)[Math.floor(values.length / 2)].toFixed(1);
for (const count of [100_000, 1_000_000]) {
  const a = asymmetricCloud(count);
  const b = rotatedCloud(a, 617, 1, 0, true);
  for (const autoAlign of [false, true]) {
    const times = [], alignmentTimes = [];
    let matchingTravel = 0, meanTravel = 0;
    for (let run = 0; run < 3; run++) {
      global.gc?.();
      const start = performance.now();
      const prepared = prepareMorph(a, b, 42, "shape", autoAlign);
      times.push(performance.now() - start);
      alignmentTimes.push(prepared.alignmentMs);
      if (run === 0) { matchingTravel = prepared.matchingTravel; meanTravel = prepared.meanTravel; }
    }
    console.log(JSON.stringify({
      benchmark: "automatic-alignment", countA: count, countB: count, autoAlign,
      timesMs: times.map((x) => +x.toFixed(1)), medianMs: median(times),
      alignmentTimesMs: alignmentTimes.map((x) => +x.toFixed(1)), medianAlignmentMs: median(alignmentTimes),
      matchingTravel: +matchingTravel.toFixed(10), meanTravel: +meanTravel.toFixed(10),
    }));
  }
}
