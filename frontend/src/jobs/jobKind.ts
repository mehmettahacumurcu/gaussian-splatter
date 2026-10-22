import type { Job } from "../api";

export type JobKind = "spirula" | "coverage" | "compose" | "train";

export function jobKind(job: Pick<Job, "scene">): JobKind {
  if (job.scene.startsWith("spirula-")) return "spirula";
  if (job.scene.startsWith("coverage-")) return "coverage";
  if (job.scene.startsWith("compose-")) return "compose";
  return "train";
}

export const KIND_LABEL: Record<JobKind, string> = {
  spirula: "Spirula · bu PC",
  coverage: "Kapsama taraması",
  compose: "Sahne export",
  train: "Eğitim (4DGS)",
};

/** "spirula-20261022-130501-oda2" → run id "20261022-130501-oda2". */
export function spirulaRunId(job: Pick<Job, "scene">): string | null {
  return job.scene.startsWith("spirula-") ? job.scene.slice("spirula-".length) : null;
}

/** Display name: the run/scene name without the kind prefix and timestamp. */
export function jobTitle(job: Pick<Job, "scene">): string {
  const run = spirulaRunId(job);
  if (run) return run.replace(/^\d{8}-\d{6}-/, "");
  return job.scene.replace(/^(coverage|compose)-/, "");
}

export const STATUS_LABEL: Record<Job["status"], string> = {
  queued: "Sırada",
  running: "Çalışıyor",
  completed: "Bitti",
  failed: "Hata",
};

export const SPIRULA_STEPS: { phase: string; label: string }[] = [
  { phase: "spirula_frames", label: "Kareler" },
  { phase: "spirula_sfm", label: "SfM" },
  { phase: "spirula_geometry", label: "Geometri" },
  { phase: "spirula_train", label: "Eğitim" },
  { phase: "spirula_export", label: "Bitti" },
];

export function elapsedSeconds(job: Pick<Job, "started_at" | "created_at" | "finished_at">, now = Date.now() / 1000): number {
  return Math.max(0, (job.finished_at ?? now) - (job.started_at ?? job.created_at));
}

export function formatElapsed(sec: number): string {
  const s = Math.floor(sec);
  if (s < 60) return `${s} sn`;
  if (s < 3600) return `${Math.floor(s / 60)} dk ${s % 60} sn`;
  return `${Math.floor(s / 3600)} sa ${Math.floor((s % 3600) / 60)} dk`;
}
