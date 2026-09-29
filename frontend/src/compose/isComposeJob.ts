import type { Job } from "../api";

/** Compose export jobs (scene "compose-<id>") have no splat output for the viewer. */
export function isComposeJob(job: Pick<Job, "scene">): boolean {
  return job.scene.startsWith("compose-");
}
