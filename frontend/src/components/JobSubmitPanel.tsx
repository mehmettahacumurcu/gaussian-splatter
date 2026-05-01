/**
 * JobSubmitPanel — pipeline-aware dispatcher.
 *
 * The pipeline switch lives on the global top bar; this component just picks
 * the right submit form for the active pipeline.
 */
import type { JobMode, ProcessResponse } from "../api";
import { Static3DSubmit } from "./Static3DSubmit";
import { Dynamic4DSubmit } from "./Dynamic4DSubmit";

interface Props {
  pipeline: JobMode;
  onJobSubmitted: (response: ProcessResponse, sceneName: string) => void;
}

export function JobSubmitPanel({ pipeline, onJobSubmitted }: Props) {
  return pipeline === "static"
    ? <Static3DSubmit onJobSubmitted={onJobSubmitted} />
    : <Dynamic4DSubmit onJobSubmitted={onJobSubmitted} />;
}
