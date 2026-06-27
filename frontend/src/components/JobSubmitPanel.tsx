/**
 * JobSubmitPanel — static 3DGS job submission.
 *
 * strip-to-core cleanup: 4D-Dynamic ve "Obje Silme" (edit) modları park
 * edildi. Geriye tek mod (Static 3D) kaldığı için mode picker kaldırıldı;
 * panel doğrudan Static3DSubmit'i render eder. (B / walkable-world işleri
 * CLI + /image-to-splat üzerinden, D+E worlds ise Interactive tab'ından.)
 */
import type { ProcessResponse } from "../api";
import { Static3DSubmit } from "./Static3DSubmit";

interface Props {
  onJobSubmitted: (response: ProcessResponse, sceneName: string) => void;
}

export function JobSubmitPanel({ onJobSubmitted }: Props) {
  return (
    <div className="submit-mode-wrapper">
      <Static3DSubmit onJobSubmitted={onJobSubmitted} />
    </div>
  );
}
