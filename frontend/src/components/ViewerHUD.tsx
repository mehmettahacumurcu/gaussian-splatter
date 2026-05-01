import type { PerfStats } from "../api";

interface Props {
  stats: PerfStats | null;
  visible: boolean;
}

export function ViewerHUD({ stats, visible }: Props) {
  if (!visible || !stats) return null;
  return (
    <div className="viewer-hud" aria-live="polite">
      <span><strong>{stats.fps.toFixed(0)}</strong> FPS</span>
      <span>·</span>
      <span><strong>{(stats.gaussCount / 1000).toFixed(0)}k</strong> gauss</span>
      {stats.vramMB !== undefined && (
        <>
          <span>·</span>
          <span><strong>{stats.vramMB.toFixed(0)}</strong> MB</span>
        </>
      )}
    </div>
  );
}
