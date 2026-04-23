/**
 * 4D timeline kontrolleri.
 *
 * Özellikler:
 *   - Slider: 0..N-1 arası frame index seçimi (scrub)
 *   - Play/Pause: otomatik oynatma
 *   - Speed: 0.25x / 0.5x / 1x / 2x / 4x (hedef FPS bazlı)
 *   - Loop: uçlarda başa sar
 */
import { useEffect, useRef, useState } from "react";

interface Props {
  numFrames: number;
  currentFrame: number;
  onFrameChange: (idx: number) => void;
  baseFps?: number; // 1x hızdaki FPS (default: 10)
}

const SPEEDS = [0.25, 0.5, 1, 2, 4] as const;

export function TimelineSlider({
  numFrames,
  currentFrame,
  onFrameChange,
  baseFps = 10,
}: Props) {
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<number>(1);
  const [loop, setLoop] = useState(true);
  const timerRef = useRef<number | null>(null);
  const lastFrameRef = useRef<number>(currentFrame);

  // currentFrame dış kaynaktan değişirse (slider ile manuel scrub), track et
  useEffect(() => {
    lastFrameRef.current = currentFrame;
  }, [currentFrame]);

  // Play/pause tick
  useEffect(() => {
    if (!playing) {
      if (timerRef.current !== null) {
        window.clearInterval(timerRef.current);
        timerRef.current = null;
      }
      return;
    }
    const intervalMs = Math.max(10, Math.round(1000 / (baseFps * speed)));
    timerRef.current = window.setInterval(() => {
      const next = lastFrameRef.current + 1;
      if (next >= numFrames) {
        if (loop) {
          lastFrameRef.current = 0;
          onFrameChange(0);
        } else {
          setPlaying(false);
        }
      } else {
        lastFrameRef.current = next;
        onFrameChange(next);
      }
    }, intervalMs);
    return () => {
      if (timerRef.current !== null) {
        window.clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [playing, speed, baseFps, numFrames, loop, onFrameChange]);

  const maxIdx = Math.max(0, numFrames - 1);

  return (
    <div className="timeline">
      <button
        className="tl-btn tl-play"
        onClick={() => setPlaying((p) => !p)}
        disabled={numFrames === 0}
        title={playing ? "Durdur" : "Oynat"}
      >
        {playing ? "❚❚" : "▶"}
      </button>

      <span className="tl-counter">
        {String(currentFrame + 1).padStart(2, "0")} / {String(numFrames).padStart(2, "0")}
      </span>

      <input
        className="tl-slider"
        type="range"
        min={0}
        max={maxIdx}
        step={1}
        value={currentFrame}
        onChange={(e) => {
          const idx = parseInt(e.target.value, 10);
          lastFrameRef.current = idx;
          onFrameChange(idx);
        }}
      />

      <label className="tl-speed">
        Hız:
        <select
          value={speed}
          onChange={(e) => setSpeed(parseFloat(e.target.value))}
        >
          {SPEEDS.map((s) => (
            <option key={s} value={s}>
              {s}x
            </option>
          ))}
        </select>
      </label>

      <label className="tl-loop">
        <input
          type="checkbox"
          checked={loop}
          onChange={(e) => setLoop(e.target.checked)}
        />
        Loop
      </label>
    </div>
  );
}
