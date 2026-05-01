export type EventLevel = "info" | "warn" | "err" | "raw";

export interface ParsedEvent {
  ts?: string;
  phase?: string;
  level: EventLevel;
  message: string;
  iter?: number;
  raw: string;
}

const TS_RE = /^(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)/;
const LEVEL_RE = /\b(INFO|WARN|WARNING|ERR|ERROR|DEBUG)\b/i;
const PHASE_RE = /\b(foundation|colmap|init|train|training|eval|dens|density)\b/i;
const ITER_RE = /\biter[:\s]+(\d+)/i;

const LEVEL_MAP: Record<string, EventLevel> = {
  INFO: "info",
  DEBUG: "info",
  WARN: "warn",
  WARNING: "warn",
  ERR: "err",
  ERROR: "err",
};

export function parseEvent(line: string): ParsedEvent {
  const tsMatch = TS_RE.exec(line);
  const ts = tsMatch ? tsMatch[1] : undefined;

  const lvlMatch = LEVEL_RE.exec(line);
  const level: EventLevel = lvlMatch ? LEVEL_MAP[lvlMatch[1].toUpperCase()] : "info";

  const phaseMatch = PHASE_RE.exec(line);
  const phase = phaseMatch ? phaseMatch[1].toLowerCase() : undefined;

  const iterMatch = ITER_RE.exec(line);
  const iter = iterMatch ? parseInt(iterMatch[1], 10) : undefined;

  let message = line;
  if (tsMatch) message = message.slice(tsMatch[0].length).trim();
  if (lvlMatch) message = message.replace(lvlMatch[0], "").trim();

  return { ts, phase, level, message, iter, raw: line };
}

export function parseEvents(lines: string[]): ParsedEvent[] {
  return lines.map(parseEvent);
}

export interface PhaseSegment {
  phase: string;
  startTs?: string;
  endTs?: string;
  durationSec?: number;
}

export function deriveTimeline(lines: string[]): PhaseSegment[] {
  const segs: PhaseSegment[] = [];
  let currentPhase: string | undefined;
  let currentStart: string | undefined;
  for (const line of lines) {
    const m = /\bphase[:\s]+([a-z_]+)\b/i.exec(line);
    if (m) {
      const name = m[1].toLowerCase();
      if (currentPhase && currentPhase !== name) {
        segs.push({ phase: currentPhase, startTs: currentStart });
      }
      currentPhase = name;
      const tsm = TS_RE.exec(line);
      currentStart = tsm ? tsm[1] : undefined;
    }
  }
  if (currentPhase) segs.push({ phase: currentPhase, startTs: currentStart });
  return segs;
}
