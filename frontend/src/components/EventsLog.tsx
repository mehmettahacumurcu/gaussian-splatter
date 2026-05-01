import { useMemo, useState } from "react";
import type { ParsedEvent, EventLevel } from "../analytics/events";

interface Props {
  events: ParsedEvent[];
  onSelectIter?: (iter: number) => void;
}

export function EventsLog({ events, onSelectIter }: Props) {
  const [phaseFilter, setPhaseFilter] = useState<string>("all");
  const [levelFilter, setLevelFilter] = useState<EventLevel | "all">("all");

  const phases = useMemo(() => {
    const set = new Set<string>();
    for (const e of events) if (e.phase) set.add(e.phase);
    return ["all", ...Array.from(set).sort()];
  }, [events]);

  const filtered = useMemo(() => {
    return events.filter((e) =>
      (phaseFilter === "all" || e.phase === phaseFilter) &&
      (levelFilter === "all" || e.level === levelFilter)
    );
  }, [events, phaseFilter, levelFilter]);

  return (
    <div className="events-log">
      <div className="events-log-filters">
        <label>
          phase:
          <select value={phaseFilter} onChange={(e) => setPhaseFilter(e.target.value)}>
            {phases.map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
        </label>
        <label>
          level:
          <select
            value={levelFilter}
            onChange={(e) => setLevelFilter(e.target.value as EventLevel | "all")}
          >
            <option value="all">all</option>
            <option value="info">info</option>
            <option value="warn">warn</option>
            <option value="err">err</option>
          </select>
        </label>
        <span className="events-log-count">{filtered.length} / {events.length}</span>
      </div>
      <ul className="events-log-list">
        {filtered.slice(-200).map((e, i) => (
          <li
            key={i}
            className={`events-log-row events-log-${e.level}`}
            onClick={() => e.iter !== undefined && onSelectIter?.(e.iter)}
          >
            <span className="events-log-ts">{e.ts ?? ""}</span>
            <span className="events-log-phase">{e.phase ?? ""}</span>
            <span className="events-log-msg">{e.message}</span>
          </li>
        ))}
        {filtered.length === 0 && (
          <li className="events-log-empty">No events match filter.</li>
        )}
      </ul>
    </div>
  );
}
