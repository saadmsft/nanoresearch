import { useEffect, useRef, useState } from "react";

export interface RunEvent {
  ts: string;
  event: string;
  run_id: string;
  [key: string]: unknown;
}

export interface NarrationEvent {
  ts: string;
  text: string;
  run_id: string;
}

const TERMINAL = new Set(["run_completed", "run_failed", "stream_end"]);

/**
 * Subscribe to the server-sent event stream for a single run.
 * Returns the accumulated technical event log, narration messages, and a
 * "closed" flag. The narrations are intended for the chat; the technical
 * events feed the sidebar.
 */
export function useRunStream(runId: string | null): {
  events: RunEvent[];
  narrations: NarrationEvent[];
  closed: boolean;
} {
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [narrations, setNarrations] = useState<NarrationEvent[]>([]);
  const [closed, setClosed] = useState(false);
  const srcRef = useRef<EventSource | null>(null);

  useEffect(() => {
    setEvents([]);
    setNarrations([]);
    setClosed(false);
    if (!runId) return;

    const src = new EventSource(`/api/runs/${runId}/stream`);
    srcRef.current = src;

    const onMessage = (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data) as RunEvent;
        if (data.event === "narration" && typeof data.text === "string") {
          setNarrations((prev) => [
            ...prev,
            { ts: data.ts, run_id: data.run_id, text: data.text as string },
          ]);
        } else {
          setEvents((prev) => [...prev, data]);
        }
        if (TERMINAL.has(data.event)) {
          setClosed(true);
          src.close();
        }
      } catch {
        // ignore non-JSON keepalives
      }
    };
    src.addEventListener("message", onMessage);
    for (const evt of [
      "trajectory_event",
      "status_changed",
      "stage_completed",
      "awaiting_feedback",
      "feedback_received",
      "feedback_enqueued",
      "run_started",
      "run_completed",
      "run_failed",
      "stream_end",
      "narration",
    ]) {
      src.addEventListener(evt, onMessage as EventListener);
    }
    // EventSource fires `error` when the server-side generator finishes —
    // rely on the explicit TERMINAL events instead.

    return () => {
      src.close();
      srcRef.current = null;
    };
  }, [runId]);

  return { events, narrations, closed };
}
