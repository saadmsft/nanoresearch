// Compact run-status sidebar.
//
// Replaces the verbose Activities panel. Shows the active profile, the
// pipeline timeline, the current blueprint summary, and Skill/Memory
// counters. No technical event feed — the chat already narrates that.

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useSession } from "@/lib/session";

const PIPELINE_STAGES = [
  { key: "ideation", label: "Ideation" },
  { key: "planning", label: "Planning" },
  { key: "coding", label: "Experimentation" },
  { key: "analysis", label: "Analysis" },
  { key: "writing", label: "Writing" },
  { key: "review", label: "Review" },
] as const;

const STATUS_TINT: Record<string, string> = {
  pending: "text-slate-400",
  running: "text-amber-400 animate-pulse",
  awaiting_feedback: "text-sky-400 animate-pulse",
  completed: "text-emerald-400",
  failed: "text-rose-400",
};

export function StatusPanel() {
  const s = useSession();
  return (
    <aside className="flex h-full min-h-0 flex-col gap-3 overflow-y-auto bg-slate-950 p-3">
      <ProfileCard />
      <PipelineCard />
      <StoresCard />
      {!s.userId && (
        <p className="text-xs italic text-slate-500">
          Introduce yourself in the chat to populate this panel.
        </p>
      )}
    </aside>
  );
}

// ============================================================ Profile

function ProfileCard() {
  const s = useSession();
  if (!s.profile) {
    return (
      <Card title="Profile">
        <p className="text-xs text-slate-500">No active profile.</p>
      </Card>
    );
  }
  return (
    <Card title="Profile">
      <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
        <Field label="name" value={s.profile.user_id} mono />
        <Field label="archetype" value={s.profile.archetype} />
        <Field label="domain" value={s.profile.domain} />
        <Field label="risk" value={s.profile.risk_preference ?? "—"} />
        <Field label="strictness" value={s.profile.baseline_strictness ?? "—"} />
        {s.profile.resource_budget && (
          <Field label="budget" value={s.profile.resource_budget} />
        )}
      </div>
      {s.profile.persona_brief && (
        <p className="mt-2 text-xs italic text-slate-400">“{s.profile.persona_brief}”</p>
      )}
    </Card>
  );
}

// ============================================================ Pipeline

function PipelineCard() {
  const s = useSession();
  const snap = s.snapshot;

  useEffect(() => {
    if (!s.runId) return;
    const terminal = new Set(["completed", "failed"]);
    if (snap && terminal.has(snap.status)) return;
    let live = true;
    const tick = () =>
      api
        .getRun(s.runId!)
        .then((next) => {
          if (live) s.setSnapshot(next);
        })
        .catch(() => undefined);
    tick();
    const t = setInterval(tick, 8000);
    return () => {
      live = false;
      clearInterval(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s.runId, snap?.status === "completed" || snap?.status === "failed"]);

  return (
    <Card
      title="Pipeline"
      right={
        snap ? (
          <span className={`text-[11px] font-mono ${STATUS_TINT[snap.status] ?? ""}`}>
            {snap.status}
          </span>
        ) : null
      }
    >
      {!snap ? (
        <p className="text-xs text-slate-500">No active run.</p>
      ) : (
        <>
          <p className="mb-2 text-xs italic text-slate-400 line-clamp-2">{snap.topic}</p>
          <ol className="space-y-1.5">
            {PIPELINE_STAGES.map((stage) => {
              const done = snap.stages_completed.includes(stage.key);
              const active = snap.current_stage === stage.key;
              const dot = done
                ? "bg-emerald-500"
                : active
                  ? "bg-amber-400 animate-pulse"
                  : "bg-slate-700";
              const text = done
                ? "text-emerald-300"
                : active
                  ? "text-amber-200"
                  : "text-slate-500";
              return (
                <li key={stage.key} className="flex items-center gap-2 text-xs">
                  <span className={`inline-block h-2 w-2 rounded-full ${dot}`} />
                  <span className={text}>{stage.label}</span>
                  {active && snap.status === "awaiting_feedback" && (
                    <span className="ml-auto text-[10px] uppercase tracking-wide text-sky-400">
                      needs feedback
                    </span>
                  )}
                </li>
              );
            })}
          </ol>
          {snap.last_summary && (
            <p className="mt-2 line-clamp-4 border-l-2 border-slate-700 pl-2 text-xs text-slate-300">
              {snap.last_summary}
            </p>
          )}
          {snap.error && <p className="mt-2 text-xs text-rose-400">{snap.error}</p>}
        </>
      )}
    </Card>
  );
}

// ============================================================ Stores

function StoresCard() {
  const s = useSession();
  const [skills, setSkills] = useState<number | null>(null);
  const [memories, setMemories] = useState<number | null>(null);

  useEffect(() => {
    if (!s.userId) return;
    let live = true;
    const refresh = async () => {
      try {
        const [sk, mm] = await Promise.all([
          api.getSkills(s.userId!),
          api.getMemories(s.userId!),
        ]);
        if (live) {
          setSkills(sk.length);
          setMemories(mm.length);
        }
      } catch {
        if (live) {
          setSkills(null);
          setMemories(null);
        }
      }
    };
    refresh();
    const liveRun =
      s.snapshot?.status === "running" ||
      s.snapshot?.status === "awaiting_feedback";
    if (!liveRun) return () => {
      live = false;
    };
    const t = setInterval(refresh, 15000);
    return () => {
      live = false;
      clearInterval(t);
    };
  }, [s.userId, s.snapshot?.status]);

  if (!s.userId) return null;
  return (
    <Card title="What I've learned about you">
      <div className="grid grid-cols-2 gap-2">
        <Stat label="Skills" value={skills} />
        <Stat label="Memories" value={memories} />
      </div>
      <p className="mt-2 text-[10px] text-slate-500">
        Ask me “show my skills” or “show my memories” to inspect.
      </p>
    </Card>
  );
}

// ============================================================ primitives

function Card({
  title,
  right,
  children,
}: {
  title: string;
  right?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900/60 p-3">
      <header className="mb-2 flex items-center justify-between">
        <h2 className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
          {title}
        </h2>
        {right}
      </header>
      {children}
    </section>
  );
}

function Field({
  label,
  value,
  mono,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <>
      <dt className="text-slate-500">{label}</dt>
      <dd className={`truncate text-slate-200 ${mono ? "font-mono" : ""}`}>{value}</dd>
    </>
  );
}

function Stat({ label, value }: { label: string; value: number | null }) {
  return (
    <div className="rounded-md bg-slate-800/60 px-2.5 py-2">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className="mt-0.5 font-mono text-lg text-slate-100">{value ?? "—"}</div>
    </div>
  );
}
