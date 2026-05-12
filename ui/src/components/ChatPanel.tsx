// ChatGPT-style chat. The chat owns the conversation, the sidebar shows
// compact run status. Every user turn:
//   1. Render as a user bubble immediately.
//   2. Call /api/intent to classify it (LLM with slash-command fast path).
//   3. Execute the side effect against the right API (`/users`, `/runs`,
//      `/feedback`, …) and render the result as one or more assistant turns.
//   4. While a run is live, narration events from the SSE stream are
//      *appended* as additional assistant turns in real time.

import { useCallback, useEffect, useRef, useState } from "react";
import { api, type UserProfile } from "@/lib/api";
import { classifyIntent, type Intent } from "@/lib/intent";
import { useRunStream } from "@/hooks/useRunStream";
import { useSession } from "@/lib/session";

type Author = "user" | "assistant";

interface ChatTurn {
  id: string;
  author: Author;
  text: string;
}

const STORAGE_KEY = "nano.chat.thread";

function loadThread(): ChatTurn[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const v = JSON.parse(raw);
    return Array.isArray(v) ? (v as ChatTurn[]) : [];
  } catch {
    return [];
  }
}

function saveThread(turns: ChatTurn[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(turns.slice(-200)));
  } catch {
    /* quota — give up silently */
  }
}

export function ChatPanel() {
  const sess = useSession();
  const [turns, setTurns] = useState<ChatTurn[]>(() => loadThread());
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const lastNarrationIdx = useRef(0);
  const viewportRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    sess.hydrate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    saveThread(turns);
    requestAnimationFrame(() => {
      const el = viewportRef.current;
      if (el) el.scrollTop = el.scrollHeight;
    });
  }, [turns]);

  // ---- narration stream → assistant turns ---------------------------
  const { narrations } = useRunStream(sess.runId);
  useEffect(() => {
    if (narrations.length <= lastNarrationIdx.current) return;
    const fresh = narrations.slice(lastNarrationIdx.current);
    lastNarrationIdx.current = narrations.length;
    setTurns((prev) => [
      ...prev,
      ...fresh.map((n, i) => ({
        id: `narr-${n.ts}-${i}`,
        author: "assistant" as const,
        text: n.text,
      })),
    ]);
  }, [narrations]);

  useEffect(() => {
    lastNarrationIdx.current = 0;
  }, [sess.runId]);

  const append = (turn: ChatTurn) => setTurns((prev) => [...prev, turn]);

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    append({ id: `u-${Date.now()}`, author: "user", text });
    setBusy(true);

    const session = {
      user_id: sess.userId,
      run_id: sess.runId,
      run_status: sess.snapshot?.status,
      has_profile: !!sess.profile,
    };
    try {
      const { intent } = await classifyIntent(text, session);
      const responses = await execute(intent, sess);
      for (const r of responses) append(r);
    } catch (e) {
      append({
        id: `e-${Date.now()}`,
        author: "assistant",
        text: `Something went wrong: ${String(e)}`,
      });
    } finally {
      setBusy(false);
    }
  }, [input, busy, sess]);

  const execute = async (
    intent: Intent,
    s: ReturnType<typeof useSession>,
  ): Promise<ChatTurn[]> => {
    const out: ChatTurn[] = [];
    const reply = (txt: string) =>
      out.push({ id: `a-${Date.now()}-${out.length}`, author: "assistant", text: txt });

    if (intent.reply) reply(intent.reply);

    switch (intent.action) {
      case "help":
        reply(HELP_TEXT);
        break;
      case "list_users": {
        const users = await api.listUsers();
        reply(
          users.length === 0
            ? "No profiles yet. Tell me your name and field and I'll create one."
            : `Profiles: ${users.map((u) => `\`${u}\``).join(", ")}.`,
        );
        break;
      }
      case "create_user": {
        if (!intent.user_id) {
          reply("What short identifier should I use for your profile?");
          break;
        }
        const draft: UserProfile = {
          user_id: intent.user_id,
          archetype: "general_research",
          domain: "General",
          risk_preference: "moderate",
          baseline_strictness: "high",
          ...(intent.profile_updates as Partial<UserProfile>),
        };
        const saved = await api.upsertUser(draft);
        await s.selectUser(saved.user_id);
        s.persist();
        reply(
          `Created profile **${saved.user_id}**. Tell me a research topic when you're ready, or refine your preferences first.`,
        );
        break;
      }
      case "select_user": {
        if (!intent.user_id) break;
        try {
          await api.getUser(intent.user_id);
          await s.selectUser(intent.user_id);
          s.persist();
          reply(`Switched to **${intent.user_id}**.`);
        } catch {
          reply(`No profile **${intent.user_id}** yet. Want me to create one?`);
        }
        break;
      }
      case "update_profile": {
        if (!s.profile) {
          reply("Let's set up a profile first — what should I call you?");
          break;
        }
        const next: UserProfile = {
          ...s.profile,
          ...(intent.profile_updates as Partial<UserProfile>),
        };
        const saved = await api.upsertUser(next);
        await s.selectUser(saved.user_id);
        const keys = Object.keys(intent.profile_updates ?? {});
        reply(`Updated: ${keys.map((k) => `\`${k}\``).join(", ") || "(nothing changed)"}.`);
        break;
      }
      case "start_run": {
        if (!s.userId) {
          reply("Let's set up a profile first. What should I call you?");
          break;
        }
        if (!intent.topic) {
          reply("What topic should I work on?");
          break;
        }
        if (
          s.snapshot?.status === "running" ||
          s.snapshot?.status === "awaiting_feedback"
        ) {
          reply(
            `A run is already \`${s.snapshot.status}\`. Send feedback or wait for it to finish.`,
          );
          break;
        }
        try {
          const snap = await api.startRun(s.userId, intent.topic);
          s.setSnapshot(snap);
          s.persist();
          lastNarrationIdx.current = 0;
        } catch (e) {
          reply(`Couldn't start: ${String(e)}`);
        }
        break;
      }
      case "submit_feedback": {
        if (!s.runId) {
          reply("There's no active run. Tell me a topic to start one.");
          break;
        }
        const fb = (intent.feedback ?? "").trim();
        if (!fb) break;
        try {
          const snap = await api.submitFeedback(s.runId, fb);
          s.setSnapshot(snap);
        } catch (e) {
          reply(`Couldn't submit feedback: ${String(e)}`);
        }
        break;
      }
      case "status": {
        if (!s.runId) {
          reply("No active run.");
          break;
        }
        const snap = await api.getRun(s.runId);
        s.setSnapshot(snap);
        reply(
          `Status: \`${snap.status}\`, current stage: \`${snap.current_stage ?? "—"}\`.${
            snap.last_summary ? `\n\n_${snap.last_summary}_` : ""
          }`,
        );
        break;
      }
      case "list_skills": {
        if (!s.userId) break;
        const skills = await api.getSkills(s.userId);
        reply(
          skills.length === 0
            ? "No skills accumulated yet — run a research cycle first."
            : `**Skills (${skills.length}):**\n` +
                skills
                  .map(
                    (sk) =>
                      `- ${sk.name}  _conf=${sk.confidence.toFixed(2)}, used=${sk.usage_count}_`,
                  )
                  .join("\n"),
        );
        break;
      }
      case "list_memories": {
        if (!s.userId) break;
        const mems = await api.getMemories(s.userId);
        reply(
          mems.length === 0
            ? "No memories yet."
            : `**Memories (${mems.length}):**\n` +
                mems
                  .map((m) => `- _${m.topic_scope}_: ${m.content.slice(0, 160)}`)
                  .join("\n"),
        );
        break;
      }
      case "train_planner":
        reply(
          "SDPO training plugs in once the local planner adapter is loaded. Your feedback is being buffered for the next round.",
        );
        break;
      case "chitchat":
        break;
    }
    return out;
  };

  useEffect(() => {
    if (turns.length === 0) {
      setTurns([
        {
          id: "welcome",
          author: "assistant",
          text:
            "👋 Hi — I'm **NanoResearch**.\n\nI take a research idea from a one-liner to a peer-reviewed experiment plan, in any scholarly field (biology, social sciences, engineering, computer science — whatever you work on).\n\nTo get going, just tell me **your name and your field**, e.g. _“I'm Mia, ecology, prefer field studies on a 6-month budget.”_\n\nOr jump right in: _“Start a run on the impact of urban green roofs on songbird diversity.”_",
        },
      ]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <section className="flex h-full min-h-0 flex-col bg-slate-950">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 px-4 py-2.5">
        <div className="min-w-0">
          <h1 className="text-base font-semibold tracking-tight">NanoResearch</h1>
          <p className="truncate text-xs text-slate-500">
            Chat with a research assistant that learns your preferences.
          </p>
        </div>
        <Identity />
      </header>

      <div ref={viewportRef} className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
        {turns.map((t) => (
          <Bubble key={t.id} turn={t} />
        ))}
        {busy && <Bubble turn={{ id: "thinking", author: "assistant", text: "…" }} />}
      </div>

      <Composer
        value={input}
        onChange={setInput}
        onSend={send}
        disabled={busy}
        placeholder={composerPlaceholder(sess)}
      />
    </section>
  );
}

// ============================================================ subcomponents

function composerPlaceholder(s: ReturnType<typeof useSession>): string {
  if (!s.userId)
    return "Introduce yourself and your field — e.g. “I'm Saad, environmental engineering.”";
  if (s.snapshot?.status === "awaiting_feedback")
    return "Type feedback on the last step — what to emphasise, change, or rule out.";
  if (s.snapshot?.status === "running")
    return "Working on your run… you can still ask questions or for status.";
  return "Tell me a topic to research, or update your profile.";
}

function Identity() {
  const s = useSession();
  if (!s.userId) return <span className="shrink-0 text-xs text-slate-500">no profile</span>;
  return (
    <span className="shrink-0 whitespace-nowrap text-xs text-slate-400">
      <span className="font-mono text-slate-200">{s.userId}</span>
      {s.snapshot && (
        <>
          <span className="mx-2 text-slate-600">·</span>
          <span className="text-slate-300">{s.snapshot.status}</span>
        </>
      )}
    </span>
  );
}

function Bubble({ turn }: { turn: ChatTurn }) {
  const isUser = turn.author === "user";
  return (
    <div className={`mb-3 flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={
          isUser
            ? "max-w-[80%] whitespace-pre-wrap rounded-2xl bg-brand-600/90 px-3.5 py-2 text-sm leading-relaxed text-slate-50"
            : "max-w-[85%] whitespace-pre-wrap rounded-2xl bg-slate-800/80 px-3.5 py-2 text-sm leading-relaxed text-slate-100"
        }
      >
        {renderMarkdownish(turn.text)}
      </div>
    </div>
  );
}

function Composer(props: {
  value: string;
  onChange: (v: string) => void;
  onSend: () => void;
  disabled: boolean;
  placeholder: string;
}) {
  const onKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      props.onSend();
    }
  };
  return (
    <div className="flex items-end gap-2 border-t border-slate-800 px-3 py-2.5">
      <textarea
        rows={2}
        placeholder={props.placeholder}
        value={props.value}
        onChange={(e) => props.onChange(e.target.value)}
        onKeyDown={onKey}
        className="flex-1 resize-none rounded-md bg-slate-800/70 px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-brand-500"
      />
      <button
        type="button"
        disabled={props.disabled || !props.value.trim()}
        onClick={props.onSend}
        className="rounded-md bg-brand-600 px-3 py-1.5 text-sm font-medium hover:bg-brand-500 disabled:opacity-40"
      >
        Send
      </button>
    </div>
  );
}

// Small markdown subset: **bold**, _italic_, `code`, "- " bullets.
function renderMarkdownish(text: string): React.ReactNode {
  const lines = text.split("\n");
  return lines.map((line, i) => {
    if (line.trim() === "") return <div key={i} className="h-2" />;
    if (line.trimStart().startsWith("- ")) {
      return (
        <div key={i} className="ml-3">
          • {inline(line.trimStart().slice(2))}
        </div>
      );
    }
    return <div key={i}>{inline(line)}</div>;
  });
}

function inline(s: string): React.ReactNode {
  const parts: React.ReactNode[] = [];
  // Matches: **bold**, _italic_, `code`, [text](url)
  const pattern = /(\*\*[^*]+\*\*|_[^_]+_|`[^`]+`|\[[^\]]+\]\([^)]+\))/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let key = 0;
  while ((m = pattern.exec(s)) !== null) {
    if (m.index > last) parts.push(s.slice(last, m.index));
    const token = m[0];
    if (token.startsWith("**"))
      parts.push(<strong key={key++}>{token.slice(2, -2)}</strong>);
    else if (token.startsWith("`"))
      parts.push(
        <code key={key++} className="rounded bg-slate-900/80 px-1 text-[0.85em]">
          {token.slice(1, -1)}
        </code>,
      );
    else if (token.startsWith("[")) {
      const linkMatch = token.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
      if (linkMatch) {
        parts.push(
          <a
            key={key++}
            href={linkMatch[2]}
            target="_blank"
            rel="noreferrer noopener"
            className="text-brand-400 underline underline-offset-2 hover:text-brand-300"
          >
            {linkMatch[1]}
          </a>,
        );
      } else {
        parts.push(token);
      }
    } else if (token.startsWith("_")) parts.push(<em key={key++}>{token.slice(1, -1)}</em>);
    last = m.index + token.length;
  }
  if (last < s.length) parts.push(s.slice(last));
  return parts;
}

const HELP_TEXT = `**You don't need commands** — just talk to me naturally. Examples:

- _“I'm Mia, ecology, prefer field studies.”_ — sets up a profile
- _“Switch to Saad's profile.”_
- _“Use a 3-month timeline and require IRB approval.”_ — updates the active profile
- _“Start a run on sleep regularity and academic performance in undergrads.”_
- _“What's the status?”_ / _“Show me my skills.”_

While I'm working you can interject with feedback like _“drop the field-survey arm and emphasise the EEG sub-study”_ and I'll bake it in.`;
