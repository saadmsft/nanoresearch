// Typed wrappers around the FastAPI backend.
// Keeping every shape narrow + explicit so the UI stays honest about what
// the server actually returns.

export type RunStatus =
  | "pending"
  | "running"
  | "awaiting_feedback"
  | "completed"
  | "failed";

export interface UserProfile {
  user_id: string;
  archetype: string;
  domain: string;
  research_preference?: string;
  method_preference?: string;
  risk_preference?: string;
  baseline_strictness?: string;
  resource_budget?: string;
  feasibility_bias?: string;
  writing_tone?: string;
  claim_strength?: string;
  section_organization?: string;
  venue_style?: string;
  latex_template?: string;
  persona_brief?: string;
}

export interface RunSnapshot {
  run_id: string;
  user_id: string;
  topic: string;
  project_id: string;
  status: RunStatus;
  current_stage: string | null;
  stages_completed: string[];
  last_summary: string;
  started_at: string;
  updated_at: string;
  error: string | null;
}

export interface Skill {
  skill_id: string;
  skill_type: string;
  name: string;
  when_to_apply: string;
  procedure: string;
  tags: string[];
  usage_count: number;
  confidence: number;
}

export interface Memory {
  memory_id: string;
  memory_type: string;
  topic_scope: string;
  content: string;
  tags: string[];
}

async function jsonFetch<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
  const res = await fetch(input, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`HTTP ${res.status}: ${body || res.statusText}`);
  }
  return (await res.json()) as T;
}

export const api = {
  listUsers: () => jsonFetch<string[]>("/api/users"),
  getUser: (id: string) => jsonFetch<UserProfile>(`/api/users/${id}`),
  upsertUser: (profile: UserProfile) =>
    jsonFetch<UserProfile>("/api/users", {
      method: "POST",
      body: JSON.stringify(profile),
    }),
  getSkills: (id: string) => jsonFetch<Skill[]>(`/api/users/${id}/skills`),
  getMemories: (id: string) => jsonFetch<Memory[]>(`/api/users/${id}/memories`),

  listRuns: () => jsonFetch<RunSnapshot[]>("/api/runs"),
  getRun: (runId: string) => jsonFetch<RunSnapshot>(`/api/runs/${runId}`),
  startRun: (userId: string, topic: string) =>
    jsonFetch<RunSnapshot>("/api/runs", {
      method: "POST",
      body: JSON.stringify({ user_id: userId, topic }),
    }),
  submitFeedback: (runId: string, text: string) =>
    jsonFetch<RunSnapshot>(`/api/runs/${runId}/feedback`, {
      method: "POST",
      body: JSON.stringify({ text }),
    }),
};
