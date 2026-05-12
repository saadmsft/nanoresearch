// Typed wrapper for the /api/intent endpoint.

import { api } from "./api";

export type IntentAction =
  | "help"
  | "create_user"
  | "select_user"
  | "update_profile"
  | "list_users"
  | "start_run"
  | "submit_feedback"
  | "status"
  | "list_skills"
  | "list_memories"
  | "train_planner"
  | "chitchat";

export interface Intent {
  action: IntentAction;
  user_id?: string;
  topic?: string;
  feedback?: string;
  profile_updates?: Record<string, string>;
  reply: string;
}

export interface IntentSession {
  user_id?: string | null;
  run_id?: string | null;
  run_status?: string | null;
  has_profile?: boolean;
}

export interface IntentResponse {
  source: "local" | "llm";
  intent: Intent;
}

export async function classifyIntent(
  text: string,
  session: IntentSession,
): Promise<IntentResponse> {
  const res = await fetch("/api/intent", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, session }),
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`HTTP ${res.status}: ${body}`);
  }
  return (await res.json()) as IntentResponse;
}

// Re-export for convenience.
export { api };
