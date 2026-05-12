// Lightweight client-side store for the UI's current selection state.
// Keeps the chat + activities panel in sync without prop-drilling everywhere.

import { useEffect, useState } from "react";
import { api, type RunSnapshot, type UserProfile } from "./api";

type Listener = () => void;

class SessionStore {
  private _userId: string | null = null;
  private _runId: string | null = null;
  private _profile: UserProfile | null = null;
  private _snap: RunSnapshot | null = null;
  private listeners = new Set<Listener>();

  subscribe(l: Listener) {
    this.listeners.add(l);
    return () => {
      this.listeners.delete(l);
    };
  }

  private emit() {
    for (const l of this.listeners) l();
  }

  get userId() {
    return this._userId;
  }
  get runId() {
    return this._runId;
  }
  get profile() {
    return this._profile;
  }
  get snapshot() {
    return this._snap;
  }

  async selectUser(id: string | null) {
    this._userId = id;
    this._profile = null;
    if (id) {
      try {
        this._profile = await api.getUser(id);
      } catch {
        this._profile = null;
      }
    }
    this.emit();
  }

  setSnapshot(s: RunSnapshot | null) {
    this._snap = s;
    if (s) this._runId = s.run_id;
    this.emit();
  }

  setRunId(id: string | null) {
    this._runId = id;
    if (id === null) this._snap = null;
    this.emit();
  }

  // Persistence helpers — survive page reloads via localStorage.
  hydrate() {
    const u = localStorage.getItem("nano.userId");
    if (u) this.selectUser(u);
    const r = localStorage.getItem("nano.runId");
    if (r) this.setRunId(r);
  }
  persist() {
    if (this._userId) localStorage.setItem("nano.userId", this._userId);
    else localStorage.removeItem("nano.userId");
    if (this._runId) localStorage.setItem("nano.runId", this._runId);
    else localStorage.removeItem("nano.runId");
  }
}

export const session = new SessionStore();

export function useSession() {
  const [, force] = useState(0);
  useEffect(() => {
    const unsub = session.subscribe(() => force((n) => n + 1));
    return unsub;
  }, []);
  return session;
}
