export type View = "memory" | "pending" | "a2a";

export interface Hit {
  text: string;
  source?: string;
  confidence?: number;
  timestamp?: string;
  metadata?: Record<string, unknown>;
}

export interface PendingItem {
  id: string;
  subject?: string;
  predicate?: string;
  object?: string;
  kind?: string;
}

export interface Channel {
  channel: string;
  members: string[];
  message_count: number;
  created_ts: number;
  last_ts: number;
}

export interface A2AMessage {
  id: string | number;
  ts: number;
  from: string | null;
  body: string | null;
  thread: string | null;
  reply_to?: string | number | null;
}

export interface HealthInfo {
  status: string;
  version: string;
}
