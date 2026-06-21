export type View = "memory" | "pending" | "a2a" | "projects" | "settings";

export type ControlValue = boolean | string | number;

export interface ControlSpec {
  id: string;
  label: string;
  category: "hardware" | "quality" | "integrity";
  scope: "runtime" | "store" | "consumer";
  type: "bool" | "choice" | "int";
  config_key: string;
  default: ControlValue;
  choices: string[];
  int_range: number[];
  cost: string;
  pros: string;
  cons: string;
  description: string;
  benchmarks_anchor: string;
}

export interface PresetSpec {
  id: string;
  label: string;
  description: string;
  values: Record<string, ControlValue>;
}

export interface ControlsSchema {
  controls: ControlSpec[];
  presets: PresetSpec[];
}

export type ControlsSettings = Record<string, ControlValue>;

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

export interface Project {
  project_id: string;
  agents: string[];
  last_ingest: number;
}

export interface Shelf {
  agent: string;
  facts: number;
  last_ingest: number;
}
