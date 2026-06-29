export type View =
  | "home"
  | "memory"
  | "explorer"
  | "pending"
  | "a2a"
  | "projects"
  | "settings";

export interface Memory {
  text: string;
  agent: string | null;
  kind: string;
  ts: number;
}

export interface GraphNode {
  id: string;
  name: string;
  type: string;
  degree: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  predicate: string;
  confidence: number;
  active: boolean;
}

export interface Graph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  capped: boolean;
  total_nodes: number;
  total_edges: number;
}

export interface Stats {
  scope: string;
  memories: { total: number; disk_mb: number };
  agents: number;
  projects: number;
  growth: { date: string; count: number }[];
  verification: {
    supported: number;
    unverified: number;
    flagged: number;
    hallucination_rate: number;
  };
  categories: { name: string; count: number }[];
  top_agents: { name: string; count: number }[];
  top_projects: { name: string; count: number }[];
  recent_activity: { kind: string; label: string; ts: number }[];
}

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

export interface GeneratorProfile {
  id: string;
  label: string;
  workload: string;
  models: Record<string, string>;
}

export interface GeneratorProfileState {
  profiles: GeneratorProfile[];
  active: string;
  scope: string;
}
