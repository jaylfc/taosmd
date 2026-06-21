import type {
  Hit,
  PendingItem,
  Channel,
  A2AMessage,
  HealthInfo,
  Project,
  Shelf,
  ControlsSchema,
  ControlsSettings,
  ControlValue,
  Stats,
  Memory,
  Graph,
} from "./types";

async function req<T>(url: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(url, opts);
  const body = await res.json();
  if (!res.ok) {
    throw new Error((body as { error?: string }).error ?? `HTTP ${res.status}`);
  }
  return body as T;
}

export async function getHealth(): Promise<HealthInfo> {
  return req<HealthInfo>("/health");
}

export async function getStats(scope?: string): Promise<Stats> {
  const q = scope && scope !== "all" ? `?scope=${encodeURIComponent(scope)}` : "";
  return req<Stats>(`/stats${q}`);
}

export async function getMemories(scope?: string, limit = 50): Promise<Memory[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (scope && scope !== "all") params.set("scope", scope);
  const { memories } = await req<{ memories: Memory[] }>(`/memories?${params}`);
  return memories;
}

export async function getGraph(limit = 300): Promise<Graph> {
  return req<Graph>(`/graph?limit=${limit}`);
}

export async function getGraphActivations(
  window = 60,
): Promise<{ activations: { id: string; last_accessed_at: number }[]; now: number }> {
  return req(`/graph/activations?window=${window}`);
}

export async function searchMemory(
  query: string,
  agent: string,
  limit = 10
): Promise<Hit[]> {
  const { hits } = await req<{ hits: Hit[] }>("/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, agent, limit }),
  });
  return hits;
}

export async function getPending(agent: string): Promise<PendingItem[]> {
  const url = `/pending${agent ? `?agent=${encodeURIComponent(agent)}` : ""}`;
  const { pending } = await req<{ pending: PendingItem[] }>(url);
  return pending;
}

export async function getChannels(): Promise<Channel[]> {
  const { channels } = await req<{ channels: Channel[] }>("/a2a/channels");
  return channels;
}

export async function getMembers(channel: string): Promise<string[]> {
  const { members } = await req<{ members: string[] }>(
    `/a2a/members?channel=${encodeURIComponent(channel)}`
  );
  return members;
}

export async function getMessages(
  thread: string,
  limit = 200
): Promise<A2AMessage[]> {
  const { messages } = await req<{ messages: A2AMessage[] }>(
    `/a2a/messages?thread=${encodeURIComponent(thread)}&limit=${limit}`
  );
  return messages;
}

export async function getProjects(): Promise<Project[]> {
  const { projects } = await req<{ projects: Project[] }>("/projects");
  return projects;
}

export async function getShelves(project: string): Promise<Shelf[]> {
  const { shelves } = await req<{ shelves: Shelf[] }>(
    `/shelves?project=${encodeURIComponent(project)}`
  );
  return shelves;
}

export async function getControls(): Promise<{
  settings: ControlsSettings;
  schema: ControlsSchema;
}> {
  return req<{ settings: ControlsSettings; schema: ControlsSchema }>("/controls");
}

// Write one or more controls, or apply a preset. A 400 that carries per-field
// `errors` is a validation result, not a transport failure, so we return it.
// Other failures (unknown preset, empty body, managed-by-taOS, network) throw.
export async function postControls(
  payload: Record<string, ControlValue> | { preset: string }
): Promise<{ settings: ControlsSettings; errors: Record<string, string> }> {
  const res = await fetch("/controls", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = (await res.json().catch(() => ({}))) as {
    settings?: ControlsSettings;
    errors?: Record<string, string>;
    error?: string;
  };
  if (!res.ok && !body.errors) {
    throw new Error(body.error ?? `HTTP ${res.status}`);
  }
  return { settings: body.settings ?? {}, errors: body.errors ?? {} };
}
