import type {
  Hit,
  PendingItem,
  Channel,
  A2AMessage,
  HealthInfo,
  Project,
  Shelf,
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
