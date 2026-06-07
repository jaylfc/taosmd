import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
  useMemo,
} from "react";
import { getChannels, getMembers, getMessages } from "../api";
import type { Channel, A2AMessage } from "../types";
import { Skeleton } from "../components/Skeleton";
import { EmptyState } from "../components/EmptyState";
import { ErrorBanner } from "../components/ErrorBanner";
import { relativeTime, shortTime } from "../utils/time";

// Per-sender background tints (no side-stripe borders — per spec)
const SENDER_BG = [
  "var(--sender-0-bg)",
  "var(--sender-1-bg)",
  "var(--sender-2-bg)",
  "var(--sender-3-bg)",
  "var(--sender-4-bg)",
  "var(--sender-5-bg)",
];

function getSenderColor(index: number): string {
  return SENDER_BG[index % SENDER_BG.length];
}

// Stable sender → index mapping
function useSenderIndex(messages: A2AMessage[]) {
  return useMemo(() => {
    const map = new Map<string, number>();
    for (const msg of messages) {
      const s = msg.from ?? "";
      if (!map.has(s)) map.set(s, map.size);
    }
    return map;
  }, [messages]);
}

// ---- Channel list ----

function ChannelItem({
  channel,
  selected,
  onSelect,
}: {
  channel: Channel;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      onClick={onSelect}
      className="w-full text-left px-3 py-2.5 rounded transition-colors duration-150 flex flex-col gap-0.5"
      style={{
        background: selected ? "var(--accent-dim)" : "transparent",
        color: selected ? "var(--accent)" : "var(--ink)",
      }}
      aria-current={selected ? "true" : undefined}
    >
      <span className="text-sm font-medium truncate">#{channel.channel}</span>
      <span
        className="text-xs"
        style={{ color: selected ? "var(--accent)" : "var(--muted)" }}
      >
        {channel.message_count} msgs · {channel.members.length} members ·{" "}
        {relativeTime(channel.last_ts)}
      </span>
    </button>
  );
}

// ---- Message bubble ----

function MessageBubble({
  msg,
  senderIndex,
  isNew,
}: {
  msg: A2AMessage;
  senderIndex: number;
  isNew: boolean;
}) {
  const sender = msg.from ?? "?";
  const body = msg.body ?? "";
  const ts = shortTime(msg.ts);

  return (
    <article
      className={`rounded-lg px-4 py-3 ${isNew ? "msg-new" : ""}`}
      style={{
        background: getSenderColor(senderIndex),
        border: "1px solid var(--border-subtle)",
      }}
      aria-label={`Message from ${sender}`}
    >
      <div className="flex items-baseline gap-2 mb-1">
        <span
          className="text-xs font-semibold truncate max-w-[14ch]"
          style={{ color: "var(--muted-bright)" }}
          title={sender}
        >
          {sender}
        </span>
        <span
          className="text-xs shrink-0"
          style={{
            color: "var(--muted)",
            fontFamily: "var(--font-mono)",
          }}
        >
          {ts}
        </span>
        {msg.reply_to != null && (
          <span
            className="text-xs shrink-0"
            style={{
              color: "var(--muted)",
              fontFamily: "var(--font-mono)",
            }}
          >
            ↩ {String(msg.reply_to)}
          </span>
        )}
      </div>
      <p
        className="text-sm whitespace-pre-wrap break-words"
        style={{ color: "var(--ink)" }}
      >
        {body}
      </p>
    </article>
  );
}

// ---- Members panel ----

function MembersPanel({ channel }: { channel: string }) {
  const [members, setMembers] = useState<string[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setMembers(null);
    setError(null);
    getMembers(channel)
      .then(setMembers)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [channel]);

  return (
    <aside
      className="flex flex-col"
      style={{ borderLeft: "1px solid var(--border)" }}
      aria-label="Channel members"
    >
      <div
        className="px-4 py-3"
        style={{ borderBottom: "1px solid var(--border)" }}
      >
        <span
          className="text-xs font-medium"
          style={{ color: "var(--muted-bright)" }}
        >
          Members
        </span>
      </div>
      <div className="flex flex-col gap-1 px-3 py-2 overflow-y-auto flex-1">
        {error && (
          <span className="text-xs" style={{ color: "var(--error)" }}>
            {error}
          </span>
        )}
        {members === null && !error && (
          <Skeleton count={4} className="mt-1" />
        )}
        {members?.length === 0 && (
          <span className="text-xs" style={{ color: "var(--muted)" }}>
            No members
          </span>
        )}
        {members?.map((m) => (
          <span
            key={m}
            className="text-sm py-1 truncate"
            style={{ color: "var(--ink)" }}
            title={m}
          >
            {m}
          </span>
        ))}
      </div>
    </aside>
  );
}

// ---- Live indicator ----

type StreamStatus = "idle" | "connecting" | "live" | "reconnecting" | "disconnected";

function LiveBadge({ status }: { status: StreamStatus }) {
  const labels: Record<StreamStatus, string> = {
    idle: "",
    connecting: "Connecting",
    live: "Live",
    reconnecting: "Reconnecting",
    disconnected: "Disconnected",
  };

  if (status === "idle") return null;

  return (
    <span
      className="inline-flex items-center gap-1.5 rounded px-2 py-0.5 text-xs font-medium"
      style={{
        background:
          status === "live"
            ? "rgba(91,156,246,0.12)"
            : status === "disconnected"
            ? "rgba(248,113,113,0.1)"
            : "rgba(245,158,11,0.1)",
        color:
          status === "live"
            ? "var(--accent)"
            : status === "disconnected"
            ? "var(--error)"
            : "var(--warning)",
      }}
    >
      {status === "live" && (
        <span
          className="w-1.5 h-1.5 rounded-full"
          style={{ background: "var(--accent)" }}
          aria-hidden="true"
        />
      )}
      {labels[status]}
    </span>
  );
}

// ---- Message thread panel ----

function MessageThread({
  channel,
  onStatusChange,
}: {
  channel: Channel;
  onStatusChange: (s: StreamStatus) => void;
}) {
  const [messages, setMessages] = useState<A2AMessage[]>([]);
  const [newIds, setNewIds] = useState<Set<string | number>>(new Set());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const esRef = useRef<EventSource | null>(null);
  const lastTsRef = useRef<number>(0);
  const autoScroll = useRef(true);

  const senderIndex = useSenderIndex(messages);

  const scrollToBottom = useCallback((force = false) => {
    const el = listRef.current;
    if (!el) return;
    if (force || autoScroll.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, []);

  // Track whether user has scrolled away from bottom
  const handleScroll = () => {
    const el = listRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    autoScroll.current = atBottom;
  };

  const openStream = useCallback(
    (thread: string, since: number) => {
      if (esRef.current) {
        esRef.current.close();
        esRef.current = null;
      }
      onStatusChange("connecting");
      const es = new EventSource(
        `/a2a/stream?thread=${encodeURIComponent(thread)}&since=${since}`
      );
      esRef.current = es;
      es.onopen = () => onStatusChange("live");
      es.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data as string) as A2AMessage;
          if (msg.ts > lastTsRef.current) {
            lastTsRef.current = msg.ts;
            setMessages((prev) => [...prev, msg]);
            setNewIds((prev) => new Set([...prev, msg.id]));
            scrollToBottom();
          }
        } catch {}
      };
      es.onerror = () => {
        onStatusChange("disconnected");
        es.close();
        esRef.current = null;
      };
    },
    [onStatusChange, scrollToBottom]
  );

  // Load history + open stream when channel changes
  useEffect(() => {
    setMessages([]);
    setNewIds(new Set());
    setLoading(true);
    setError(null);
    lastTsRef.current = 0;
    autoScroll.current = true;

    getMessages(channel.channel, 200)
      .then((msgs) => {
        setMessages(msgs);
        const lastTs = msgs.length ? msgs[msgs.length - 1].ts : Date.now() / 1000;
        lastTsRef.current = lastTs;
        setLoading(false);
        // Wait for DOM update then scroll
        requestAnimationFrame(() => scrollToBottom(true));
        openStream(channel.channel, lastTs);
      })
      .catch((e) => {
        setError(e instanceof Error ? e.message : String(e));
        setLoading(false);
      });

    return () => {
      if (esRef.current) {
        esRef.current.close();
        esRef.current = null;
      }
      onStatusChange("idle");
    };
  }, [channel.channel, openStream, scrollToBottom, onStatusChange]);

  if (loading) {
    return (
      <div
        className="flex-1 flex flex-col gap-3 p-4 overflow-y-auto"
        aria-busy="true"
        aria-label="Loading messages"
      >
        {Array.from({ length: 6 }, (_, i) => (
          <div
            key={i}
            className="skeleton rounded-lg"
            style={{ height: i % 2 === 0 ? 64 : 48 }}
            aria-hidden="true"
          />
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex-1 p-4">
        <ErrorBanner message={error} />
      </div>
    );
  }

  if (!messages.length) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <EmptyState
          title={`#${channel.channel} is quiet`}
          description="No messages yet. Agents post via POST /a2a/send."
        />
      </div>
    );
  }

  return (
    <div
      ref={listRef}
      onScroll={handleScroll}
      className="flex-1 flex flex-col gap-2 px-4 py-4 overflow-y-auto"
      role="log"
      aria-live="polite"
      aria-label={`Messages in #${channel.channel}`}
    >
      {messages.map((msg) => (
        <MessageBubble
          key={msg.id}
          msg={msg}
          senderIndex={senderIndex.get(msg.from ?? "") ?? 0}
          isNew={newIds.has(msg.id)}
        />
      ))}
    </div>
  );
}

// ---- Root A2A view ----

type ChannelState =
  | { kind: "loading" }
  | { kind: "results"; channels: Channel[] }
  | { kind: "empty" }
  | { kind: "error"; message: string };

export function A2AView() {
  const [channelState, setChannelState] = useState<ChannelState>({ kind: "loading" });
  const [selected, setSelected] = useState<Channel | null>(null);
  const [streamStatus, setStreamStatus] = useState<StreamStatus>("idle");

  const loadChannels = useCallback(() => {
    setChannelState({ kind: "loading" });
    getChannels()
      .then((channels) => {
        if (!channels.length) {
          setChannelState({ kind: "empty" });
          return;
        }
        setChannelState({ kind: "results", channels });
        // Select first if none selected
        setSelected((prev) => prev ?? channels[0]);
      })
      .catch((e) =>
        setChannelState({
          kind: "error",
          message: e instanceof Error ? e.message : String(e),
        })
      );
  }, []);

  useEffect(() => {
    loadChannels();
  }, [loadChannels]);

  return (
    <div className="flex h-full" style={{ minHeight: 0 }}>
      {/* Channel sidebar */}
      <aside
        className="flex flex-col shrink-0"
        style={{
          width: 220,
          borderRight: "1px solid var(--border)",
          background: "var(--surface)",
        }}
        aria-label="Channels"
      >
        <div
          className="px-4 py-3 flex items-center gap-2"
          style={{ borderBottom: "1px solid var(--border)" }}
        >
          <span
            className="text-xs font-medium"
            style={{ color: "var(--muted-bright)" }}
          >
            Channels
          </span>
          <button
            onClick={loadChannels}
            className="ml-auto text-xs rounded px-1.5 py-0.5 transition-colors duration-150"
            style={{ color: "var(--muted)", background: "transparent" }}
            aria-label="Refresh channels"
            title="Refresh channels"
          >
            ↻
          </button>
        </div>

        <div className="flex-1 overflow-y-auto py-1 px-2">
          {channelState.kind === "loading" && (
            <div className="px-2 pt-2">
              <Skeleton count={4} />
            </div>
          )}
          {channelState.kind === "error" && (
            <p className="text-xs px-2 pt-2" style={{ color: "var(--error)" }}>
              {channelState.message}
            </p>
          )}
          {channelState.kind === "empty" && (
            <p className="text-xs px-2 pt-2" style={{ color: "var(--muted)" }}>
              No channels yet.
            </p>
          )}
          {channelState.kind === "results" &&
            channelState.channels.map((ch) => (
              <ChannelItem
                key={ch.channel}
                channel={ch}
                selected={selected?.channel === ch.channel}
                onSelect={() => setSelected(ch)}
              />
            ))}
        </div>
      </aside>

      {/* Main: message area */}
      {selected ? (
        <div className="flex flex-col flex-1 min-w-0">
          {/* Channel header */}
          <div
            className="px-5 py-3 flex items-center gap-3 shrink-0"
            style={{
              borderBottom: "1px solid var(--border)",
              background: "var(--surface)",
            }}
          >
            <div className="flex flex-col min-w-0">
              <span className="text-sm font-medium" style={{ color: "var(--ink)" }}>
                #{selected.channel}
              </span>
              <span className="text-xs" style={{ color: "var(--muted)" }}>
                Append-only memory — full history, nothing lost.
              </span>
            </div>
            <div className="ml-auto flex items-center gap-2">
              <LiveBadge status={streamStatus} />
              {streamStatus === "disconnected" && (
                <button
                  onClick={() => window.location.reload()}
                  className="text-xs underline underline-offset-2"
                  style={{ color: "var(--muted)" }}
                >
                  Reload
                </button>
              )}
            </div>
          </div>

          {/* Messages + members in a row */}
          <div className="flex flex-1 min-h-0">
            <MessageThread
              channel={selected}
              onStatusChange={setStreamStatus}
            />
            <div className="w-44 shrink-0 hidden lg:flex flex-col">
              <MembersPanel channel={selected.channel} />
            </div>
          </div>
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center">
          <EmptyState
            title="Select a channel"
            description="Choose a channel from the left to view its message history."
          />
        </div>
      )}
    </div>
  );
}
