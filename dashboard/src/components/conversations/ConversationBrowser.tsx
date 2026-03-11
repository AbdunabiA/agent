import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { MessageSquare, Smartphone, Monitor, Terminal, ExternalLink } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { api } from "@/lib/api";
import type { SessionSummary, MessageOut } from "@/lib/types";

const channelIcons: Record<string, React.ElementType> = {
  telegram: Smartphone,
  webchat: Monitor,
  cli: Terminal,
  api: Terminal,
};

function formatTimeAgo(dateStr: string): string {
  const now = Date.now();
  const then = new Date(dateStr).getTime();
  const diff = now - then;
  const mins = Math.floor(diff / 60000);
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export function ConversationBrowser() {
  const navigate = useNavigate();
  const [conversations, setConversations] = useState<SessionSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [messages, setMessages] = useState<MessageOut[]>([]);
  const [loadingMessages, setLoadingMessages] = useState(false);
  const [channelFilter, setChannelFilter] = useState<string>("all");

  useEffect(() => {
    api
      .conversations(100)
      .then(setConversations)
      .catch(() => setConversations([]))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!selectedId) {
      setMessages([]);
      return;
    }
    setLoadingMessages(true);
    api
      .messages(selectedId, 50)
      .then(setMessages)
      .catch(() => setMessages([]))
      .finally(() => setLoadingMessages(false));
  }, [selectedId]);

  const filtered =
    channelFilter === "all"
      ? conversations
      : conversations.filter((c) => c.channel === channelFilter);

  const channels = [...new Set(conversations.map((c) => c.channel))];

  return (
    <div className="flex h-[calc(100vh-10rem)] gap-4">
      {/* Left panel: conversation list */}
      <Card className="flex w-72 shrink-0 flex-col overflow-hidden">
        <div className="border-b border-gray-800 p-3">
          <select
            value={channelFilter}
            onChange={(e) => setChannelFilter(e.target.value)}
            className="w-full rounded-md border border-gray-700 bg-gray-800 px-3 py-1.5 text-xs text-gray-300 focus:border-indigo-500 focus:outline-none"
          >
            <option value="all">All channels</option>
            {channels.map((ch) => (
              <option key={ch} value={ch}>
                {ch}
              </option>
            ))}
          </select>
        </div>

        <div className="flex-1 overflow-y-auto">
          {loading ? (
            <div className="space-y-2 p-3">
              {[1, 2, 3].map((i) => (
                <div key={i} className="h-14 animate-pulse rounded-lg bg-gray-800" />
              ))}
            </div>
          ) : filtered.length === 0 ? (
            <EmptyState
              icon={MessageSquare}
              title="No conversations"
              description="Start chatting to see conversations here"
            />
          ) : (
            <div className="space-y-1 p-2">
              {filtered.map((conv) => {
                const Icon = channelIcons[conv.channel] || Terminal;
                const isSelected = selectedId === conv.id;
                return (
                  <button
                    key={conv.id}
                    onClick={() => setSelectedId(conv.id)}
                    className={`flex w-full items-start gap-3 rounded-lg p-3 text-left transition-colors ${
                      isSelected
                        ? "bg-indigo-600/20 text-indigo-400"
                        : "text-gray-400 hover:bg-gray-800"
                    }`}
                  >
                    <Icon className="mt-0.5 h-4 w-4 shrink-0" />
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium text-gray-200">
                        {conv.channel} session
                      </p>
                      <p className="text-xs text-gray-500">
                        {conv.message_count} msgs · {formatTimeAgo(conv.updated_at)}
                      </p>
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </div>
      </Card>

      {/* Right panel: conversation preview */}
      <Card className="flex flex-1 flex-col overflow-hidden">
        {!selectedId ? (
          <div className="flex flex-1 items-center justify-center">
            <EmptyState
              icon={MessageSquare}
              title="Select a conversation"
              description="Click a conversation on the left to preview"
            />
          </div>
        ) : loadingMessages ? (
          <div className="space-y-3 p-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-16 animate-pulse rounded-lg bg-gray-800" />
            ))}
          </div>
        ) : (
          <>
            <div className="flex-1 space-y-3 overflow-y-auto p-4">
              {messages.map((msg, i) => (
                <div
                  key={i}
                  className={`rounded-lg p-3 ${
                    msg.role === "user"
                      ? "ml-8 bg-indigo-600/20"
                      : msg.role === "tool"
                        ? "border border-gray-700 bg-gray-800/50"
                        : "mr-8 bg-gray-800"
                  }`}
                >
                  <p className="mb-1 text-xs font-medium text-gray-500">
                    {msg.role === "user"
                      ? "User"
                      : msg.role === "tool"
                        ? "Tool Result"
                        : "Agent"}
                  </p>
                  <p className="whitespace-pre-wrap text-sm text-gray-300">
                    {msg.content.length > 500
                      ? msg.content.slice(0, 500) + "..."
                      : msg.content}
                  </p>
                  {msg.tool_calls && msg.tool_calls.length > 0 && (
                    <div className="mt-2 space-y-1">
                      {msg.tool_calls.map((tc) => (
                        <span
                          key={tc.id}
                          className="inline-block rounded bg-amber-900/30 px-2 py-0.5 text-xs text-amber-400"
                        >
                          🔧 {tc.name}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
            <div className="border-t border-gray-800 p-3">
              <button
                onClick={() => navigate(`/chat/${selectedId}`)}
                className="flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-500"
              >
                <ExternalLink className="h-4 w-4" />
                Open in Chat
              </button>
            </div>
          </>
        )}
      </Card>
    </div>
  );
}
