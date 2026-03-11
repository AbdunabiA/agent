import { useMemo } from "react";
import { Plus } from "lucide-react";
import type { SessionSummary } from "@/lib/types";
import { cn, dateGroupLabel, deriveTitle, formatRelativeTime } from "@/lib/utils";

interface ConversationListProps {
  sessions: SessionSummary[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNewChat: () => void;
}

interface GroupedSessions {
  label: string;
  sessions: SessionSummary[];
}

export function ConversationList({
  sessions,
  activeId,
  onSelect,
  onNewChat,
}: ConversationListProps) {
  // Group sessions by date
  const grouped = useMemo<GroupedSessions[]>(() => {
    const map = new Map<string, SessionSummary[]>();
    for (const session of sessions) {
      const label = dateGroupLabel(session.updated_at);
      const group = map.get(label);
      if (group) {
        group.push(session);
      } else {
        map.set(label, [session]);
      }
    }
    return Array.from(map.entries()).map(([label, items]) => ({
      label,
      sessions: items,
    }));
  }, [sessions]);

  return (
    <div className="flex h-full flex-col">
      {/* New Chat button */}
      <div className="p-3">
        <button
          type="button"
          onClick={onNewChat}
          className={cn(
            "flex w-full items-center justify-center gap-2 rounded-lg px-4 py-2.5",
            "bg-indigo-600 text-sm font-medium text-white",
            "hover:bg-indigo-500 active:bg-indigo-700 transition-colors",
          )}
        >
          <Plus className="h-4 w-4" />
          New Chat
        </button>
      </div>

      {/* Scrollable session list */}
      <div className="flex-1 overflow-y-auto px-2 pb-4">
        {grouped.map((group) => (
          <div key={group.label} className="mb-3">
            {/* Date group label */}
            <p className="px-2 pb-1 pt-2 text-xs font-medium uppercase tracking-wider text-gray-500">
              {group.label}
            </p>

            {/* Session items */}
            {group.sessions.map((session) => {
              const isActive = session.id === activeId;
              return (
                <button
                  key={session.id}
                  type="button"
                  onClick={() => onSelect(session.id)}
                  className={cn(
                    "flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left transition-colors",
                    isActive
                      ? "bg-gray-800 border-l-2 border-indigo-500"
                      : "hover:bg-gray-800/50 border-l-2 border-transparent",
                  )}
                >
                  <div className="flex-1 min-w-0">
                    <p
                      className={cn(
                        "truncate text-sm",
                        isActive ? "text-gray-100 font-medium" : "text-gray-300",
                      )}
                    >
                      {deriveTitle(session.channel, session.id)}
                    </p>
                    <p className="text-xs text-gray-500 mt-0.5">
                      {formatRelativeTime(session.updated_at)}
                    </p>
                  </div>

                  {/* Message count badge */}
                  <span
                    className={cn(
                      "shrink-0 rounded-full px-2 py-0.5 text-xs",
                      isActive
                        ? "bg-indigo-600/30 text-indigo-300"
                        : "bg-gray-700 text-gray-400",
                    )}
                  >
                    {session.message_count}
                  </span>
                </button>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}
