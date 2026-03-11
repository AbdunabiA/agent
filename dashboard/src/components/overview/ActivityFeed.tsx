import { useEffect, useRef } from "react";
import {
  MessageSquare,
  Send,
  Wrench,
  CheckCircle,
  Heart,
  AlertTriangle,
  Zap,
  Trash2,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { AgentEvent } from "@/lib/types";
import { Card } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { formatRelativeTime, cn } from "@/lib/utils";

interface ActivityFeedProps {
  events: AgentEvent[];
  connected: boolean;
  onClear: () => void;
}

const EVENT_ICON_MAP: Record<string, LucideIcon> = {
  "message.incoming": MessageSquare,
  "message.outgoing": Send,
  "tool.execute": Wrench,
  "tool.result": CheckCircle,
  "heartbeat.tick": Heart,
  "agent.error": AlertTriangle,
};

function getEventIcon(event: string): LucideIcon {
  return EVENT_ICON_MAP[event] ?? Zap;
}

function getEventColor(event: string): string {
  if (event.startsWith("message.")) return "text-blue-400";
  if (event.startsWith("tool.")) return "text-amber-400";
  if (event.startsWith("heartbeat.")) return "text-green-400";
  if (event.includes("error")) return "text-red-400";
  return "text-gray-400";
}

export function ActivityFeed({ events, connected, onClear }: ActivityFeedProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [events]);

  return (
    <Card className="flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-medium text-gray-200">Activity</h3>
          <span
            className={cn(
              "h-2 w-2 rounded-full",
              connected ? "bg-green-500" : "bg-red-500",
            )}
          />
        </div>
        {events.length > 0 && (
          <button
            onClick={onClear}
            className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-300 transition-colors"
          >
            <Trash2 className="h-3 w-3" />
            Clear
          </button>
        )}
      </div>

      {/* Event list */}
      {events.length === 0 ? (
        <EmptyState icon={Zap} title="No events yet" description="Events will appear here in real time." />
      ) : (
        <div ref={scrollRef} className="max-h-96 overflow-y-auto space-y-2">
          {events.map((evt, idx) => {
            const Icon = getEventIcon(evt.event);
            const color = getEventColor(evt.event);
            const dataPreview =
              Object.keys(evt.data).length > 0
                ? JSON.stringify(evt.data).slice(0, 120)
                : null;

            return (
              <div
                key={`${evt.timestamp}-${idx}`}
                className="flex items-start gap-2 rounded-lg bg-gray-800/50 px-3 py-2"
              >
                <Icon className={cn("h-4 w-4 mt-0.5 shrink-0", color)} />
                <div className="min-w-0 flex-1">
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="text-sm font-medium text-gray-200 truncate">
                      {evt.event}
                    </span>
                    <span className="text-xs text-gray-500 shrink-0">
                      {formatRelativeTime(evt.timestamp)}
                    </span>
                  </div>
                  {dataPreview && (
                    <p className="text-xs text-gray-500 mt-0.5 truncate">{dataPreview}</p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </Card>
  );
}
