import { useState, useEffect } from "react";
import {
  CheckCircle,
  AlertCircle,
  XCircle,
  Wrench,
  Heart,
  MessageSquare,
  Clock,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import { Card } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { api } from "@/lib/api";
import type { TimelineEvent } from "@/lib/types";

const iconMap: Record<string, React.ElementType> = {
  "check-circle": CheckCircle,
  "alert-circle": AlertCircle,
  "x-circle": XCircle,
  wrench: Wrench,
  heart: Heart,
  message: MessageSquare,
};

const iconColorMap: Record<string, string> = {
  "check-circle": "text-green-400",
  "alert-circle": "text-red-400",
  "x-circle": "text-amber-400",
  wrench: "text-blue-400",
  heart: "text-pink-400",
  message: "text-indigo-400",
};

const dotColorMap: Record<string, string> = {
  "check-circle": "bg-green-400",
  "alert-circle": "bg-red-400",
  "x-circle": "bg-amber-400",
  wrench: "bg-blue-400",
  heart: "bg-pink-400",
  message: "bg-indigo-400",
};

const EVENT_TYPES = [
  { value: "tool.success", label: "Success" },
  { value: "tool.error", label: "Error" },
  { value: "tool.denied", label: "Denied" },
  { value: "tool.timeout", label: "Timeout" },
];

function formatTime(ts: string): string {
  return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function groupByDate(events: TimelineEvent[]): Map<string, TimelineEvent[]> {
  const groups = new Map<string, TimelineEvent[]>();
  const today = new Date().toLocaleDateString();
  const yesterday = new Date(Date.now() - 86400000).toLocaleDateString();

  for (const ev of events) {
    const dateStr = new Date(ev.timestamp).toLocaleDateString();
    const label = dateStr === today ? "Today" : dateStr === yesterday ? "Yesterday" : dateStr;
    if (!groups.has(label)) groups.set(label, []);
    groups.get(label)!.push(ev);
  }
  return groups;
}

export function TimelineView() {
  const [events, setEvents] = useState<TimelineEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState<Set<string>>(new Set());
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());

  useEffect(() => {
    const filterStr = filters.size > 0 ? [...filters].join(",") : undefined;
    setLoading(true);
    api
      .timeline(200, filterStr)
      .then((data) => setEvents(data.events))
      .catch(() => setEvents([]))
      .finally(() => setLoading(false));
  }, [filters]);

  function toggleFilter(value: string) {
    setFilters((prev) => {
      const next = new Set(prev);
      if (next.has(value)) next.delete(value);
      else next.add(value);
      return next;
    });
  }

  function toggleExpand(id: string) {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  const groups = groupByDate(events);

  return (
    <div className="space-y-4">
      {/* Filters */}
      <div className="flex flex-wrap gap-2">
        {EVENT_TYPES.map((et) => (
          <button
            key={et.value}
            onClick={() => toggleFilter(et.value)}
            className={`rounded-full border px-3 py-1 text-xs font-medium transition-colors ${
              filters.has(et.value)
                ? "border-indigo-500 bg-indigo-600/20 text-indigo-400"
                : "border-gray-700 text-gray-400 hover:border-gray-600"
            }`}
          >
            {et.label}
          </button>
        ))}
      </div>

      {/* Timeline */}
      {loading ? (
        <Card>
          <div className="space-y-4">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-12 animate-pulse rounded bg-gray-800" />
            ))}
          </div>
        </Card>
      ) : events.length === 0 ? (
        <Card>
          <EmptyState
            icon={Clock}
            title="No events"
            description="Agent events will appear here as they occur"
          />
        </Card>
      ) : (
        [...groups.entries()].map(([date, groupEvents]) => (
          <div key={date}>
            <div className="mb-3 flex items-center gap-3">
              <span className="text-sm font-medium text-gray-400">{date}</span>
              <div className="h-px flex-1 bg-gray-800" />
            </div>

            <div className="relative ml-4 border-l-2 border-gray-800 pl-6">
              {groupEvents.map((ev) => {
                const icon = ev.icon || "wrench";
                const Icon = iconMap[icon] || Wrench;
                const colorClass = iconColorMap[icon] || "text-gray-400";
                const dotColor = dotColorMap[icon] || "bg-gray-400";
                const isExpanded = expandedIds.has(ev.id);

                return (
                  <div key={ev.id} className="group relative mb-4 last:mb-0">
                    {/* Dot on the timeline */}
                    <div
                      className={`absolute -left-[31px] top-1.5 h-3 w-3 rounded-full border-2 border-gray-900 ${dotColor}`}
                    />

                    <button
                      onClick={() => toggleExpand(ev.id)}
                      className="flex w-full items-start gap-3 rounded-lg p-2 text-left transition-colors hover:bg-gray-800/50"
                    >
                      <span className="mt-0.5 text-xs text-gray-500 font-mono shrink-0 w-12">
                        {formatTime(ev.timestamp)}
                      </span>
                      <Icon className={`mt-0.5 h-4 w-4 shrink-0 ${colorClass}`} />
                      <span className="flex-1 text-sm text-gray-300">{ev.description}</span>
                      {isExpanded ? (
                        <ChevronDown className="h-4 w-4 shrink-0 text-gray-500" />
                      ) : (
                        <ChevronRight className="h-4 w-4 shrink-0 text-gray-500" />
                      )}
                    </button>

                    {isExpanded && ev.details && (
                      <div className="ml-16 mt-1 rounded-lg border border-gray-800 bg-gray-800/50 p-3">
                        <pre className="text-xs text-gray-400 whitespace-pre-wrap">
                          {JSON.stringify(ev.details, null, 2)}
                        </pre>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        ))
      )}
    </div>
  );
}
