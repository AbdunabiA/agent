import { useState, useEffect, useCallback } from "react";
import {
  CheckCircle,
  XCircle,
  Clock,
  ShieldOff,
  Ban,
  ChevronRight,
  ChevronDown,
  ClipboardList,
} from "lucide-react";
import { Card } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { AuditDetail } from "./AuditDetail";
import { api } from "@/lib/api";
import type { AuditEntry } from "@/lib/types";

const statusIcons: Record<string, React.ElementType> = {
  success: CheckCircle,
  error: XCircle,
  timeout: Clock,
  denied: Ban,
  blocked: ShieldOff,
};

const statusColors: Record<string, string> = {
  success: "text-green-400",
  error: "text-red-400",
  timeout: "text-amber-400",
  denied: "text-red-400",
  blocked: "text-amber-400",
};

function durationColor(ms: number): string {
  if (ms < 100) return "text-green-400";
  if (ms < 1000) return "text-amber-400";
  return "text-red-400";
}

interface AuditTableProps {
  toolFilter: string;
  statusFilter: string;
}

export function AuditTable({ toolFilter, statusFilter }: AuditTableProps) {
  const [entries, setEntries] = useState<AuditEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const [limit, setLimit] = useState(50);

  const loadEntries = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.audit(limit);
      let filtered = data;
      if (toolFilter) {
        filtered = filtered.filter((e) => e.tool_name === toolFilter);
      }
      if (statusFilter) {
        filtered = filtered.filter((e) => e.status === statusFilter);
      }
      setEntries(filtered);
    } catch {
      setEntries([]);
    } finally {
      setLoading(false);
    }
  }, [limit, toolFilter, statusFilter]);

  useEffect(() => {
    loadEntries();
  }, [loadEntries]);

  function toggleExpand(id: string) {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  if (loading) {
    return (
      <Card>
        <div className="space-y-2">
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="h-10 animate-pulse rounded bg-gray-800" />
          ))}
        </div>
      </Card>
    );
  }

  if (entries.length === 0) {
    return (
      <Card>
        <EmptyState
          icon={ClipboardList}
          title="No audit entries"
          description="Tool executions will be logged here"
        />
      </Card>
    );
  }

  return (
    <Card className="overflow-hidden p-0">
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-700 text-left text-xs text-gray-500">
              <th className="px-4 py-3">Time</th>
              <th className="px-4 py-3">Tool</th>
              <th className="px-4 py-3 text-center">Status</th>
              <th className="px-4 py-3 text-right">Duration</th>
              <th className="px-4 py-3 w-10"></th>
            </tr>
          </thead>
          <tbody>
            {entries.map((entry) => {
              const StatusIcon = statusIcons[entry.status] || XCircle;
              const statusColor = statusColors[entry.status] || "text-gray-400";
              const isExpanded = expandedIds.has(entry.id);

              return (
                <>
                  <tr
                    key={entry.id}
                    onClick={() => toggleExpand(entry.id)}
                    className="cursor-pointer border-b border-gray-800 transition-colors hover:bg-gray-800/50"
                  >
                    <td className="px-4 py-3 text-xs text-gray-500 font-mono">
                      {new Date(entry.timestamp).toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                        second: "2-digit",
                      })}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-300 font-mono">
                      {entry.tool_name}
                    </td>
                    <td className="px-4 py-3 text-center">
                      <StatusIcon className={`inline-block h-4 w-4 ${statusColor}`} />
                    </td>
                    <td
                      className={`px-4 py-3 text-right text-xs font-mono ${durationColor(entry.duration_ms)}`}
                    >
                      {entry.duration_ms}ms
                    </td>
                    <td className="px-4 py-3 text-right">
                      {isExpanded ? (
                        <ChevronDown className="inline-block h-4 w-4 text-gray-500" />
                      ) : (
                        <ChevronRight className="inline-block h-4 w-4 text-gray-500" />
                      )}
                    </td>
                  </tr>
                  {isExpanded && (
                    <AuditDetail key={`${entry.id}-detail`} entry={entry} />
                  )}
                </>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="border-t border-gray-800 p-3 text-center">
        <button
          onClick={() => setLimit((prev) => prev + 50)}
          className="rounded-lg border border-gray-700 px-4 py-2 text-xs text-gray-400 transition-colors hover:border-gray-600 hover:text-gray-200"
        >
          Load more
        </button>
      </div>
    </Card>
  );
}
