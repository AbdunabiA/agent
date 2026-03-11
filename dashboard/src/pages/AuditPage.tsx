import { useState, useEffect } from "react";
import { ClipboardList, CheckCircle, Clock } from "lucide-react";
import { MetricCard } from "@/components/overview/MetricCard";
import { AuditTable } from "@/components/audit/AuditTable";
import { api } from "@/lib/api";
import type { AuditStats } from "@/lib/types";

export function AuditPage() {
  const [stats, setStats] = useState<AuditStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [toolFilter, setToolFilter] = useState("");
  const [statusFilter, setStatusFilter] = useState("");

  useEffect(() => {
    api
      .auditStats()
      .then(setStats)
      .catch(() => setStats(null))
      .finally(() => setLoading(false));
  }, []);

  const toolNames = stats?.tools_used ? Object.keys(stats.tools_used) : [];

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-100">Audit Log</h1>

      {/* Stats bar */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        {loading ? (
          [1, 2, 3].map((i) => (
            <div
              key={i}
              className="h-20 animate-pulse rounded-xl border border-gray-800 bg-gray-900"
            />
          ))
        ) : (
          <>
            <MetricCard
              icon={ClipboardList}
              label="Total Calls"
              value={stats?.total_calls ?? 0}
            />
            <MetricCard
              icon={CheckCircle}
              label="Success Rate"
              value={`${((stats?.success_rate ?? 0) * 100).toFixed(0)}%`}
              subValue={`${stats?.success_count ?? 0} success / ${stats?.error_count ?? 0} errors`}
            />
            <MetricCard
              icon={Clock}
              label="Avg Duration"
              value={`${stats?.avg_duration_ms ?? 0}ms`}
            />
          </>
        )}
      </div>

      {/* Filters */}
      <div className="flex gap-3">
        <select
          value={toolFilter}
          onChange={(e) => setToolFilter(e.target.value)}
          className="rounded-md border border-gray-700 bg-gray-800 px-3 py-1.5 text-xs text-gray-300 focus:border-indigo-500 focus:outline-none"
        >
          <option value="">All tools</option>
          {toolNames.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          className="rounded-md border border-gray-700 bg-gray-800 px-3 py-1.5 text-xs text-gray-300 focus:border-indigo-500 focus:outline-none"
        >
          <option value="">All statuses</option>
          <option value="success">Success</option>
          <option value="error">Error</option>
          <option value="timeout">Timeout</option>
          <option value="denied">Denied</option>
          <option value="blocked">Blocked</option>
        </select>
      </div>

      <AuditTable toolFilter={toolFilter} statusFilter={statusFilter} />
    </div>
  );
}
