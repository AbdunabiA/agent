import { useState, useEffect, useCallback } from "react";
import { Wrench, Shield, ShieldAlert, ShieldCheck } from "lucide-react";
import { ToolGrid } from "@/components/tools/ToolGrid";
import { Spinner } from "@/components/ui/Spinner";
import { EmptyState } from "@/components/ui/EmptyState";
import { api } from "@/lib/api";
import type { ToolInfo } from "@/lib/types";

export function ToolsPage() {
  const [tools, setTools] = useState<ToolInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("");

  const fetchTools = useCallback(() => {
    api
      .tools()
      .then(setTools)
      .catch(() => setTools([]))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetchTools();
  }, [fetchTools]);

  const filtered = filter ? tools.filter((t) => t.tier === filter) : tools;

  const stats = {
    total: tools.length,
    enabled: tools.filter((t) => t.enabled).length,
    safe: tools.filter((t) => t.tier === "safe").length,
    moderate: tools.filter((t) => t.tier === "moderate").length,
    dangerous: tools.filter((t) => t.tier === "dangerous").length,
  };

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <Spinner size="lg" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-100">Tools</h1>
        <div className="flex items-center gap-3 text-xs text-gray-500">
          <span>{stats.enabled}/{stats.total} enabled</span>
        </div>
      </div>

      {/* Stats & Filter */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-3 text-xs text-gray-400">
          <span className="flex items-center gap-1">
            <ShieldCheck className="h-3.5 w-3.5 text-green-400" />
            {stats.safe}
          </span>
          <span className="flex items-center gap-1">
            <Shield className="h-3.5 w-3.5 text-yellow-400" />
            {stats.moderate}
          </span>
          <span className="flex items-center gap-1">
            <ShieldAlert className="h-3.5 w-3.5 text-red-400" />
            {stats.dangerous}
          </span>
        </div>
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="rounded-md border border-gray-700 bg-gray-800 px-3 py-1.5 text-xs text-gray-300 focus:border-indigo-500 focus:outline-none"
        >
          <option value="">All tiers</option>
          <option value="safe">Safe</option>
          <option value="moderate">Moderate</option>
          <option value="dangerous">Dangerous</option>
        </select>
      </div>

      {filtered.length === 0 ? (
        <EmptyState icon={Wrench} title="No tools found" description="No tools match the current filter." />
      ) : (
        <ToolGrid tools={filtered} onToggle={fetchTools} />
      )}
    </div>
  );
}
