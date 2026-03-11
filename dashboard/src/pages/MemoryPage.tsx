import { useState, useEffect } from "react";
import { MemoryStats } from "@/components/memory/MemoryStats";
import { SemanticSearch } from "@/components/memory/SemanticSearch";
import { FactsTable } from "@/components/memory/FactsTable";
import { api } from "@/lib/api";
import type { MemoryStats as MemoryStatsType } from "@/lib/types";

export function MemoryPage() {
  const [stats, setStats] = useState<MemoryStatsType | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .memoryStats()
      .then(setStats)
      .catch(() => setStats({ facts_count: 0, vectors_count: 0, soul_loaded: false }))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-100">Memory</h1>
      <MemoryStats stats={stats} loading={loading} />
      <SemanticSearch />
      <FactsTable />
    </div>
  );
}
