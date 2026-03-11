import { Brain, Database, FileText } from "lucide-react";
import { MetricCard } from "@/components/overview/MetricCard";
import type { MemoryStats as MemoryStatsType } from "@/lib/types";

interface MemoryStatsProps {
  stats: MemoryStatsType | null;
  loading: boolean;
}

export function MemoryStats({ stats, loading }: MemoryStatsProps) {
  if (loading || !stats) {
    return (
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        {[1, 2, 3].map((i) => (
          <div
            key={i}
            className="h-20 animate-pulse rounded-xl border border-gray-800 bg-gray-900"
          />
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
      <MetricCard
        icon={Database}
        label="Facts"
        value={stats.facts_count}
        subValue="Structured memories"
      />
      <MetricCard
        icon={Brain}
        label="Vectors"
        value={stats.vectors_count.toLocaleString()}
        subValue="Semantic chunks"
      />
      <MetricCard
        icon={FileText}
        label="soul.md"
        value={stats.soul_loaded ? "Loaded" : "Default"}
        subValue={stats.soul_loaded ? "Custom personality" : "Using built-in"}
      />
    </div>
  );
}
