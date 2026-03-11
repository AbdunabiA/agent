import { DollarSign, Zap, Activity } from "lucide-react";
import { MetricCard } from "@/components/overview/MetricCard";
import type { CostStats } from "@/lib/types";

interface CostSummaryProps {
  stats: CostStats | null;
  loading: boolean;
}

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

export function CostSummary({ stats, loading }: CostSummaryProps) {
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

  const totalTokens = stats.total_tokens.input + stats.total_tokens.output;

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
      <MetricCard
        icon={DollarSign}
        label="Total Cost"
        value={`$${stats.total_cost.toFixed(4)}`}
        subValue={`${stats.period} period`}
      />
      <MetricCard
        icon={Zap}
        label="Tokens"
        value={formatTokens(totalTokens)}
        subValue={`${formatTokens(stats.total_tokens.input)} in / ${formatTokens(stats.total_tokens.output)} out`}
      />
      <MetricCard
        icon={Activity}
        label="API Calls"
        value={stats.total_calls}
        subValue={`${stats.period} period`}
      />
    </div>
  );
}
