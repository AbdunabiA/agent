import { useState, useEffect } from "react";
import { Card } from "@/components/ui/Card";
import { CostSummary } from "@/components/costs/CostSummary";
import { UsageChart } from "@/components/costs/UsageChart";
import { api } from "@/lib/api";
import type { CostStats } from "@/lib/types";

const PERIODS = [
  { value: "day", label: "Day" },
  { value: "week", label: "Week" },
  { value: "month", label: "Month" },
];

export function CostsPage() {
  const [period, setPeriod] = useState("day");
  const [stats, setStats] = useState<CostStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api
      .costStats(period)
      .then(setStats)
      .catch(() => setStats(null))
      .finally(() => setLoading(false));
  }, [period]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-100">Costs</h1>
        <div className="flex gap-1 rounded-lg border border-gray-700 p-1">
          {PERIODS.map((p) => (
            <button
              key={p.value}
              onClick={() => setPeriod(p.value)}
              className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
                period === p.value
                  ? "bg-indigo-600 text-white"
                  : "text-gray-400 hover:text-gray-200"
              }`}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      <CostSummary stats={stats} loading={loading} />

      <Card>
        <h3 className="mb-4 text-sm font-medium text-gray-300">Usage Over Time</h3>
        {loading ? (
          <div className="space-y-2">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-6 animate-pulse rounded-full bg-gray-800" />
            ))}
          </div>
        ) : (
          <UsageChart data={stats?.by_time ?? []} />
        )}
      </Card>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {/* By Model */}
        <Card>
          <h3 className="mb-4 text-sm font-medium text-gray-300">By Model</h3>
          {!stats || stats.by_model.length === 0 ? (
            <p className="py-4 text-center text-sm text-gray-500">No data</p>
          ) : (
            <div className="space-y-3">
              {stats.by_model.map((m) => (
                <div key={m.model} className="flex items-center gap-3">
                  <span className="w-36 shrink-0 truncate text-sm text-gray-300 font-mono">
                    {m.model}
                  </span>
                  <div className="flex-1 overflow-hidden rounded-full bg-gray-800 h-5">
                    <div
                      className="h-full rounded-full bg-indigo-500 transition-all duration-300"
                      style={{ width: `${m.percentage}%` }}
                    />
                  </div>
                  <span className="w-16 shrink-0 text-right text-xs text-gray-400 font-mono">
                    ${m.cost.toFixed(4)}
                  </span>
                  <span className="w-10 shrink-0 text-right text-xs text-gray-500">
                    {m.percentage.toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          )}
        </Card>

        {/* By Channel */}
        <Card>
          <h3 className="mb-4 text-sm font-medium text-gray-300">By Channel</h3>
          {!stats || stats.by_channel.length === 0 ? (
            <p className="py-4 text-center text-sm text-gray-500">No data</p>
          ) : (
            <div className="space-y-3">
              {(() => {
                const maxCost = Math.max(...stats.by_channel.map((c) => c.cost), 0.001);
                return stats.by_channel.map((c) => (
                  <div key={c.channel} className="flex items-center gap-3">
                    <span className="w-24 shrink-0 text-sm text-gray-300">{c.channel}</span>
                    <div className="flex-1 overflow-hidden rounded-full bg-gray-800 h-5">
                      <div
                        className="h-full rounded-full bg-emerald-500 transition-all duration-300"
                        style={{ width: `${(c.cost / maxCost) * 100}%` }}
                      />
                    </div>
                    <span className="w-16 shrink-0 text-right text-xs text-gray-400 font-mono">
                      ${c.cost.toFixed(4)}
                    </span>
                  </div>
                ));
              })()}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
