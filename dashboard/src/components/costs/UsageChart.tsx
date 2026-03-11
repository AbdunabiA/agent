interface UsageChartProps {
  data: { time: string; cost: number; tokens: number }[];
}

export function UsageChart({ data }: UsageChartProps) {
  if (data.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-gray-500">
        No usage data for this period
      </p>
    );
  }

  const maxCost = Math.max(...data.map((d) => d.cost), 0.001);

  return (
    <div className="space-y-2">
      {data.map((d) => (
        <div key={d.time} className="flex items-center gap-3">
          <span className="w-14 shrink-0 text-right text-xs text-gray-500 font-mono">
            {d.time}
          </span>
          <div className="flex-1 overflow-hidden rounded-full bg-gray-800 h-6">
            <div
              className="h-full rounded-full bg-blue-500 transition-all duration-300"
              style={{ width: `${(d.cost / maxCost) * 100}%` }}
            />
          </div>
          <span className="w-16 shrink-0 text-xs text-gray-300 font-mono">
            ${d.cost.toFixed(4)}
          </span>
        </div>
      ))}
    </div>
  );
}
