import { ToolCard } from "./ToolCard";
import type { ToolInfo } from "@/lib/types";

interface ToolGridProps {
  tools: ToolInfo[];
  onToggle: () => void;
}

const tierOrder = ["safe", "moderate", "dangerous"] as const;
const tierLabels: Record<string, string> = {
  safe: "Safe",
  moderate: "Moderate",
  dangerous: "Dangerous",
};

export function ToolGrid({ tools, onToggle }: ToolGridProps) {
  const grouped = tierOrder
    .map((tier) => ({
      tier,
      label: tierLabels[tier],
      tools: tools.filter((t) => t.tier === tier),
    }))
    .filter((g) => g.tools.length > 0);

  return (
    <div className="space-y-6">
      {grouped.map(({ tier, label, tools: tierTools }) => (
        <div key={tier} className="space-y-3">
          <h2 className="text-sm font-medium text-gray-400">
            {label}{" "}
            <span className="text-gray-600">({tierTools.length})</span>
          </h2>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {tierTools.map((tool) => (
              <ToolCard key={tool.name} tool={tool} onToggle={onToggle} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
