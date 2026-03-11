import { useState } from "react";
import { Card } from "@/components/ui/Card";
import { api } from "@/lib/api";
import type { ToolInfo } from "@/lib/types";

interface ToolCardProps {
  tool: ToolInfo;
  onToggle: () => void;
}

const tierColors = {
  safe: { bg: "bg-green-600/20", text: "text-green-400", label: "Safe" },
  moderate: { bg: "bg-yellow-600/20", text: "text-yellow-400", label: "Moderate" },
  dangerous: { bg: "bg-red-600/20", text: "text-red-400", label: "Dangerous" },
} as const;

export function ToolCard({ tool, onToggle }: ToolCardProps) {
  const [toggling, setToggling] = useState(false);
  const tier = tierColors[tool.tier];

  const handleToggle = async () => {
    setToggling(true);
    try {
      await api.toggleTool(tool.name, !tool.enabled);
      onToggle();
    } catch {
      // ignore
    } finally {
      setToggling(false);
    }
  };

  return (
    <Card className={!tool.enabled ? "opacity-60" : ""}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1 space-y-1">
          <div className="flex items-center gap-2">
            <h3 className="truncate text-sm font-medium text-gray-100">{tool.name}</h3>
            <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${tier.bg} ${tier.text}`}>
              {tier.label}
            </span>
          </div>
          <p className="text-xs text-gray-400 line-clamp-2">{tool.description}</p>
        </div>

        <button
          onClick={handleToggle}
          disabled={toggling}
          className={`relative h-6 w-11 shrink-0 rounded-full transition-colors ${
            tool.enabled ? "bg-indigo-600" : "bg-gray-700"
          }`}
          aria-label={`${tool.enabled ? "Disable" : "Enable"} ${tool.name}`}
        >
          <span
            className={`absolute left-0.5 top-0.5 h-5 w-5 rounded-full bg-white transition-transform ${
              tool.enabled ? "translate-x-5" : "translate-x-0"
            }`}
          />
        </button>
      </div>
    </Card>
  );
}
