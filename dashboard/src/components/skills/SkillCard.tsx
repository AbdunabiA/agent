import { useState } from "react";
import { RefreshCw } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { api } from "@/lib/api";
import type { SkillInfo } from "@/lib/types";

interface SkillCardProps {
  skill: SkillInfo;
  onRefresh: () => void;
}

export function SkillCard({ skill, onRefresh }: SkillCardProps) {
  const [toggling, setToggling] = useState(false);
  const [reloading, setReloading] = useState(false);

  const handleToggle = async () => {
    setToggling(true);
    try {
      if (skill.loaded) {
        await api.disableSkill(skill.name);
      } else {
        await api.enableSkill(skill.name);
      }
      onRefresh();
    } catch {
      // ignore
    } finally {
      setToggling(false);
    }
  };

  const handleReload = async () => {
    setReloading(true);
    try {
      await api.reloadSkill(skill.name);
      onRefresh();
    } catch {
      // ignore
    } finally {
      setReloading(false);
    }
  };

  return (
    <Card className={!skill.loaded ? "opacity-60" : ""}>
      <div className="space-y-2">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 flex-1 space-y-1">
            <div className="flex items-center gap-2">
              <h3 className="truncate text-sm font-medium text-gray-100">
                {skill.display_name || skill.name}
              </h3>
              <span className="rounded bg-gray-700/50 px-1.5 py-0.5 text-[10px] text-gray-400">
                v{skill.version}
              </span>
            </div>
            <p className="text-xs text-gray-400 line-clamp-2">{skill.description || "No description"}</p>
          </div>

          <div className="flex shrink-0 items-center gap-2">
            {skill.loaded && (
              <button
                onClick={handleReload}
                disabled={reloading}
                className="rounded p-1 text-gray-500 transition-colors hover:bg-gray-800 hover:text-gray-300"
                aria-label={`Reload ${skill.name}`}
              >
                <RefreshCw className={`h-3.5 w-3.5 ${reloading ? "animate-spin" : ""}`} />
              </button>
            )}

            <button
              onClick={handleToggle}
              disabled={toggling}
              className={`relative h-6 w-11 shrink-0 rounded-full transition-colors ${
                skill.loaded ? "bg-indigo-600" : "bg-gray-700"
              }`}
              aria-label={`${skill.loaded ? "Disable" : "Enable"} ${skill.name}`}
            >
              <span
                className={`absolute left-0.5 top-0.5 h-5 w-5 rounded-full bg-white transition-transform ${
                  skill.loaded ? "translate-x-5" : "translate-x-0"
                }`}
              />
            </button>
          </div>
        </div>

        {/* Metadata */}
        <div className="flex flex-wrap items-center gap-2 text-[10px] text-gray-500">
          {skill.permissions.map((p) => (
            <span
              key={p}
              className={`rounded px-1.5 py-0.5 ${
                p === "safe"
                  ? "bg-green-600/20 text-green-400"
                  : p === "moderate"
                    ? "bg-yellow-600/20 text-yellow-400"
                    : "bg-red-600/20 text-red-400"
              }`}
            >
              {p}
            </span>
          ))}
          {skill.tools.length > 0 && (
            <span className="text-gray-500">
              {skill.tools.length} tool{skill.tools.length !== 1 ? "s" : ""}
            </span>
          )}
          {skill.author && <span>by {skill.author}</span>}
        </div>
      </div>
    </Card>
  );
}
