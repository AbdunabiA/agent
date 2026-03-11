import { Pause, Play, VolumeX, Volume2 } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { ControlAction } from "@/lib/types";
import { Card } from "@/components/ui/Card";
import { cn } from "@/lib/utils";

interface QuickActionsProps {
  onAction: (action: ControlAction) => void;
  loading?: boolean;
}

interface ActionButton {
  action: ControlAction;
  label: string;
  icon: LucideIcon;
}

const actions: ActionButton[] = [
  { action: "pause", label: "Pause", icon: Pause },
  { action: "resume", label: "Resume", icon: Play },
  { action: "mute", label: "Mute", icon: VolumeX },
  { action: "unmute", label: "Unmute", icon: Volume2 },
];

export function QuickActions({ onAction, loading }: QuickActionsProps) {
  return (
    <Card className="flex flex-col">
      <h3 className="text-sm font-medium text-gray-200 mb-3">Quick Actions</h3>
      <div className="grid grid-cols-2 gap-2">
        {actions.map(({ action, label, icon: Icon }) => (
          <button
            key={action}
            onClick={() => onAction(action)}
            disabled={loading}
            className={cn(
              "flex items-center gap-2 bg-gray-800 hover:bg-gray-700 rounded-lg px-4 py-3 transition-colors",
              "text-sm text-gray-200",
              loading && "opacity-50 cursor-not-allowed",
            )}
          >
            <Icon className="h-4 w-4 text-gray-400" />
            {label}
          </button>
        ))}
      </div>
    </Card>
  );
}
