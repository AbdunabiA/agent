import { useState } from "react";
import { Pause, Play, VolumeX, Volume2, OctagonX } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { api } from "@/lib/api";
import type { ControlAction } from "@/lib/types";

export function EmergencyPanel() {
  const [loading, setLoading] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const handleAction = async (action: ControlAction) => {
    setLoading(action);
    setMessage(null);
    try {
      const res = await api.control(action);
      setMessage(res.message);
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Action failed");
    } finally {
      setLoading(null);
    }
  };

  const buttons: { action: ControlAction; icon: React.ElementType; label: string; color: string }[] = [
    { action: "pause", icon: Pause, label: "Pause", color: "bg-yellow-600 hover:bg-yellow-500" },
    { action: "resume", icon: Play, label: "Resume", color: "bg-green-600 hover:bg-green-500" },
    { action: "mute", icon: VolumeX, label: "Mute", color: "bg-orange-600 hover:bg-orange-500" },
    { action: "unmute", icon: Volume2, label: "Unmute", color: "bg-blue-600 hover:bg-blue-500" },
  ];

  return (
    <Card>
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <OctagonX className="h-5 w-5 text-red-400" />
          <h2 className="text-lg font-semibold text-gray-100">Emergency Controls</h2>
        </div>

        <div className="flex flex-wrap gap-3">
          {buttons.map(({ action, icon: Icon, label, color }) => (
            <button
              key={action}
              onClick={() => handleAction(action)}
              disabled={loading !== null}
              className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium text-white transition-colors disabled:opacity-50 ${color}`}
            >
              <Icon className="h-4 w-4" />
              {loading === action ? "..." : label}
            </button>
          ))}
        </div>

        {message && (
          <p className="text-sm text-gray-400">{message}</p>
        )}
      </div>
    </Card>
  );
}
