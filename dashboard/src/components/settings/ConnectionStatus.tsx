import { Wifi, WifiOff, Activity, Database, Radio } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/StatusBadge";
import type { AgentStatus } from "@/lib/types";

interface ConnectionStatusProps {
  status: AgentStatus;
}

export function ConnectionStatus({ status }: ConnectionStatusProps) {
  const items: { label: string; icon: React.ElementType; ok: boolean }[] = [
    {
      label: "API",
      icon: Wifi,
      ok: status.health !== null && !status.error,
    },
    {
      label: "Heartbeat",
      icon: Activity,
      ok: status.status?.heartbeat?.enabled ?? false,
    },
    {
      label: "Database",
      icon: Database,
      ok: status.health !== null,
    },
    {
      label: "WebSocket",
      icon: Radio,
      ok: status.health !== null && !status.error,
    },
  ];

  return (
    <Card>
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          {status.error ? (
            <WifiOff className="h-5 w-5 text-red-400" />
          ) : (
            <Wifi className="h-5 w-5 text-green-400" />
          )}
          <h2 className="text-lg font-semibold text-gray-100">Connection Status</h2>
        </div>

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {items.map(({ label, icon: Icon, ok }) => (
            <div
              key={label}
              className="flex items-center gap-2 rounded-lg border border-gray-800 bg-gray-950 px-3 py-2"
            >
              <Icon className={`h-4 w-4 ${ok ? "text-green-400" : "text-gray-600"}`} />
              <span className="text-sm text-gray-300">{label}</span>
              <StatusBadge variant={ok ? "green" : "red"} label="" className="ml-auto" />
            </div>
          ))}
        </div>

        {status.health && (
          <p className="text-xs text-gray-500">
            Uptime: {Math.floor(status.health.uptime_seconds / 60)}m
            {" | "}Version: {status.health.version}
          </p>
        )}
      </div>
    </Card>
  );
}
