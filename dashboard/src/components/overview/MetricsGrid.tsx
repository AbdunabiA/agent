import { Activity, MessageSquare, Heart, Wrench, Clock, Info } from "lucide-react";
import type { AgentStatus } from "@/lib/types";
import { formatDuration, formatRelativeTime } from "@/lib/utils";
import { MetricCard } from "@/components/overview/MetricCard";
import { Spinner } from "@/components/ui/Spinner";
import { ErrorMessage } from "@/components/ui/ErrorMessage";

interface MetricsGridProps extends AgentStatus {
  onRetry?: () => void;
}

export function MetricsGrid({ health, status, loading, error, onRetry }: MetricsGridProps) {
  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Spinner size="lg" />
      </div>
    );
  }

  if (error) {
    return <ErrorMessage message={error} onRetry={onRetry} />;
  }

  const statusValue = status?.status ?? health?.status ?? "unknown";

  const sessionsValue = status?.active_sessions ?? 0;

  let heartbeatValue: string;
  if (status?.heartbeat.enabled) {
    heartbeatValue = status.heartbeat.last_tick
      ? formatRelativeTime(status.heartbeat.last_tick)
      : "No ticks";
  } else {
    heartbeatValue = "Disabled";
  }

  const toolsValue = status
    ? `${status.tools.enabled}/${status.tools.total}`
    : "0/0";

  const uptimeValue = health?.uptime_seconds != null
    ? formatDuration(health.uptime_seconds)
    : "N/A";

  const versionValue = health?.version ?? "?";

  return (
    <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
      <MetricCard icon={Activity} label="Status" value={statusValue} />
      <MetricCard icon={MessageSquare} label="Sessions" value={sessionsValue} />
      <MetricCard icon={Heart} label="Heartbeat" value={heartbeatValue} />
      <MetricCard icon={Wrench} label="Tools" value={toolsValue} />
      <MetricCard icon={Clock} label="Uptime" value={uptimeValue} />
      <MetricCard icon={Info} label="Version" value={versionValue} />
    </div>
  );
}
