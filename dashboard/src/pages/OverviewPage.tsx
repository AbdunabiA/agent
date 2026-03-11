import { useCallback, useState } from "react";
import { useAgentStatus } from "@/hooks/useAgentStatus";
import { useEventStream } from "@/hooks/useEventStream";
import { MetricsGrid } from "@/components/overview/MetricsGrid";
import { ActivityFeed } from "@/components/overview/ActivityFeed";
import { QuickActions } from "@/components/overview/QuickActions";
import { api } from "@/lib/api";
import type { ControlAction } from "@/lib/types";

export function OverviewPage() {
  const agentStatus = useAgentStatus();
  const { events, connected, clear } = useEventStream();
  const [actionLoading, setActionLoading] = useState(false);

  const handleAction = useCallback(async (action: ControlAction) => {
    setActionLoading(true);
    try {
      await api.control(action);
      agentStatus.refetch();
    } catch {
      // error is visible via status refetch
    } finally {
      setActionLoading(false);
    }
  }, [agentStatus]);

  return (
    <div className="space-y-6">
      <MetricsGrid
        health={agentStatus.health}
        status={agentStatus.status}
        loading={agentStatus.loading}
        error={agentStatus.error}
        onRetry={agentStatus.refetch}
      />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <ActivityFeed events={events} connected={connected} onClear={clear} />
        </div>
        <div>
          <QuickActions onAction={handleAction} loading={actionLoading} />
        </div>
      </div>
    </div>
  );
}
