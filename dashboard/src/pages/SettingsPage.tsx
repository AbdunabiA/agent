import { useAgentStatus } from "@/hooks/useAgentStatus";
import { EmergencyPanel } from "@/components/settings/EmergencyPanel";
import { ConnectionStatus } from "@/components/settings/ConnectionStatus";
import { ConfigEditor } from "@/components/settings/ConfigEditor";

export function SettingsPage() {
  const status = useAgentStatus();

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-100">Settings</h1>
      <EmergencyPanel />
      <ConnectionStatus status={status} />
      <ConfigEditor />
    </div>
  );
}
