import { StatusBadge } from "@/components/ui/StatusBadge";
import { WorkspaceSwitcher } from "@/components/layout/WorkspaceSwitcher";
import { getToken, clearToken } from "@/lib/api";

interface HeaderProps {
  title: string;
  agentStatus: string;
}

function statusVariant(status: string): "green" | "yellow" | "red" | "gray" {
  switch (status) {
    case "running":
      return "green";
    case "paused":
      return "yellow";
    case "error":
      return "red";
    default:
      return "gray";
  }
}

export function Header({ title, agentStatus }: HeaderProps) {
  return (
    <header className="h-14 border-b border-gray-800 bg-gray-950 px-6 flex items-center justify-between">
      <h1 className="text-lg font-semibold text-gray-100">{title}</h1>
      <div className="flex items-center gap-4">
        <WorkspaceSwitcher />
        <StatusBadge
          variant={statusVariant(agentStatus)}
          label={agentStatus.charAt(0).toUpperCase() + agentStatus.slice(1)}
        />
        {getToken() && (
          <button
            onClick={() => { clearToken(); window.location.reload(); }}
            className="text-xs text-gray-500 hover:text-gray-300 transition-colors"
            title="Log out"
          >
            Logout
          </button>
        )}
      </div>
    </header>
  );
}
