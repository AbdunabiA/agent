import { Outlet, useLocation } from "react-router-dom";
import { useAgentStatus } from "@/hooks/useAgentStatus";
import { Sidebar } from "@/components/layout/Sidebar";
import { Header } from "@/components/layout/Header";

const pageTitles: Record<string, string> = {
  "/": "Overview",
  "/chat": "Chat",
};

function derivePageTitle(pathname: string): string {
  if (pageTitles[pathname]) {
    return pageTitles[pathname];
  }

  // Fallback: capitalize the first segment of the path
  const segment = pathname.split("/").filter(Boolean)[0];
  if (segment) {
    return segment.charAt(0).toUpperCase() + segment.slice(1);
  }

  return "Dashboard";
}

function deriveStatus(
  statusStr: string | undefined | null,
  loading: boolean,
  error: string | null,
): string {
  if (loading) return "loading";
  if (error) return "error";
  return statusStr ?? "unknown";
}

export function Layout() {
  const location = useLocation();
  const agentStatus = useAgentStatus();

  const connected = !agentStatus.loading && agentStatus.error === null;
  const statusString = deriveStatus(
    agentStatus.status?.status,
    agentStatus.loading,
    agentStatus.error,
  );
  const pageTitle = derivePageTitle(location.pathname);

  return (
    <div className="flex h-screen bg-gray-950 text-gray-100">
      <Sidebar connected={connected} />
      <div className="flex flex-1 flex-col overflow-hidden">
        <Header title={pageTitle} agentStatus={statusString} />
        <main className="flex-1 overflow-auto p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
