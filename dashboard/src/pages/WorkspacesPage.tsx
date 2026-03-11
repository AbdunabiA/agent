import { useState } from "react";
import { api } from "@/lib/api";
import { useApi } from "@/hooks/useApi";
import { Card } from "@/components/ui/Card";
import { Spinner } from "@/components/ui/Spinner";
import { ErrorMessage } from "@/components/ui/ErrorMessage";
import type { WorkspaceInfo } from "@/lib/types";

export function WorkspacesPage() {
  const { data: workspaces, loading, error, refetch } = useApi(() => api.workspaces());

  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDisplayName, setNewDisplayName] = useState("");
  const [newDescription, setNewDescription] = useState("");
  const [actionError, setActionError] = useState<string | null>(null);

  async function handleCreate() {
    if (!newName.trim()) return;
    setActionError(null);
    try {
      await api.createWorkspace({
        name: newName.trim(),
        display_name: newDisplayName.trim() || undefined,
        description: newDescription.trim() || undefined,
      });
      setNewName("");
      setNewDisplayName("");
      setNewDescription("");
      setCreating(false);
      refetch();
    } catch (err: unknown) {
      setActionError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleSwitch(name: string) {
    setActionError(null);
    try {
      await api.switchWorkspace(name);
      refetch();
    } catch (err: unknown) {
      setActionError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleDelete(name: string) {
    if (!window.confirm(`Delete workspace "${name}"? This cannot be undone.`)) return;
    setActionError(null);
    try {
      await api.deleteWorkspace(name);
      refetch();
    } catch (err: unknown) {
      setActionError(err instanceof Error ? err.message : String(err));
    }
  }

  if (loading) return <Spinner />;
  if (error) return <ErrorMessage message={error} />;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-gray-100">Workspaces</h2>
        <button
          onClick={() => setCreating(true)}
          className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-500"
        >
          + New Workspace
        </button>
      </div>

      {actionError && <ErrorMessage message={actionError} />}

      {/* Create form */}
      {creating && (
        <Card>
          <div className="space-y-3 p-4">
            <h3 className="text-sm font-semibold text-gray-200">Create Workspace</h3>
            <input
              type="text"
              placeholder="Name (e.g. work)"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              className="w-full rounded border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-gray-200 focus:border-indigo-500 focus:outline-none"
            />
            <input
              type="text"
              placeholder="Display name (optional)"
              value={newDisplayName}
              onChange={(e) => setNewDisplayName(e.target.value)}
              className="w-full rounded border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-gray-200 focus:border-indigo-500 focus:outline-none"
            />
            <input
              type="text"
              placeholder="Description (optional)"
              value={newDescription}
              onChange={(e) => setNewDescription(e.target.value)}
              className="w-full rounded border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-gray-200 focus:border-indigo-500 focus:outline-none"
            />
            <div className="flex gap-2">
              <button
                onClick={handleCreate}
                className="rounded bg-indigo-600 px-4 py-1.5 text-sm text-white hover:bg-indigo-500"
              >
                Create
              </button>
              <button
                onClick={() => setCreating(false)}
                className="rounded border border-gray-700 px-4 py-1.5 text-sm text-gray-400 hover:bg-gray-800"
              >
                Cancel
              </button>
            </div>
          </div>
        </Card>
      )}

      {/* Workspace list */}
      <div className="space-y-3">
        {(workspaces ?? []).map((ws: WorkspaceInfo) => (
          <Card key={ws.name}>
            <div className="flex items-start justify-between p-4">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <h3 className="text-base font-semibold text-gray-100">
                    {ws.display_name || ws.name}
                  </h3>
                  {ws.is_active && (
                    <span className="rounded-full bg-indigo-600/20 px-2 py-0.5 text-xs font-medium text-indigo-400">
                      active
                    </span>
                  )}
                </div>
                {ws.description && (
                  <p className="mt-1 text-sm text-gray-400">{ws.description}</p>
                )}
                <div className="mt-2 flex gap-4 text-xs text-gray-500">
                  {ws.model && <span>Model: {ws.model}</span>}
                  <span>Name: {ws.name}</span>
                </div>
              </div>

              <div className="flex shrink-0 gap-2">
                {!ws.is_active && (
                  <button
                    onClick={() => handleSwitch(ws.name)}
                    className="rounded border border-gray-700 px-3 py-1.5 text-xs text-gray-300 transition-colors hover:bg-gray-800"
                  >
                    Switch To
                  </button>
                )}
                {ws.name !== "default" && (
                  <button
                    onClick={() => handleDelete(ws.name)}
                    className="rounded border border-red-800/50 px-3 py-1.5 text-xs text-red-400 transition-colors hover:bg-red-900/20"
                  >
                    Delete
                  </button>
                )}
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* Routing info */}
      <Card>
        <div className="p-4">
          <h3 className="text-sm font-semibold text-gray-200">Routing Rules</h3>
          <p className="mt-2 text-xs text-gray-500">
            Configure channel-to-workspace routing in <code className="text-gray-400">agent.yaml</code> under{" "}
            <code className="text-gray-400">workspaces.routing.rules</code>.
          </p>
        </div>
      </Card>
    </div>
  );
}
