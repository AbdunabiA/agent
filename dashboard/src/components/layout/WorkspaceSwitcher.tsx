import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { ChevronDown, Plus, Settings2 } from "lucide-react";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { WorkspaceInfo } from "@/lib/types";

export function WorkspaceSwitcher() {
  const [workspaces, setWorkspaces] = useState<WorkspaceInfo[]>([]);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  const active = workspaces.find((w) => w.is_active);

  useEffect(() => {
    api.workspaces().then(setWorkspaces).catch(() => {});
  }, []);

  // Close dropdown on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  async function handleSwitch(name: string) {
    try {
      await api.switchWorkspace(name);
      const updated = await api.workspaces();
      setWorkspaces(updated);
    } catch {
      // ignore
    }
    setOpen(false);
  }

  if (workspaces.length === 0) return null;

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((prev) => !prev)}
        className="flex items-center gap-2 rounded-md border border-gray-700 bg-gray-900 px-3 py-1.5 text-sm text-gray-200 transition-colors hover:bg-gray-800"
      >
        <span className="max-w-[120px] truncate">
          {active?.display_name ?? "Workspace"}
        </span>
        <ChevronDown className={cn("h-4 w-4 transition-transform", open && "rotate-180")} />
      </button>

      {open && (
        <div className="absolute right-0 top-full z-50 mt-1 w-52 rounded-md border border-gray-700 bg-gray-900 py-1 shadow-lg">
          {workspaces.map((ws) => (
            <button
              key={ws.name}
              onClick={() => handleSwitch(ws.name)}
              className={cn(
                "flex w-full items-center justify-between px-3 py-2 text-left text-sm transition-colors hover:bg-gray-800",
                ws.is_active ? "text-indigo-400" : "text-gray-300",
              )}
            >
              <span className="truncate">{ws.display_name || ws.name}</span>
              {ws.is_active && (
                <span className="ml-2 text-xs text-indigo-400">active</span>
              )}
            </button>
          ))}

          <div className="my-1 border-t border-gray-700" />

          <button
            onClick={() => {
              setOpen(false);
              navigate("/workspaces");
            }}
            className="flex w-full items-center gap-2 px-3 py-2 text-sm text-gray-400 transition-colors hover:bg-gray-800 hover:text-gray-200"
          >
            <Plus className="h-4 w-4" />
            <span>New workspace</span>
          </button>

          <button
            onClick={() => {
              setOpen(false);
              navigate("/workspaces");
            }}
            className="flex w-full items-center gap-2 px-3 py-2 text-sm text-gray-400 transition-colors hover:bg-gray-800 hover:text-gray-200"
          >
            <Settings2 className="h-4 w-4" />
            <span>Manage...</span>
          </button>
        </div>
      )}
    </div>
  );
}
