import { useState } from "react";
import { NavLink } from "react-router-dom";
import {
  Bot,
  LayoutDashboard,
  MessageSquare,
  Brain,
  Clock,
  Settings,
  PanelLeftClose,
  PanelLeftOpen,
  MessagesSquare,
  DollarSign,
  ClipboardList,
  Wrench,
  Puzzle,
  CalendarCheck,
  Ghost,
  Layers,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { StatusBadge } from "@/components/ui/StatusBadge";

interface SidebarProps {
  connected: boolean;
}

interface NavItem {
  label: string;
  icon: React.ElementType;
  to: string;
  disabled?: boolean;
  tooltip?: string;
}

const navItems: NavItem[] = [
  { label: "Overview", icon: LayoutDashboard, to: "/" },
  { label: "Chat", icon: MessageSquare, to: "/chat" },
  { label: "Conversations", icon: MessagesSquare, to: "/conversations" },
  { label: "Memory", icon: Brain, to: "/memory" },
  { label: "Timeline", icon: Clock, to: "/timeline" },
  { label: "Costs", icon: DollarSign, to: "/costs" },
  { label: "Audit Log", icon: ClipboardList, to: "/audit" },
];

const managementItems: NavItem[] = [
  { label: "Tools", icon: Wrench, to: "/tools" },
  { label: "Skills", icon: Puzzle, to: "/skills" },
  { label: "Tasks", icon: CalendarCheck, to: "/tasks" },
  { label: "Workspaces", icon: Layers, to: "/workspaces" },
  { label: "Soul Editor", icon: Ghost, to: "/soul" },
  { label: "Settings", icon: Settings, to: "/settings" },
];

export function Sidebar({ connected }: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={cn(
        "flex h-full flex-col border-r border-gray-800 bg-gray-950 transition-all duration-200",
        collapsed ? "w-16" : "w-56",
      )}
    >
      {/* Logo / Title */}
      <div className="flex h-14 items-center gap-2 border-b border-gray-800 px-4">
        <Bot className="h-6 w-6 shrink-0 text-indigo-400" />
        {!collapsed && (
          <span className="text-lg font-semibold text-gray-100">Agent</span>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 overflow-y-auto px-2 py-4">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.label}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                cn(
                  "group flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-indigo-600/20 text-indigo-400"
                    : "text-gray-400 hover:bg-gray-800 hover:text-gray-200",
                  collapsed && "justify-center",
                )
              }
            >
              <Icon className="h-5 w-5 shrink-0" />
              {!collapsed && <span>{item.label}</span>}
            </NavLink>
          );
        })}

        {/* Separator */}
        <div className="my-3 border-t border-gray-800" />

        {managementItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.label}
              to={item.to}
              className={({ isActive }) =>
                cn(
                  "group flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-indigo-600/20 text-indigo-400"
                    : "text-gray-400 hover:bg-gray-800 hover:text-gray-200",
                  collapsed && "justify-center",
                )
              }
            >
              <Icon className="h-5 w-5 shrink-0" />
              {!collapsed && <span>{item.label}</span>}
            </NavLink>
          );
        })}
      </nav>

      {/* Bottom section: status + collapse toggle */}
      <div className="space-y-2 border-t border-gray-800 px-3 py-3">
        {/* Connection status */}
        <div
          className={cn(
            "flex items-center",
            collapsed ? "justify-center" : "px-1",
          )}
        >
          <StatusBadge
            variant={connected ? "green" : "red"}
            label={collapsed ? "" : connected ? "Connected" : "Disconnected"}
          />
        </div>

        {/* Collapse toggle */}
        <button
          onClick={() => setCollapsed((prev) => !prev)}
          className={cn(
            "flex w-full items-center rounded-md px-3 py-2 text-sm text-gray-400 transition-colors hover:bg-gray-800 hover:text-gray-200",
            collapsed && "justify-center",
          )}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? (
            <PanelLeftOpen className="h-5 w-5 shrink-0" />
          ) : (
            <>
              <PanelLeftClose className="h-5 w-5 shrink-0" />
              <span className="ml-3">Collapse</span>
            </>
          )}
        </button>
      </div>
    </aside>
  );
}
