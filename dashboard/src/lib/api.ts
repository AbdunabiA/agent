import type {
  HealthResponse,
  StatusResponse,
  ChatResponse,
  SessionSummary,
  MessageOut,
  AuditEntry,
  AuditStats,
  ToolInfo,
  ControlAction,
  ControlResponse,
  FactsResponse,
  VectorSearchResponse,
  MemoryStats,
  CostStats,
  TimelineResponse,
  SoulResponse,
  SoulUpdateResponse,
  ScheduledTask,
  CreateTaskRequest,
  DisplayConfig,
  EditableConfig,
  ConfigUpdateResponse,
  SkillInfo,
  WorkspaceInfo,
  CreateWorkspaceRequest,
} from "./types";

const BASE_URL = import.meta.env.VITE_API_URL ?? "";
const WS_BASE = import.meta.env.VITE_WS_URL ?? "";
const AUTH_TOKEN = import.meta.env.VITE_AUTH_TOKEN ?? "";

function authHeaders(): Record<string, string> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (AUTH_TOKEN) headers["Authorization"] = `Bearer ${AUTH_TOKEN}`;
  return headers;
}

function wsUrl(path: string, params?: Record<string, string>): string {
  const base = WS_BASE || `ws://${window.location.host}`;
  const url = new URL(path, base);
  if (AUTH_TOKEN) url.searchParams.set("token", AUTH_TOKEN);
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      url.searchParams.set(k, v);
    }
  }
  return url.toString();
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${BASE_URL}/api/v1${path}`;
  const res = await fetch(url, { ...init, headers: { ...authHeaders(), ...init?.headers } });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => apiFetch<HealthResponse>("/health"),

  status: () => apiFetch<StatusResponse>("/status"),

  chat: (message: string, sessionId?: string) =>
    apiFetch<ChatResponse>("/chat", {
      method: "POST",
      body: JSON.stringify({ message, session_id: sessionId ?? null, channel: "webchat" }),
    }),

  conversations: (limit = 50) =>
    apiFetch<SessionSummary[]>(`/conversations?limit=${limit}`),

  messages: (sessionId: string, limit = 100) =>
    apiFetch<MessageOut[]>(`/conversations/${sessionId}/messages?limit=${limit}`),

  audit: (limit = 50) => apiFetch<AuditEntry[]>(`/audit?limit=${limit}`),

  auditStats: () => apiFetch<AuditStats>("/audit/stats"),

  tools: () => apiFetch<ToolInfo[]>("/tools"),

  control: (action: ControlAction) =>
    apiFetch<ControlResponse>("/control", {
      method: "POST",
      body: JSON.stringify({ action }),
    }),

  config: () => apiFetch<Record<string, unknown>>("/config"),

  // Memory
  memoryFacts: (limit = 50, offset = 0, q?: string) =>
    apiFetch<FactsResponse>(
      `/memory/facts?limit=${limit}&offset=${offset}${q ? `&q=${encodeURIComponent(q)}` : ""}`,
    ),
  memorySearch: (q: string, limit = 10) =>
    apiFetch<VectorSearchResponse>(
      `/memory/search?q=${encodeURIComponent(q)}&limit=${limit}`,
    ),
  memoryStats: () => apiFetch<MemoryStats>("/memory/stats"),
  deleteFact: (factId: string) =>
    apiFetch<{ success: boolean }>(`/memory/facts/${factId}`, { method: "DELETE" }),

  // Stats
  costStats: (period = "day") =>
    apiFetch<CostStats>(`/stats/costs?period=${period}`),
  timeline: (limit = 100, eventTypes?: string) =>
    apiFetch<TimelineResponse>(
      `/stats/timeline?limit=${limit}${eventTypes ? `&event_types=${encodeURIComponent(eventTypes)}` : ""}`,
    ),

  // Soul
  getSoul: () => apiFetch<SoulResponse>("/soul"),
  updateSoul: (content: string) =>
    apiFetch<SoulUpdateResponse>("/soul", {
      method: "PUT",
      body: JSON.stringify({ content }),
    }),

  // Tools toggle
  toggleTool: (name: string, enabled: boolean) =>
    apiFetch<{ success: boolean }>(`/tools/${encodeURIComponent(name)}/toggle`, {
      method: "PUT",
      body: JSON.stringify({ enabled }),
    }),

  // Tasks
  listTasks: () => apiFetch<ScheduledTask[]>("/tasks"),
  createTask: (task: CreateTaskRequest) =>
    apiFetch<ScheduledTask>("/tasks", {
      method: "POST",
      body: JSON.stringify(task),
    }),
  deleteTask: (taskId: string) =>
    apiFetch<{ success: boolean }>(`/tasks/${encodeURIComponent(taskId)}`, {
      method: "DELETE",
    }),

  // Skills
  skills: () => apiFetch<SkillInfo[]>("/skills"),
  reloadSkill: (name: string) =>
    apiFetch<{ success: boolean }>(`/skills/${encodeURIComponent(name)}/reload`, {
      method: "POST",
    }),
  enableSkill: (name: string) =>
    apiFetch<{ success: boolean }>(`/skills/${encodeURIComponent(name)}/enable`, {
      method: "POST",
    }),
  disableSkill: (name: string) =>
    apiFetch<{ success: boolean }>(`/skills/${encodeURIComponent(name)}/disable`, {
      method: "POST",
    }),

  // Config (display + editing)
  displayConfig: () => apiFetch<DisplayConfig>("/config"),
  editableConfig: () => apiFetch<EditableConfig>("/config/editable"),
  updateConfig: (section: string, data: Record<string, unknown>) =>
    apiFetch<ConfigUpdateResponse>(`/config/${encodeURIComponent(section)}`, {
      method: "PUT",
      body: JSON.stringify({ data }),
    }),

  // Workspaces
  workspaces: () => apiFetch<WorkspaceInfo[]>("/workspaces"),
  activeWorkspace: () => apiFetch<WorkspaceInfo>("/workspaces/active"),
  getWorkspace: (name: string) =>
    apiFetch<WorkspaceInfo>(`/workspaces/${encodeURIComponent(name)}`),
  createWorkspace: (body: CreateWorkspaceRequest) =>
    apiFetch<WorkspaceInfo>("/workspaces", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  deleteWorkspace: (name: string) =>
    apiFetch<{ success: boolean }>(`/workspaces/${encodeURIComponent(name)}?confirm=true`, {
      method: "DELETE",
    }),
  switchWorkspace: (name: string) =>
    apiFetch<WorkspaceInfo>(`/workspaces/${encodeURIComponent(name)}/switch`, {
      method: "POST",
    }),
};

export function createEventSocket(): WebSocket {
  return new WebSocket(wsUrl("/ws/events"));
}

export function createChatSocket(sessionId?: string): WebSocket {
  const params: Record<string, string> = {};
  if (sessionId) params["session_id"] = sessionId;
  return new WebSocket(wsUrl("/ws/chat", params));
}
