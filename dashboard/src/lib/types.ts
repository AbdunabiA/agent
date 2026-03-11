// ── REST API responses ──

export interface HealthResponse {
  status: string;
  version: string;
  uptime_seconds: number;
  timestamp: string;
}

export interface StatusResponse {
  status: string;
  active_sessions: number;
  heartbeat: {
    enabled: boolean;
    last_tick: string | null;
  };
  tools: {
    total: number;
    enabled: number;
  };
}

export interface ChatRequest {
  message: string;
  session_id?: string | null;
  channel?: string;
}

export interface TokenUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
}

export interface ChatResponse {
  response: string;
  session_id: string;
  model: string;
  usage: TokenUsage;
}

export interface ToolCallInfo {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

export interface MessageOut {
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  model: string | null;
  timestamp: string;
  tool_calls: ToolCallInfo[] | null;
}

export interface SessionSummary {
  id: string;
  channel: string;
  message_count: number;
  total_tokens: number;
  created_at: string;
  updated_at: string;
}

export interface AuditEntry {
  id: string;
  timestamp: string;
  tool_name: string;
  status: "success" | "error" | "timeout" | "denied" | "blocked";
  duration_ms: number;
  trigger: string;
  error: string | null;
}

export interface AuditStats {
  total_calls: number;
  success_count: number;
  error_count: number;
  success_rate: number;
  avg_duration_ms: number;
  tools_used: Record<string, number>;
}

export interface ToolInfo {
  name: string;
  description: string;
  tier: "safe" | "moderate" | "dangerous";
  enabled: boolean;
  parameters: Record<string, unknown>;
}

export interface ControlResponse {
  status: string;
  message: string;
}

// ── WebSocket: Event Stream ──

export interface AgentEvent {
  event: string;
  timestamp: string;
  data: Record<string, unknown>;
}

// ── WebSocket: Chat Protocol ──

export type WsChatMessage =
  | { type: "response.start"; session_id: string }
  | { type: "response.chunk"; content: string }
  | { type: "typing"; status: boolean }
  | { type: "tool.execute"; tool: string; arguments: Record<string, unknown> }
  | { type: "tool.result"; tool: string; success: boolean; output: string }
  | { type: "response.end"; content: string; model: string; usage: TokenUsage }
  | { type: "voice.transcription"; text: string }
  | { type: "voice.audio"; audio: string; mime_type: string; duration?: number }
  | { type: "pong" }
  | { type: "error"; message: string };

// ── UI state types ──

export interface AgentStatus {
  health: HealthResponse | null;
  status: StatusResponse | null;
  loading: boolean;
  error: string | null;
}

export type ControlAction = "pause" | "resume" | "mute" | "unmute";

// ── Memory types ──

export interface Fact {
  id: string;
  key: string;
  value: string;
  category: string;
  confidence: number;
  source: string;
  created_at: string;
  updated_at: string;
}

export interface FactsResponse {
  facts: Fact[];
  total: number;
}

export interface VectorResult {
  text: string;
  similarity: number;
  metadata: Record<string, unknown>;
}

export interface VectorSearchResponse {
  results: VectorResult[];
}

export interface MemoryStats {
  facts_count: number;
  vectors_count: number;
  soul_loaded: boolean;
}

// ── Cost types ──

export interface CostStats {
  total_cost: number;
  total_tokens: { input: number; output: number };
  total_calls: number;
  period: string;
  by_time: { time: string; cost: number; tokens: number }[];
  by_model: { model: string; cost: number; percentage: number }[];
  by_channel: { channel: string; cost: number }[];
}

// ── Timeline types ──

export interface TimelineEvent {
  id: string;
  timestamp: string;
  event: string;
  description: string;
  details: Record<string, unknown>;
  icon?: string;
}

export interface TimelineResponse {
  events: TimelineEvent[];
}

// ── Soul types ──

export interface SoulResponse {
  content: string;
  loaded: boolean;
  path: string;
  last_modified: string;
}

export interface SoulUpdateResponse {
  success: boolean;
  content: string;
}

// ── Task types ──

export interface ScheduledTask {
  id: string;
  description: string;
  type: "reminder" | "cron";
  schedule: string;
  status: string;
  channel: string | null;
  created_at: string;
  next_run: string | null;
  last_run: string | null;
}

export interface CreateTaskRequest {
  type: "reminder" | "cron";
  description: string;
  schedule: string;
  channel?: string;
}

// ── Skill types ──

export interface SkillInfo {
  name: string;
  display_name: string;
  loaded: boolean;
  version: string;
  author: string;
  description: string;
  permissions: string[];
  tools: string[];
  path: string;
}

// ── Workspace types ──

export interface WorkspaceInfo {
  name: string;
  display_name: string;
  description: string;
  is_active: boolean;
  model?: string | null;
  root_dir?: string;
  data_dir?: string;
  soul_path?: string;
  error?: string;
}

export interface CreateWorkspaceRequest {
  name: string;
  display_name?: string;
  description?: string;
  clone_from?: string | null;
}

// ── Config types ──

export type DisplayConfig = Record<string, unknown>;

export interface ConfigFieldMeta {
  value: unknown;
  type: "string" | "number" | "boolean" | "array" | "object";
  editable: boolean;
  options?: string[];
  /** Sub-fields for nested objects (e.g. claude_sdk settings) */
  fields?: Record<string, ConfigFieldMeta>;
}

export type EditableConfigSection = Record<string, ConfigFieldMeta>;
export type EditableConfig = Record<string, EditableConfigSection>;

export interface ConfigUpdateResponse {
  success: boolean;
  section: string;
  config: DisplayConfig;
}
