import { useState, useEffect, useRef, useCallback } from "react";
import { createChatSocket } from "@/lib/api";
import type { WsChatMessage, MessageOut, ToolCallInfo } from "@/lib/types";

interface ToolEvent {
  type: "execute" | "result";
  tool: string;
  arguments?: Record<string, unknown>;
  success?: boolean;
  output?: string;
}

interface UseChatSocketResult {
  messages: MessageOut[];
  toolEvents: ToolEvent[];
  typing: boolean;
  connected: boolean;
  sessionId: string | null;
  error: string | null;
  sendMessage: (content: string) => void;
  sendVoice: (audio: string, mimeType: string) => void;
  setMessages: React.Dispatch<React.SetStateAction<MessageOut[]>>;
}

const RECONNECT_DELAY = 3000;

export function useChatSocket(initialSessionId?: string): UseChatSocketResult {
  const [messages, setMessages] = useState<MessageOut[]>([]);
  const [toolEvents, setToolEvents] = useState<ToolEvent[]>([]);
  const [typing, setTyping] = useState(false);
  const [connected, setConnected] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(initialSessionId ?? null);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();
  const pendingTools = useRef<ToolCallInfo[]>([]);
  const streamingRef = useRef<string | null>(null);
  // Track session_id in a ref so reconnections use the latest value
  // without triggering a new connection via useEffect dependency changes.
  const sessionIdRef = useRef<string | null>(initialSessionId ?? null);

  const connect = useCallback(() => {
    const ws = createChatSocket(sessionIdRef.current ?? undefined);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);

    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data as string) as WsChatMessage;

      switch (msg.type) {
        case "response.start":
          sessionIdRef.current = msg.session_id;
          setSessionId(msg.session_id);
          setError(null);
          pendingTools.current = [];
          streamingRef.current = null;
          setToolEvents([]);
          break;

        case "response.chunk":
          if (streamingRef.current === null) {
            // First chunk — create a new streaming message
            streamingRef.current = msg.content;
            setMessages((prev) => [
              ...prev,
              {
                role: "assistant",
                content: msg.content,
                model: null,
                timestamp: new Date().toISOString(),
                tool_calls: null,
              },
            ]);
          } else {
            // Subsequent chunks — update the last message
            streamingRef.current += msg.content;
            const updated = streamingRef.current;
            setMessages((prev) => {
              const last = prev[prev.length - 1]!;
              return [
                ...prev.slice(0, -1),
                { role: last.role, content: updated, model: last.model, timestamp: last.timestamp, tool_calls: last.tool_calls },
              ];
            });
          }
          break;

        case "typing":
          setTyping(msg.status);
          break;

        case "tool.execute":
          // If we were streaming text, finalize it before tool events
          if (streamingRef.current !== null) {
            streamingRef.current = null;
          }
          pendingTools.current.push({
            id: `tc_${Date.now()}`,
            name: msg.tool,
            arguments: msg.arguments,
          });
          setToolEvents((prev) => [
            ...prev,
            { type: "execute", tool: msg.tool, arguments: msg.arguments },
          ]);
          break;

        case "tool.result":
          setToolEvents((prev) => [
            ...prev,
            { type: "result", tool: msg.tool, success: msg.success, output: msg.output },
          ]);
          break;

        case "response.end":
          setTyping(false);
          if (streamingRef.current !== null) {
            // Finalize the streaming message with model info
            setMessages((prev) => [
              ...prev.slice(0, -1),
              {
                role: "assistant",
                content: msg.content,
                model: msg.model,
                timestamp: new Date().toISOString(),
                tool_calls:
                  pendingTools.current.length > 0
                    ? [...pendingTools.current]
                    : null,
              },
            ]);
          } else {
            // No streaming occurred (e.g., voice response)
            setMessages((prev) => [
              ...prev,
              {
                role: "assistant",
                content: msg.content,
                model: msg.model,
                timestamp: new Date().toISOString(),
                tool_calls:
                  pendingTools.current.length > 0
                    ? [...pendingTools.current]
                    : null,
              },
            ]);
          }
          streamingRef.current = null;
          pendingTools.current = [];
          break;

        case "voice.transcription":
          // Show transcription as a user message
          setMessages((prev) => [
            ...prev,
            {
              role: "user",
              content: `🎤 ${msg.text}`,
              model: null,
              timestamp: new Date().toISOString(),
              tool_calls: null,
            },
          ]);
          break;

        case "voice.audio": {
          // Auto-play received audio
          const audioB64 = msg.audio;
          const audioMime = msg.mime_type || "audio/ogg";
          try {
            const audioBlob = new Blob(
              [Uint8Array.from(atob(audioB64), (c) => c.charCodeAt(0))],
              { type: audioMime },
            );
            const audioUrl = URL.createObjectURL(audioBlob);
            const audio = new Audio(audioUrl);
            audio.play().catch(() => {});
            audio.onended = () => URL.revokeObjectURL(audioUrl);
          } catch {
            // Silently ignore audio playback errors
          }
          break;
        }

        case "error":
          setTyping(false);
          setError(msg.message);
          break;

        case "pong":
          break;
      }
    };

    ws.onclose = () => {
      setConnected(false);
      setTyping(false);
      reconnectTimer.current = setTimeout(() => connect(), RECONNECT_DELAY);
    };

    ws.onerror = () => ws.close();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const sendMessage = useCallback(
    (content: string) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
      setMessages((prev) => [
        ...prev,
        {
          role: "user",
          content,
          model: null,
          timestamp: new Date().toISOString(),
          tool_calls: null,
        },
      ]);
      wsRef.current.send(JSON.stringify({ type: "message", content }));
    },
    [],
  );

  const sendVoice = useCallback(
    (audio: string, mimeType: string) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
      wsRef.current.send(
        JSON.stringify({ type: "voice.data", audio, mime_type: mimeType }),
      );
    },
    [],
  );

  return {
    messages, toolEvents, typing, connected, sessionId, error,
    sendMessage, sendVoice, setMessages,
  };
}
