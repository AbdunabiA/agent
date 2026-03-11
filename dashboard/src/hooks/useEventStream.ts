import { useState, useEffect, useRef, useCallback } from "react";
import { createEventSocket } from "@/lib/api";
import type { AgentEvent } from "@/lib/types";

const MAX_EVENTS = 200;
const RECONNECT_DELAY = 3000;

interface UseEventStreamResult {
  events: AgentEvent[];
  connected: boolean;
  clear: () => void;
}

export function useEventStream(): UseEventStreamResult {
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();

  const clear = useCallback(() => setEvents([]), []);

  useEffect(() => {
    let mounted = true;

    function connect() {
      if (!mounted) return;
      const ws = createEventSocket();
      wsRef.current = ws;

      ws.onopen = () => {
        if (mounted) setConnected(true);
      };

      ws.onmessage = (ev) => {
        if (!mounted) return;
        try {
          const parsed = JSON.parse(ev.data as string) as AgentEvent;
          setEvents((prev) => {
            const next = [...prev, parsed];
            return next.length > MAX_EVENTS ? next.slice(-MAX_EVENTS) : next;
          });
        } catch {
          // ignore non-JSON messages like "pong"
        }
      };

      ws.onclose = () => {
        if (!mounted) return;
        setConnected(false);
        reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY);
      };

      ws.onerror = () => {
        ws.close();
      };
    }

    connect();

    return () => {
      mounted = false;
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, []);

  return { events, connected, clear };
}
