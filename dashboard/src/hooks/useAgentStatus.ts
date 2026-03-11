import { useState, useEffect, useRef, useCallback } from "react";
import { api } from "@/lib/api";
import type { AgentStatus } from "@/lib/types";

const POLL_INTERVAL = 10_000;

export function useAgentStatus(): AgentStatus & { refetch: () => void } {
  const [state, setState] = useState<AgentStatus>({
    health: null,
    status: null,
    loading: true,
    error: null,
  });
  const mountedRef = useRef(true);

  const fetchStatus = useCallback(async () => {
    try {
      const [health, status] = await Promise.all([api.health(), api.status()]);
      if (mountedRef.current) {
        setState({ health, status, loading: false, error: null });
      }
    } catch (err: unknown) {
      if (mountedRef.current) {
        setState((prev) => ({
          ...prev,
          loading: false,
          error: err instanceof Error ? err.message : String(err),
        }));
      }
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    fetchStatus();
    const interval = setInterval(fetchStatus, POLL_INTERVAL);
    return () => {
      mountedRef.current = false;
      clearInterval(interval);
    };
  }, [fetchStatus]);

  return { ...state, refetch: fetchStatus };
}
