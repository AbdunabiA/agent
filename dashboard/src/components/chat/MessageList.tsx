import { useEffect, useRef } from "react";
import { MessageSquare } from "lucide-react";
import type { MessageOut } from "@/lib/types";
import { ChatMessage } from "./ChatMessage";
import { ToolCallCard } from "./ToolCallCard";
import { EmptyState } from "@/components/ui/EmptyState";

export interface ToolEvent {
  type: "execute" | "result";
  tool: string;
  arguments?: Record<string, unknown>;
  success?: boolean;
  output?: string;
}

interface MessageListProps {
  messages: MessageOut[];
  toolEvents: ToolEvent[];
  typing: boolean;
}

export function MessageList({ messages, toolEvents, typing }: MessageListProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive or typing changes
  useEffect(() => {
    const el = containerRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages, toolEvents, typing]);

  if (messages.length === 0 && !typing) {
    return (
      <div ref={containerRef} className="flex flex-1 items-center justify-center">
        <EmptyState
          icon={MessageSquare}
          title="Start a conversation"
          description="Send a message to begin chatting with the agent."
        />
      </div>
    );
  }

  return (
    <div ref={containerRef} className="flex-1 overflow-y-auto py-4 space-y-2">
      {messages.map((msg, idx) => (
        <ChatMessage key={`${msg.timestamp}-${idx}`} message={msg} />
      ))}

      {/* Tool event cards shown after the last assistant message */}
      {toolEvents.length > 0 && (
        <div className="px-4 space-y-2 max-w-[75%]">
          {toolEvents.map((event, idx) => (
            <ToolCallCard
              key={`tool-${idx}`}
              name={event.tool}
              arguments={event.arguments}
              success={event.type === "result" ? event.success : undefined}
              output={event.type === "result" ? event.output : undefined}
            />
          ))}
        </div>
      )}

      {/* Typing indicator */}
      {typing && (
        <div className="flex justify-start px-4 py-1">
          <div className="rounded-2xl rounded-bl-sm bg-gray-800 px-4 py-3">
            <div className="flex items-center gap-1.5">
              <span
                className="h-2 w-2 rounded-full bg-gray-500 animate-bounce"
                style={{ animationDelay: "0ms" }}
              />
              <span
                className="h-2 w-2 rounded-full bg-gray-500 animate-bounce"
                style={{ animationDelay: "150ms" }}
              />
              <span
                className="h-2 w-2 rounded-full bg-gray-500 animate-bounce"
                style={{ animationDelay: "300ms" }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
