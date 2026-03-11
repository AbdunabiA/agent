import { useEffect, useCallback } from "react";
import { useChatSocket } from "@/hooks/useChatSocket";
import { api } from "@/lib/api";
import { MessageList } from "./MessageList";
import { ChatInput } from "./ChatInput";
import { ErrorMessage } from "@/components/ui/ErrorMessage";
import { cn } from "@/lib/utils";

interface ChatWindowProps {
  sessionId?: string;
}

export function ChatWindow({ sessionId }: ChatWindowProps) {
  const {
    messages,
    toolEvents,
    typing,
    connected,
    error,
    sendMessage,
    sendVoice,
    setMessages,
  } = useChatSocket(sessionId);

  // Load existing messages when sessionId is provided or changes
  useEffect(() => {
    if (!sessionId) return;

    let cancelled = false;
    api.messages(sessionId).then((data) => {
      if (!cancelled) {
        setMessages(data);
      }
    }).catch(() => {
      // Silently ignore fetch errors; the socket error state handles display
    });

    return () => {
      cancelled = true;
    };
  }, [sessionId, setMessages]);

  const handleSend = useCallback(
    (content: string) => {
      sendMessage(content);
    },
    [sendMessage],
  );

  const handleVoiceData = useCallback(
    (audio: string, mimeType: string) => {
      sendVoice(audio, mimeType);
    },
    [sendVoice],
  );

  return (
    <div className="flex h-full flex-col bg-gray-900">
      {/* Connection status indicator */}
      <div
        className={cn(
          "flex items-center gap-2 border-b border-gray-800 px-4 py-2 text-xs transition-colors",
          connected ? "text-gray-500" : "text-yellow-500 bg-yellow-950/20",
        )}
      >
        <span
          className={cn(
            "h-2 w-2 rounded-full",
            connected ? "bg-green-500" : "bg-yellow-500 animate-pulse",
          )}
        />
        {connected ? "Connected" : "Reconnecting..."}
      </div>

      {/* Error banner */}
      {error && (
        <div className="px-4 pt-3">
          <ErrorMessage message={error} />
        </div>
      )}

      {/* Message list (flex-1 to fill remaining space) */}
      <MessageList
        messages={messages}
        toolEvents={toolEvents}
        typing={typing}
      />

      {/* Input bar pinned to bottom */}
      <ChatInput
        onSend={handleSend}
        onVoiceData={handleVoiceData}
        disabled={!connected || typing}
      />
    </div>
  );
}
