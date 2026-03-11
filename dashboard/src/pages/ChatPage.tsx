import { useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useApi } from "@/hooks/useApi";
import { api } from "@/lib/api";
import { ConversationList } from "@/components/chat/ConversationList";
import { ChatWindow } from "@/components/chat/ChatWindow";
import { Spinner } from "@/components/ui/Spinner";

export function ChatPage() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();

  const { data: sessions, loading } = useApi(() => api.conversations(), []);

  const handleSelect = useCallback(
    (id: string) => {
      navigate(`/chat/${id}`);
    },
    [navigate],
  );

  const handleNewChat = useCallback(() => {
    navigate("/chat");
  }, [navigate]);

  return (
    <div className="flex h-[calc(100vh-theme(spacing.14)-theme(spacing.12))] gap-4">
      {/* Conversation sidebar */}
      <div className="w-72 shrink-0 flex flex-col bg-gray-900 rounded-xl border border-gray-800 p-3">
        <ConversationList
          sessions={sessions ?? []}
          activeId={sessionId ?? null}
          onSelect={handleSelect}
          onNewChat={handleNewChat}
        />
        {loading && (
          <div className="flex justify-center py-4">
            <Spinner size="sm" />
          </div>
        )}
      </div>

      {/* Chat area */}
      <div className="flex-1 bg-gray-900 rounded-xl border border-gray-800 flex flex-col overflow-hidden">
        <ChatWindow key={sessionId ?? "__new"} sessionId={sessionId} />
      </div>
    </div>
  );
}
