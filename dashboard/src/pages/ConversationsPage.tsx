import { ConversationBrowser } from "@/components/conversations/ConversationBrowser";

export function ConversationsPage() {
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-100">Conversations</h1>
      <ConversationBrowser />
    </div>
  );
}
