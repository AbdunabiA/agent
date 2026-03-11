import ReactMarkdown from "react-markdown";
import type { MessageOut } from "@/lib/types";
import { cn, formatRelativeTime } from "@/lib/utils";

interface ChatMessageProps {
  message: MessageOut;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const { role, content, model, timestamp } = message;

  // System and tool messages: small, centered, gray italic
  if (role === "system" || role === "tool") {
    return (
      <div className="flex justify-center px-4 py-1">
        <div className="max-w-md text-center">
          <p className="text-xs italic text-gray-500">{content}</p>
          <span className="text-[10px] text-gray-600">
            {formatRelativeTime(timestamp)}
          </span>
        </div>
      </div>
    );
  }

  const isUser = role === "user";

  return (
    <div
      className={cn(
        "flex w-full px-4 py-1",
        isUser ? "justify-end" : "justify-start",
      )}
    >
      <div className={cn("max-w-[75%] flex flex-col", isUser ? "items-end" : "items-start")}>
        {/* Bubble */}
        <div
          className={cn(
            "px-4 py-2.5 text-sm text-gray-100",
            isUser
              ? "bg-indigo-600 rounded-2xl rounded-br-sm"
              : "bg-gray-800 rounded-2xl rounded-bl-sm",
          )}
        >
          <div className="prose prose-sm prose-invert max-w-none [&>p]:my-1 [&>p:first-child]:mt-0 [&>p:last-child]:mb-0 [&_code]:rounded [&_code]:bg-gray-700/60 [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:text-xs [&_pre]:my-2 [&_pre]:rounded-lg [&_pre]:bg-gray-900/80 [&_pre]:p-3 [&_pre_code]:bg-transparent [&_pre_code]:p-0">
            <ReactMarkdown>{content}</ReactMarkdown>
          </div>
        </div>

        {/* Meta row: timestamp + model */}
        <div className="mt-1 flex items-center gap-2">
          <span className="text-xs text-gray-500">
            {formatRelativeTime(timestamp)}
          </span>
          {!isUser && model && (
            <span className="text-xs text-gray-600">{model}</span>
          )}
        </div>
      </div>
    </div>
  );
}
