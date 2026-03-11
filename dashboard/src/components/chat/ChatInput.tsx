import { useState, useRef, useCallback, type KeyboardEvent, type ChangeEvent } from "react";
import { Send } from "lucide-react";
import { cn } from "@/lib/utils";
import { VoiceInput } from "./VoiceInput";

interface ChatInputProps {
  onSend: (message: string) => void;
  onVoiceData?: (audio: string, mimeType: string) => void;
  disabled?: boolean;
}

export function ChatInput({ onSend, onVoiceData, disabled = false }: ChatInputProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const adjustHeight = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, []);

  const handleChange = useCallback(
    (e: ChangeEvent<HTMLTextAreaElement>) => {
      setValue(e.target.value);
      adjustHeight();
    },
    [adjustHeight],
  );

  const handleSend = useCallback(() => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
    // Reset textarea height after clearing
    requestAnimationFrame(() => {
      const el = textareaRef.current;
      if (el) {
        el.style.height = "auto";
      }
    });
  }, [value, disabled, onSend]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  return (
    <div
      className={cn(
        "flex items-end gap-2 border-t border-gray-700 bg-gray-900 p-4",
        disabled && "opacity-50 cursor-not-allowed",
      )}
    >
      <textarea
        ref={textareaRef}
        value={value}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        placeholder="Type a message..."
        rows={1}
        className={cn(
          "flex-1 resize-none rounded-lg border border-gray-700 bg-gray-800 px-4 py-2.5",
          "text-sm text-gray-100 placeholder:text-gray-500",
          "focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500/50",
          "transition-colors",
          "max-h-[200px]",
          disabled && "cursor-not-allowed",
        )}
      />
      {onVoiceData && <VoiceInput onVoiceData={onVoiceData} disabled={disabled} />}
      <button
        type="button"
        onClick={handleSend}
        disabled={disabled || !value.trim()}
        className={cn(
          "flex h-10 w-10 shrink-0 items-center justify-center rounded-lg",
          "bg-indigo-600 text-white transition-colors",
          "hover:bg-indigo-500 active:bg-indigo-700",
          "disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-indigo-600",
        )}
      >
        <Send className="h-4 w-4" />
      </button>
    </div>
  );
}
