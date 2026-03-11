import { useRef, useEffect } from "react";

interface SoulEditorProps {
  content: string;
  onChange: (content: string) => void;
  onSave: () => void;
}

export function SoulEditor({ content, onChange, onSave }: SoulEditorProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault();
        onSave();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onSave]);

  return (
    <textarea
      ref={textareaRef}
      value={content}
      onChange={(e) => onChange(e.target.value)}
      className="h-full w-full resize-none rounded-xl border border-gray-800 bg-gray-900 p-4 font-mono text-sm text-gray-200 placeholder-gray-600 focus:border-indigo-500 focus:outline-none"
      placeholder="# soul.md — Define your agent's personality..."
      spellCheck={false}
    />
  );
}
