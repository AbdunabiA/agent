import { useState } from "react";
import { Wrench, ChevronDown, ChevronUp } from "lucide-react";
import { cn } from "@/lib/utils";

interface ToolCallCardProps {
  name: string;
  arguments?: Record<string, unknown>;
  success?: boolean;
  output?: string;
}

export function ToolCallCard({ name, arguments: args, success, output }: ToolCallCardProps) {
  const [expanded, setExpanded] = useState(false);

  const borderColor =
    success === true
      ? "border-l-green-500"
      : success === false
        ? "border-l-red-500"
        : "border-l-gray-500";

  return (
    <div
      className={cn(
        "rounded-lg border border-gray-700 bg-gray-800/50 border-l-4 text-sm",
        borderColor,
      )}
    >
      {/* Header */}
      <button
        type="button"
        onClick={() => setExpanded((prev) => !prev)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-gray-300 hover:text-gray-100 transition-colors"
      >
        <Wrench className="h-3.5 w-3.5 shrink-0 text-gray-400" />
        <span className="flex-1 font-medium truncate">{name}</span>
        {expanded ? (
          <ChevronUp className="h-3.5 w-3.5 shrink-0 text-gray-500" />
        ) : (
          <ChevronDown className="h-3.5 w-3.5 shrink-0 text-gray-500" />
        )}
      </button>

      {/* Expanded content */}
      {expanded && (
        <div className="border-t border-gray-700 px-3 py-2 space-y-2">
          {args && Object.keys(args).length > 0 && (
            <div>
              <p className="text-xs font-medium text-gray-400 mb-1">Arguments</p>
              <pre className="rounded bg-gray-900/80 p-2 text-xs text-gray-300 overflow-x-auto">
                <code>{JSON.stringify(args, null, 2)}</code>
              </pre>
            </div>
          )}
          {output != null && (
            <div>
              <p className="text-xs font-medium text-gray-400 mb-1">Output</p>
              <pre className="rounded bg-gray-900/80 p-2 text-xs text-gray-300 overflow-x-auto max-h-48 overflow-y-auto">
                <code>{output}</code>
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
