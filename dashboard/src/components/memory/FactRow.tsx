import { useState } from "react";
import { Trash2 } from "lucide-react";
import type { Fact } from "@/lib/types";

interface FactRowProps {
  fact: Fact;
  onDelete: (id: string) => void;
}

export function FactRow({ fact, onDelete }: FactRowProps) {
  const [confirming, setConfirming] = useState(false);

  function handleDelete() {
    if (confirming) {
      onDelete(fact.id);
      setConfirming(false);
    } else {
      setConfirming(true);
      setTimeout(() => setConfirming(false), 3000);
    }
  }

  return (
    <tr className="border-b border-gray-800 transition-colors hover:bg-gray-800/50">
      <td className="px-4 py-3 text-sm font-mono text-indigo-400">{fact.key}</td>
      <td className="px-4 py-3 text-sm text-gray-300">{fact.value}</td>
      <td className="px-4 py-3 text-center">
        <span className="text-xs text-gray-500">{fact.category}</span>
      </td>
      <td className="px-4 py-3 text-center">
        <span
          className={`rounded px-1.5 py-0.5 text-xs font-mono ${
            fact.confidence >= 0.9
              ? "bg-green-900/50 text-green-400"
              : fact.confidence >= 0.7
                ? "bg-yellow-900/50 text-yellow-400"
                : "bg-red-900/50 text-red-400"
          }`}
        >
          {fact.confidence.toFixed(2)}
        </span>
      </td>
      <td className="px-4 py-3 text-right">
        <button
          onClick={handleDelete}
          className={`rounded p-1 transition-colors ${
            confirming
              ? "bg-red-600 text-white"
              : "text-gray-500 hover:bg-gray-700 hover:text-red-400"
          }`}
          title={confirming ? "Click again to confirm" : "Delete fact"}
        >
          <Trash2 className="h-4 w-4" />
        </button>
      </td>
    </tr>
  );
}
