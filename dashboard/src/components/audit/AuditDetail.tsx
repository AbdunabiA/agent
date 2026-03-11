import type { AuditEntry } from "@/lib/types";

interface AuditDetailProps {
  entry: AuditEntry;
}

export function AuditDetail({ entry }: AuditDetailProps) {
  return (
    <tr>
      <td colSpan={5} className="px-4 pb-3">
        <div className="rounded-lg border border-gray-800 bg-gray-800/50 p-3 space-y-2">
          <div>
            <span className="text-xs font-medium text-gray-500">Trigger: </span>
            <span className="text-xs text-gray-400">{entry.trigger}</span>
          </div>
          {entry.error && (
            <div>
              <span className="text-xs font-medium text-red-400">Error: </span>
              <span className="text-xs text-red-300">{entry.error}</span>
            </div>
          )}
        </div>
      </td>
    </tr>
  );
}
