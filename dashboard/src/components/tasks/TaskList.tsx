import { Clock, CalendarCheck, Trash2, AlertCircle, CheckCircle } from "lucide-react";
import { useState } from "react";
import { Card } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { api } from "@/lib/api";
import type { ScheduledTask } from "@/lib/types";

interface TaskListProps {
  tasks: ScheduledTask[];
  onDelete: () => void;
}

const statusIcons: Record<string, React.ElementType> = {
  pending: Clock,
  running: AlertCircle,
  completed: CheckCircle,
  failed: AlertCircle,
};

const statusColors: Record<string, string> = {
  pending: "text-yellow-400",
  running: "text-blue-400",
  completed: "text-green-400",
  failed: "text-red-400",
};

export function TaskList({ tasks, onDelete }: TaskListProps) {
  const [deleting, setDeleting] = useState<string | null>(null);

  const handleDelete = async (taskId: string) => {
    if (!confirm("Delete this task?")) return;
    setDeleting(taskId);
    try {
      await api.deleteTask(taskId);
      onDelete();
    } catch {
      // ignore
    } finally {
      setDeleting(null);
    }
  };

  if (tasks.length === 0) {
    return (
      <EmptyState
        icon={CalendarCheck}
        title="No tasks"
        description="Create a reminder or cron task to get started."
      />
    );
  }

  return (
    <div className="space-y-3">
      {tasks.map((task) => {
        const StatusIcon = statusIcons[task.status] ?? Clock;
        const color = statusColors[task.status] ?? "text-gray-400";

        return (
          <Card key={task.id}>
            <div className="flex items-start gap-3">
              <StatusIcon className={`mt-0.5 h-4 w-4 shrink-0 ${color}`} />
              <div className="min-w-0 flex-1 space-y-1">
                <p className="text-sm text-gray-200">{task.description}</p>
                <div className="flex flex-wrap gap-3 text-xs text-gray-500">
                  <span className="rounded bg-gray-800 px-1.5 py-0.5">
                    {task.type}
                  </span>
                  <span>{task.schedule}</span>
                  {task.channel && <span>#{task.channel}</span>}
                  {task.next_run && (
                    <span>Next: {new Date(task.next_run).toLocaleString()}</span>
                  )}
                  {task.last_run && (
                    <span>Last: {new Date(task.last_run).toLocaleString()}</span>
                  )}
                </div>
              </div>
              <button
                onClick={() => handleDelete(task.id)}
                disabled={deleting === task.id}
                className="shrink-0 rounded p-1 text-gray-600 transition-colors hover:bg-gray-800 hover:text-red-400"
                aria-label="Delete task"
              >
                <Trash2 className="h-4 w-4" />
              </button>
            </div>
          </Card>
        );
      })}
    </div>
  );
}
