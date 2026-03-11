import { useState, useEffect, useCallback } from "react";
import { CalendarCheck } from "lucide-react";
import { TaskList } from "@/components/tasks/TaskList";
import { TaskForm } from "@/components/tasks/TaskForm";
import { Spinner } from "@/components/ui/Spinner";
import { api } from "@/lib/api";
import type { ScheduledTask } from "@/lib/types";

export function TasksPage() {
  const [tasks, setTasks] = useState<ScheduledTask[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchTasks = useCallback(() => {
    api
      .listTasks()
      .then(setTasks)
      .catch(() => setTasks([]))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetchTasks();
  }, [fetchTasks]);

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <Spinner size="lg" />
      </div>
    );
  }

  const active = tasks.filter((t) => t.status !== "completed");
  const completed = tasks.filter((t) => t.status === "completed");

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <CalendarCheck className="h-6 w-6 text-indigo-400" />
        <h1 className="text-2xl font-bold text-gray-100">Tasks</h1>
        <span className="text-sm text-gray-500">{tasks.length} total</span>
      </div>

      <TaskForm onCreate={fetchTasks} />

      {active.length > 0 && (
        <div className="space-y-3">
          <h2 className="text-sm font-medium text-gray-400">
            Active ({active.length})
          </h2>
          <TaskList tasks={active} onDelete={fetchTasks} />
        </div>
      )}

      {completed.length > 0 && (
        <div className="space-y-3">
          <h2 className="text-sm font-medium text-gray-400">
            Completed ({completed.length})
          </h2>
          <TaskList tasks={completed} onDelete={fetchTasks} />
        </div>
      )}
    </div>
  );
}
