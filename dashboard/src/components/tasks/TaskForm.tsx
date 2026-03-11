import { useState } from "react";
import { Plus } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { api } from "@/lib/api";

interface TaskFormProps {
  onCreate: () => void;
}

export function TaskForm({ onCreate }: TaskFormProps) {
  const [type, setType] = useState<"reminder" | "cron">("reminder");
  const [description, setDescription] = useState("");
  const [schedule, setSchedule] = useState("");
  const [channel, setChannel] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!description.trim() || !schedule.trim()) return;

    setSubmitting(true);
    setError(null);
    try {
      await api.createTask({
        type,
        description: description.trim(),
        schedule: schedule.trim(),
        channel: channel.trim() || undefined,
      });
      setDescription("");
      setSchedule("");
      setChannel("");
      onCreate();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create task");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Card>
      <form onSubmit={handleSubmit} className="space-y-4">
        <h2 className="text-sm font-medium text-gray-300">New Task</h2>

        <div className="flex gap-3">
          <button
            type="button"
            onClick={() => setType("reminder")}
            className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
              type === "reminder"
                ? "bg-indigo-600 text-white"
                : "border border-gray-700 bg-gray-800 text-gray-400 hover:text-gray-200"
            }`}
          >
            Reminder
          </button>
          <button
            type="button"
            onClick={() => setType("cron")}
            className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
              type === "cron"
                ? "bg-indigo-600 text-white"
                : "border border-gray-700 bg-gray-800 text-gray-400 hover:text-gray-200"
            }`}
          >
            Cron
          </button>
        </div>

        <input
          type="text"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Task description..."
          className="w-full rounded-md border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-200 placeholder-gray-600 focus:border-indigo-500 focus:outline-none"
        />

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <input
            type={type === "reminder" ? "datetime-local" : "text"}
            value={schedule}
            onChange={(e) => setSchedule(e.target.value)}
            placeholder={type === "cron" ? "*/5 * * * * (cron)" : ""}
            className="w-full rounded-md border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-200 placeholder-gray-600 focus:border-indigo-500 focus:outline-none"
          />
          <input
            type="text"
            value={channel}
            onChange={(e) => setChannel(e.target.value)}
            placeholder="Channel (optional)"
            className="w-full rounded-md border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-200 placeholder-gray-600 focus:border-indigo-500 focus:outline-none"
          />
        </div>

        {error && <p className="text-xs text-red-400">{error}</p>}

        <button
          type="submit"
          disabled={submitting || !description.trim() || !schedule.trim()}
          className="flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-500 disabled:opacity-50"
        >
          <Plus className="h-4 w-4" />
          {submitting ? "Creating..." : "Create Task"}
        </button>
      </form>
    </Card>
  );
}
