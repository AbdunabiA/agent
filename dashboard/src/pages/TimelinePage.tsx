import { TimelineView } from "@/components/timeline/TimelineView";

export function TimelinePage() {
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-100">Timeline</h1>
      <TimelineView />
    </div>
  );
}
