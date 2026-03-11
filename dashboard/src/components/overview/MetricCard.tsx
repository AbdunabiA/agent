import type { LucideIcon } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { cn } from "@/lib/utils";

interface MetricCardProps {
  icon: LucideIcon;
  label: string;
  value: string | number;
  subValue?: string;
  className?: string;
}

export function MetricCard({ icon: Icon, label, value, subValue, className }: MetricCardProps) {
  return (
    <Card className={cn("flex items-start gap-3", className)}>
      <Icon className="h-5 w-5 text-indigo-400 mt-0.5 shrink-0" />
      <div className="min-w-0">
        <p className="text-sm text-gray-400">{label}</p>
        <p className="text-2xl font-bold text-gray-100 truncate">{value}</p>
        {subValue && <p className="text-xs text-gray-500 mt-0.5">{subValue}</p>}
      </div>
    </Card>
  );
}
