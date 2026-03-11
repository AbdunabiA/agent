import { cn } from "@/lib/utils";

type Variant = "green" | "yellow" | "red" | "gray";

interface StatusBadgeProps {
  variant: Variant;
  label: string;
  className?: string;
}

const dotColor: Record<Variant, string> = {
  green: "bg-green-500",
  yellow: "bg-yellow-500",
  red: "bg-red-500",
  gray: "bg-gray-500",
};

export function StatusBadge({ variant, label, className }: StatusBadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 text-xs font-medium text-gray-300",
        className,
      )}
    >
      <span className={cn("h-2 w-2 rounded-full", dotColor[variant])} />
      {label}
    </span>
  );
}
