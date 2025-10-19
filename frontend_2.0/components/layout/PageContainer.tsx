import type { HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

/**
 * Consistent horizontal spacing wrapper that mirrors the dashboard layout.
 */
export function PageContainer({
  className,
  ...props
}: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("container-wide", className)} {...props} />;
}
