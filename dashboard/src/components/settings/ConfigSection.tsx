import { useState, useEffect } from "react";
import { Settings } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Spinner } from "@/components/ui/Spinner";
import { api } from "@/lib/api";
import type { DisplayConfig } from "@/lib/types";

export function ConfigSection() {
  const [config, setConfig] = useState<DisplayConfig | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .displayConfig()
      .then(setConfig)
      .catch(() => setConfig(null))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <Card>
        <div className="flex items-center justify-center py-12">
          <Spinner />
        </div>
      </Card>
    );
  }

  if (!config) {
    return (
      <Card>
        <p className="text-sm text-gray-500">Could not load configuration.</p>
      </Card>
    );
  }

  const sections = Object.entries(config);

  return (
    <Card>
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Settings className="h-5 w-5 text-gray-400" />
          <h2 className="text-lg font-semibold text-gray-100">Configuration</h2>
          <span className="ml-auto text-xs text-gray-600">Read-only</span>
        </div>

        <div className="space-y-4">
          {sections.map(([section, value]) => (
            <div key={section} className="space-y-1">
              <h3 className="text-sm font-medium capitalize text-indigo-400">{section}</h3>
              <div className="rounded-lg border border-gray-800 bg-gray-950 p-3">
                {typeof value === "object" && value !== null ? (
                  <div className="space-y-1">
                    {Object.entries(value as Record<string, unknown>).map(([k, v]) => (
                      <div key={k} className="flex items-start gap-2 text-xs">
                        <span className="w-40 shrink-0 text-gray-500">{k}:</span>
                        <span className="break-all text-gray-300">
                          {typeof v === "object" ? JSON.stringify(v) : String(v)}
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <span className="text-xs text-gray-300">{String(value)}</span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
}
