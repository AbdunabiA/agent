import { useState, useEffect, useCallback } from "react";
import { Settings, Save, RefreshCw, AlertTriangle } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Spinner } from "@/components/ui/Spinner";
import { api } from "@/lib/api";
import type { EditableConfig, EditableConfigSection, ConfigFieldMeta } from "@/lib/types";

/** Sections that need a restart to take effect. */
const RESTART_SECTIONS = new Set(["gateway", "channels"]);

/** Nice display names for config sections. */
const SECTION_LABELS: Record<string, string> = {
  agent: "Agent Identity",
  models: "Models",
  channels: "Channels",
  tools: "Tools",
  memory: "Memory",
  skills: "Skills",
  gateway: "Gateway",
  logging: "Logging",
  desktop: "Desktop",
  voice: "Voice",
  workspaces: "Workspaces",
};

/** Editable JSON textarea for objects and arrays of objects. */
function JsonTextarea({
  value,
  onChange,
}: {
  value: unknown;
  onChange: (val: unknown) => void;
}) {
  const [text, setText] = useState(() => JSON.stringify(value, null, 2));
  const [parseError, setParseError] = useState<string | null>(null);

  // Sync external value changes
  useEffect(() => {
    setText(JSON.stringify(value, null, 2));
    setParseError(null);
  }, [value]);

  const handleBlur = () => {
    try {
      const parsed = JSON.parse(text);
      setParseError(null);
      onChange(parsed);
    } catch (e) {
      setParseError((e as Error).message);
    }
  };

  const rows = Math.min(Math.max(text.split("\n").length, 2), 12);

  return (
    <div className="w-full">
      <textarea
        value={text}
        onChange={(e) => {
          setText(e.target.value);
          setParseError(null);
        }}
        onBlur={handleBlur}
        rows={rows}
        className={`w-full rounded border bg-gray-950 px-2 py-1 font-mono text-xs text-gray-200 focus:outline-none ${
          parseError
            ? "border-red-500 focus:border-red-400"
            : "border-gray-700 focus:border-indigo-500"
        }`}
      />
      {parseError && (
        <p className="mt-0.5 text-[10px] text-red-400">Invalid JSON: {parseError}</p>
      )}
    </div>
  );
}

function FieldInput({
  meta,
  value,
  onChange,
}: {
  meta: ConfigFieldMeta;
  value: unknown;
  onChange: (val: unknown) => void;
}) {
  if (!meta.editable) {
    return (
      <span className="text-xs text-gray-500 italic">
        {typeof meta.value === "string" && meta.value ? meta.value : "(set via .env)"}
      </span>
    );
  }

  if (meta.options) {
    return (
      <select
        value={String(value ?? "")}
        onChange={(e) => onChange(e.target.value)}
        className="rounded border border-gray-700 bg-gray-950 px-2 py-1 text-xs text-gray-200 focus:border-indigo-500 focus:outline-none"
      >
        {meta.options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    );
  }

  if (meta.type === "boolean") {
    return (
      <button
        type="button"
        onClick={() => onChange(!value)}
        className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
          value ? "bg-indigo-600" : "bg-gray-700"
        }`}
      >
        <span
          className={`inline-block h-3.5 w-3.5 rounded-full bg-white shadow-sm transition-transform ${
            value ? "translate-x-[18px]" : "translate-x-[2px]"
          }`}
        />
      </button>
    );
  }

  if (meta.type === "number") {
    return (
      <input
        type="number"
        value={value as number}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-24 rounded border border-gray-700 bg-gray-950 px-2 py-1 text-xs text-gray-200 focus:border-indigo-500 focus:outline-none"
      />
    );
  }

  if (meta.type === "array") {
    // Arrays of objects (e.g. routing rules) — editable JSON textarea
    const isObjectArray =
      Array.isArray(value) && value.length > 0 && typeof value[0] === "object";
    if (isObjectArray) {
      return (
        <JsonTextarea value={value} onChange={onChange} />
      );
    }
    return (
      <input
        type="text"
        value={Array.isArray(value) ? (value as string[]).join(", ") : String(value ?? "")}
        onChange={(e) =>
          onChange(
            e.target.value
              .split(",")
              .map((s) => s.trim())
              .filter(Boolean),
          )
        }
        placeholder="comma-separated"
        className="w-full rounded border border-gray-700 bg-gray-950 px-2 py-1 text-xs text-gray-200 focus:border-indigo-500 focus:outline-none"
      />
    );
  }

  if (meta.type === "object" && !meta.fields) {
    return (
      <JsonTextarea value={value} onChange={onChange} />
    );
  }

  // Default: string input
  return (
    <input
      type="text"
      value={String(value ?? "")}
      onChange={(e) => onChange(e.target.value)}
      className="w-full rounded border border-gray-700 bg-gray-950 px-2 py-1 text-xs text-gray-200 focus:border-indigo-500 focus:outline-none"
    />
  );
}

function SectionEditor({
  name,
  fields,
  onSave,
  saving,
}: {
  name: string;
  fields: EditableConfigSection;
  onSave: (section: string, data: Record<string, unknown>) => Promise<void>;
  saving: string | null;
}) {
  const [values, setValues] = useState<Record<string, unknown>>(() => {
    const init: Record<string, unknown> = {};
    for (const [k, meta] of Object.entries(fields)) {
      init[k] = meta.value;
    }
    return init;
  });
  const [dirty, setDirty] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const needsRestart = RESTART_SECTIONS.has(name);
  const hasEditableFields = Object.values(fields).some((f) => f.editable);

  const handleChange = useCallback((field: string, val: unknown) => {
    setValues((prev) => ({ ...prev, [field]: val }));
    setDirty(true);
    setSuccess(false);
    setError(null);
  }, []);

  const handleSave = async () => {
    setError(null);
    // Only send editable fields (including nested object sub-fields)
    const data: Record<string, unknown> = {};
    for (const [k, meta] of Object.entries(fields)) {
      if (meta.editable) {
        if (meta.type === "object" && meta.fields) {
          // For nested objects, only include editable sub-fields
          const nested: Record<string, unknown> = {};
          const parentVal =
            (values[k] as Record<string, unknown>) ?? {};
          for (const [subK, subMeta] of Object.entries(meta.fields)) {
            if (subMeta.editable) {
              nested[subK] = parentVal[subK] ?? subMeta.value;
            }
          }
          data[k] = nested;
        } else {
          data[k] = values[k];
        }
      }
    }
    try {
      await onSave(name, data);
      setDirty(false);
      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Save failed");
    }
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <h3 className="text-sm font-medium capitalize text-indigo-400">
          {SECTION_LABELS[name] ?? name}
        </h3>
        {needsRestart && dirty && (
          <span className="flex items-center gap-1 rounded bg-yellow-900/50 px-1.5 py-0.5 text-[10px] text-yellow-400">
            <AlertTriangle className="h-3 w-3" />
            Restart required
          </span>
        )}
        {success && (
          <span className="text-[10px] text-green-400">Saved</span>
        )}
      </div>

      <div className="rounded-lg border border-gray-800 bg-gray-950 p-3 space-y-2">
        {Object.entries(fields).map(([fieldName, meta]) => {
          // Nested object with sub-fields (e.g. claude_sdk)
          if (meta.type === "object" && meta.fields) {
            const parentVal =
              (values[fieldName] as Record<string, unknown>) ?? {};
            return (
              <div key={fieldName} className="space-y-1.5">
                <span className="text-[11px] font-medium text-indigo-300/80">
                  {fieldName}
                </span>
                <div className="ml-3 space-y-2 border-l border-gray-800 pl-3">
                  {Object.entries(meta.fields).map(([subName, subMeta]) => (
                    <div
                      key={`${fieldName}.${subName}`}
                      className="flex items-center gap-2 text-xs"
                    >
                      <span className="w-36 shrink-0 text-gray-500">
                        {subName}
                      </span>
                      <div className="flex-1">
                        <FieldInput
                          meta={subMeta}
                          value={parentVal[subName] ?? subMeta.value}
                          onChange={(val) => {
                            const updated = { ...parentVal, [subName]: val };
                            handleChange(fieldName, updated);
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            );
          }

          // Regular flat field
          return (
            <div key={fieldName} className="flex items-center gap-2 text-xs">
              <span className="w-40 shrink-0 text-gray-500">{fieldName}</span>
              <div className="flex-1">
                <FieldInput
                  meta={meta}
                  value={values[fieldName]}
                  onChange={(val) => handleChange(fieldName, val)}
                />
              </div>
            </div>
          );
        })}
      </div>

      {error && <p className="text-xs text-red-400">{error}</p>}

      {hasEditableFields && (
        <div className="flex justify-end">
          <button
            type="button"
            onClick={handleSave}
            disabled={!dirty || saving === name}
            className="flex items-center gap-1.5 rounded-md bg-indigo-600 px-3 py-1 text-xs font-medium text-white transition-colors hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {saving === name ? (
              <RefreshCw className="h-3 w-3 animate-spin" />
            ) : (
              <Save className="h-3 w-3" />
            )}
            Save
          </button>
        </div>
      )}
    </div>
  );
}

export function ConfigEditor() {
  const [config, setConfig] = useState<EditableConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    api
      .editableConfig()
      .then(setConfig)
      .catch(() => setConfig(null))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const handleSave = async (section: string, data: Record<string, unknown>) => {
    setSaving(section);
    try {
      await api.updateConfig(section, data);
      // Reload to get fresh values
      const fresh = await api.editableConfig();
      setConfig(fresh);
    } finally {
      setSaving(null);
    }
  };

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
      <div className="space-y-6">
        <div className="flex items-center gap-2">
          <Settings className="h-5 w-5 text-gray-400" />
          <h2 className="text-lg font-semibold text-gray-100">Configuration</h2>
          <button
            type="button"
            onClick={load}
            className="ml-auto text-gray-500 hover:text-gray-300"
            title="Reload config"
          >
            <RefreshCw className="h-4 w-4" />
          </button>
        </div>

        <p className="text-xs text-gray-600">
          API keys and tokens are read-only — set them in your <code>.env</code> file.
        </p>

        {sections.map(([name, fields]) => (
          <SectionEditor
            key={name}
            name={name}
            fields={fields}
            onSave={handleSave}
            saving={saving}
          />
        ))}
      </div>
    </Card>
  );
}
