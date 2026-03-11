import { useState, useEffect, useCallback } from "react";
import { Save, FileText } from "lucide-react";
import { SoulEditor } from "@/components/soul/SoulEditor";
import { SoulPreview } from "@/components/soul/SoulPreview";
import { Spinner } from "@/components/ui/Spinner";
import { api } from "@/lib/api";

export function SoulEditorPage() {
  const [content, setContent] = useState("");
  const [original, setOriginal] = useState("");
  const [path, setPath] = useState("");
  const [lastModified, setLastModified] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<string | null>(null);

  useEffect(() => {
    api
      .getSoul()
      .then((res) => {
        setContent(res.content);
        setOriginal(res.content);
        setPath(res.path);
        setLastModified(res.last_modified);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const hasChanges = content !== original;

  const handleSave = useCallback(async () => {
    if (!hasChanges || saving) return;
    setSaving(true);
    setSaveMsg(null);
    try {
      const res = await api.updateSoul(content);
      if (res.success) {
        setOriginal(content);
        setSaveMsg("Saved");
        setTimeout(() => setSaveMsg(null), 2000);
      }
    } catch (err) {
      setSaveMsg(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }, [content, hasChanges, saving]);

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Spinner size="lg" />
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <FileText className="h-6 w-6 text-indigo-400" />
          <h1 className="text-2xl font-bold text-gray-100">Soul Editor</h1>
          {hasChanges && (
            <span className="rounded bg-yellow-600/20 px-2 py-0.5 text-xs text-yellow-400">
              Unsaved changes
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          {saveMsg && <span className="text-sm text-gray-400">{saveMsg}</span>}
          <button
            onClick={handleSave}
            disabled={!hasChanges || saving}
            className="flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-500 disabled:opacity-50"
          >
            <Save className="h-4 w-4" />
            {saving ? "Saving..." : "Save"}
          </button>
        </div>
      </div>

      {/* Meta info */}
      {(path || lastModified) && (
        <div className="flex gap-4 text-xs text-gray-500">
          {path && <span>Path: {path}</span>}
          {lastModified && <span>Modified: {new Date(lastModified).toLocaleString()}</span>}
        </div>
      )}

      {/* Editor + Preview split */}
      <div className="grid min-h-0 flex-1 grid-cols-1 gap-4 lg:grid-cols-2">
        <SoulEditor content={content} onChange={setContent} onSave={handleSave} />
        <SoulPreview content={content} />
      </div>
    </div>
  );
}
