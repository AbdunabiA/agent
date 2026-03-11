import { useState } from "react";
import { Search } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { api } from "@/lib/api";
import type { VectorResult } from "@/lib/types";

export function SemanticSearch() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<VectorResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [searched, setSearched] = useState(false);

  async function handleSearch() {
    if (!query.trim()) return;
    setSearching(true);
    try {
      const data = await api.memorySearch(query.trim());
      setResults(data.results);
      setSearched(true);
    } catch {
      setResults([]);
    } finally {
      setSearching(false);
    }
  }

  return (
    <Card>
      <h3 className="mb-3 text-sm font-medium text-gray-300">Semantic Search</h3>
      <div className="flex gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-500" />
          <input
            type="text"
            placeholder="Search memories..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSearch()}
            className="w-full rounded-lg border border-gray-700 bg-gray-800 py-2 pl-9 pr-3 text-sm text-gray-200 placeholder-gray-500 focus:border-indigo-500 focus:outline-none"
          />
        </div>
        <button
          onClick={handleSearch}
          disabled={searching || !query.trim()}
          className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-500 disabled:opacity-50"
        >
          {searching ? "..." : "Search"}
        </button>
      </div>

      {searched && (
        <div className="mt-3 space-y-2">
          {results.length === 0 ? (
            <p className="py-4 text-center text-sm text-gray-500">No results found</p>
          ) : (
            results.map((r, i) => (
              <div
                key={i}
                className="flex items-start gap-3 rounded-lg border border-gray-800 bg-gray-800/50 p-3"
              >
                <span
                  className={`mt-0.5 shrink-0 rounded px-1.5 py-0.5 text-xs font-mono font-bold ${
                    r.similarity >= 0.9
                      ? "bg-green-900/50 text-green-400"
                      : r.similarity >= 0.7
                        ? "bg-yellow-900/50 text-yellow-400"
                        : "bg-gray-700 text-gray-400"
                  }`}
                >
                  {r.similarity.toFixed(2)}
                </span>
                <p className="text-sm text-gray-300">{r.text}</p>
              </div>
            ))
          )}
        </div>
      )}
    </Card>
  );
}
