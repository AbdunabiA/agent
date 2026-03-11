import { useState, useEffect, useCallback } from "react";
import { Search, Database } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { FactRow } from "./FactRow";
import { api } from "@/lib/api";
import type { Fact } from "@/lib/types";

export function FactsTable() {
  const [facts, setFacts] = useState<Fact[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("");
  const [offset, setOffset] = useState(0);
  const limit = 20;

  const loadFacts = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.memoryFacts(limit, offset, filter || undefined);
      setFacts(data.facts);
      setTotal(data.total);
    } catch {
      setFacts([]);
    } finally {
      setLoading(false);
    }
  }, [offset, filter]);

  useEffect(() => {
    loadFacts();
  }, [loadFacts]);

  async function handleDelete(factId: string) {
    try {
      await api.deleteFact(factId);
      setFacts((prev) => prev.filter((f) => f.id !== factId));
      setTotal((prev) => prev - 1);
    } catch {
      // ignore
    }
  }

  function handleFilterSubmit(e: React.FormEvent) {
    e.preventDefault();
    setOffset(0);
    loadFacts();
  }

  return (
    <Card>
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-sm font-medium text-gray-300">
          Facts{" "}
          <span className="text-gray-600">({total})</span>
        </h3>
        <form onSubmit={handleFilterSubmit} className="relative">
          <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-gray-500" />
          <input
            type="text"
            placeholder="Filter by key..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="rounded-md border border-gray-700 bg-gray-800 py-1.5 pl-8 pr-3 text-xs text-gray-200 placeholder-gray-500 focus:border-indigo-500 focus:outline-none"
          />
        </form>
      </div>

      {loading ? (
        <div className="space-y-2">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-10 animate-pulse rounded bg-gray-800" />
          ))}
        </div>
      ) : facts.length === 0 ? (
        <EmptyState icon={Database} title="No facts found" description="Memory facts will appear here as the agent learns" />
      ) : (
        <>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-700 text-left text-xs text-gray-500">
                  <th className="px-4 py-2">Key</th>
                  <th className="px-4 py-2">Value</th>
                  <th className="px-4 py-2 text-center">Category</th>
                  <th className="px-4 py-2 text-center">Confidence</th>
                  <th className="px-4 py-2 text-right w-10"></th>
                </tr>
              </thead>
              <tbody>
                {facts.map((fact) => (
                  <FactRow key={fact.id} fact={fact} onDelete={handleDelete} />
                ))}
              </tbody>
            </table>
          </div>

          {total > offset + limit && (
            <div className="mt-4 flex justify-center">
              <button
                onClick={() => setOffset((prev) => prev + limit)}
                className="rounded-lg border border-gray-700 px-4 py-2 text-xs text-gray-400 transition-colors hover:border-gray-600 hover:text-gray-200"
              >
                Load more
              </button>
            </div>
          )}
        </>
      )}
    </Card>
  );
}
