import { useState, useEffect, useCallback } from "react";
import { Puzzle } from "lucide-react";
import { SkillCard } from "@/components/skills/SkillCard";
import { Spinner } from "@/components/ui/Spinner";
import { EmptyState } from "@/components/ui/EmptyState";
import { api } from "@/lib/api";
import type { SkillInfo } from "@/lib/types";

export function SkillsPage() {
  const [skills, setSkills] = useState<SkillInfo[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchSkills = useCallback(() => {
    api
      .skills()
      .then(setSkills)
      .catch(() => setSkills([]))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetchSkills();
  }, [fetchSkills]);

  const loaded = skills.filter((s) => s.loaded).length;

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <Spinner size="lg" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-100">Skills</h1>
        <div className="flex items-center gap-3 text-xs text-gray-500">
          <span>{loaded}/{skills.length} loaded</span>
        </div>
      </div>

      {skills.length === 0 ? (
        <EmptyState
          icon={Puzzle}
          title="No skills found"
          description="Place skills in the skills/ directory or run: agent skills create my-skill"
        />
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {skills.map((skill) => (
            <SkillCard key={skill.name} skill={skill} onRefresh={fetchSkills} />
          ))}
        </div>
      )}
    </div>
  );
}
