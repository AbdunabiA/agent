import { useEffect, useState } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Layout } from "@/components/layout/Layout";
import { OverviewPage } from "@/pages/OverviewPage";
import { ChatPage } from "@/pages/ChatPage";
import { MemoryPage } from "@/pages/MemoryPage";
import { ConversationsPage } from "@/pages/ConversationsPage";
import { TimelinePage } from "@/pages/TimelinePage";
import { CostsPage } from "@/pages/CostsPage";
import { AuditPage } from "@/pages/AuditPage";
import { SettingsPage } from "@/pages/SettingsPage";
import { SoulEditorPage } from "@/pages/SoulEditorPage";
import { ToolsPage } from "@/pages/ToolsPage";
import { SkillsPage } from "@/pages/SkillsPage";
import { TasksPage } from "@/pages/TasksPage";
import { WorkspacesPage } from "@/pages/WorkspacesPage";
import { LoginPage } from "@/pages/LoginPage";
import { checkAuth } from "@/lib/api";
import { Spinner } from "@/components/ui/Spinner";

export default function App() {
  const [authState, setAuthState] = useState<
    "loading" | "open" | "authenticated" | "login_required"
  >("loading");

  useEffect(() => {
    checkAuth().then(setAuthState);
  }, []);

  if (authState === "loading") {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-950">
        <Spinner />
      </div>
    );
  }

  if (authState === "login_required") {
    return <LoginPage onLogin={() => setAuthState("authenticated")} />;
  }

  return (
    <BrowserRouter basename="/dashboard">
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<OverviewPage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/chat/:sessionId" element={<ChatPage />} />
          <Route path="/memory" element={<MemoryPage />} />
          <Route path="/conversations" element={<ConversationsPage />} />
          <Route path="/timeline" element={<TimelinePage />} />
          <Route path="/costs" element={<CostsPage />} />
          <Route path="/audit" element={<AuditPage />} />
          <Route path="/tools" element={<ToolsPage />} />
          <Route path="/skills" element={<SkillsPage />} />
          <Route path="/tasks" element={<TasksPage />} />
          <Route path="/workspaces" element={<WorkspacesPage />} />
          <Route path="/soul" element={<SoulEditorPage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
