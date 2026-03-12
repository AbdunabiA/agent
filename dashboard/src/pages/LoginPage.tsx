import { useState } from "react";
import { setToken } from "@/lib/api";

interface LoginPageProps {
  onLogin: () => void;
}

export function LoginPage({ onLogin }: LoginPageProps) {
  const [token, setTokenValue] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const res = await fetch(
        `${import.meta.env.VITE_API_URL ?? ""}/api/v1/status`,
        { headers: { Authorization: `Bearer ${token}` } },
      );

      if (res.ok) {
        setToken(token);
        onLogin();
      } else {
        setError("Invalid token. Check your gateway token and try again.");
      }
    } catch {
      setError("Cannot connect to the agent gateway.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-950 px-4">
      <div className="w-full max-w-sm space-y-6">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-white">Agent</h1>
          <p className="text-gray-400 mt-1 text-sm">
            Enter your gateway token to continue
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="token" className="block text-sm text-gray-400 mb-1">
              Gateway Token
            </label>
            <input
              id="token"
              type="password"
              value={token}
              onChange={(e) => setTokenValue(e.target.value)}
              placeholder="Paste your token here"
              className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500"
              autoFocus
            />
          </div>

          {error && (
            <p className="text-red-400 text-sm">{error}</p>
          )}

          <button
            type="submit"
            disabled={!token || loading}
            className="w-full py-2 px-4 bg-cyan-600 hover:bg-cyan-500 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg font-medium transition-colors"
          >
            {loading ? "Connecting..." : "Login"}
          </button>
        </form>

        <p className="text-center text-xs text-gray-600">
          Token is in your <code className="text-gray-500">~/.config/agent/.env</code> file
          (GATEWAY_TOKEN)
        </p>
      </div>
    </div>
  );
}
