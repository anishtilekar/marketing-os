import { useCallback, useEffect, useState } from "react";
import CreateRun from "./components/CreateRun.jsx";
import RunProgress from "./components/RunProgress.jsx";
import Results from "./components/Results.jsx";

const RECENT_KEY = "marketingos.recentRuns";

function loadRecent() {
  try {
    return JSON.parse(localStorage.getItem(RECENT_KEY)) ?? [];
  } catch {
    return [];
  }
}

export default function App() {
  // view: { name: "home" } | { name: "running", runId } | { name: "results", runId }
  const [view, setView] = useState({ name: "home" });
  const [recentRuns, setRecentRuns] = useState(loadRecent);

  useEffect(() => {
    localStorage.setItem(RECENT_KEY, JSON.stringify(recentRuns.slice(0, 8)));
  }, [recentRuns]);

  const rememberRun = useCallback((runId, patch = {}) => {
    setRecentRuns((prev) => {
      const rest = prev.filter((r) => r.runId !== runId);
      const existing = prev.find((r) => r.runId === runId) ?? {};
      return [
        { runId, createdAt: existing.createdAt ?? Date.now(), ...existing, ...patch },
        ...rest,
      ];
    });
  }, []);

  const startRun = useCallback(
    (runId) => {
      rememberRun(runId, { status: "running" });
      setView({ name: "running", runId });
    },
    [rememberRun]
  );

  const openResults = useCallback(
    (runId) => {
      rememberRun(runId);
      setView({ name: "results", runId });
    },
    [rememberRun]
  );

  const finishRun = useCallback(
    (runId, status) => {
      rememberRun(runId, { status });
      if (status === "completed") setView({ name: "results", runId });
    },
    [rememberRun]
  );

  const goHome = useCallback(() => setView({ name: "home" }), []);

  return (
    <>
      <header className="topbar">
        <div className="brand" onClick={goHome} title="Home">
          <div className="brand-mark">🚀</div>
          <div className="brand-name">
            Marketing<span>OS</span>
          </div>
        </div>
        <div className="topbar-status">AI Campaign Studio</div>
      </header>

      {view.name === "home" && (
        <CreateRun
          recentRuns={recentRuns}
          onRunStarted={startRun}
          onOpenRun={openResults}
        />
      )}

      {view.name === "running" && (
        <RunProgress runId={view.runId} onFinished={finishRun} onBack={goHome} />
      )}

      {view.name === "results" && (
        <Results runId={view.runId} onBack={goHome} onLabelKnown={rememberRun} />
      )}
    </>
  );
}
