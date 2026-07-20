import { useEffect, useMemo, useRef, useState } from "react";
import { getRunStatus } from "../api.js";

// The 10 pipeline stages, with rough duration weights used only to animate
// an *estimated* progress indicator — the backend reports run-level status
// (running/completed/failed), not per-agent progress.
const STAGES = [
  { icon: "🔍", name: "Research", desc: "Scraping the website & socials", weight: 8 },
  { icon: "🧪", name: "Synthetic Source", desc: "Distilling source material", weight: 2 },
  { icon: "📊", name: "Business Analysis", desc: "Understanding the business", weight: 8 },
  { icon: "🎯", name: "Strategist", desc: "Positioning, goals & pillars", weight: 9 },
  { icon: "🗓️", name: "Planner", desc: "Planning the week of content", weight: 10 },
  { icon: "✍️", name: "Copywriter", desc: "Writing every caption", weight: 12 },
  { icon: "🎨", name: "Designer", desc: "Composing image creatives", weight: 8 },
  { icon: "🎬", name: "Video Director", desc: "Scripting short-form videos", weight: 12 },
  { icon: "✅", name: "Quality Audit", desc: "Auditing the full bundle", weight: 8 },
  { icon: "📦", name: "Packaging", desc: "Bundling the deliverables", weight: 4 },
];

const TOTAL_WEIGHT = STAGES.reduce((sum, s) => sum + s.weight, 0);
const ESTIMATED_TOTAL_SECONDS = 85;

export default function RunProgress({ runId, onFinished, onBack }) {
  const [status, setStatus] = useState("running");
  const [error, setError] = useState(null);
  const [elapsed, setElapsed] = useState(0);
  const startedAt = useRef(Date.now());
  const finishedRef = useRef(false);

  // Elapsed-time ticker driving the estimated stage indicator.
  useEffect(() => {
    const tick = setInterval(
      () => setElapsed((Date.now() - startedAt.current) / 1000),
      500
    );
    return () => clearInterval(tick);
  }, []);

  // Status polling.
  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const res = await getRunStatus(runId);
        if (cancelled) return;
        setStatus(res.status);
        setError(res.error ?? null);
        if (res.status !== "running" && !finishedRef.current) {
          finishedRef.current = true;
          // Brief pause so the final checkmark lands before transition.
          setTimeout(() => onFinished(runId, res.status), 900);
        }
      } catch (err) {
        if (!cancelled) setError(err.message);
      }
    }

    poll();
    const interval = setInterval(poll, 3000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [runId, onFinished]);

  // Map elapsed seconds onto an estimated active stage. Never claims the
  // final stage is done until the backend says the run finished.
  const activeIndex = useMemo(() => {
    if (status === "completed") return STAGES.length;
    const progress = Math.min(elapsed / ESTIMATED_TOTAL_SECONDS, 0.97);
    let cumulative = 0;
    for (let i = 0; i < STAGES.length; i += 1) {
      cumulative += STAGES[i].weight / TOTAL_WEIGHT;
      if (progress < cumulative) return i;
    }
    return STAGES.length - 1;
  }, [elapsed, status]);

  const failed = status === "failed";

  return (
    <main>
      <div className="progress-wrap glass reveal">
        <div className="progress-head">
          <h2>
            {failed ? (
              "Run failed"
            ) : status === "completed" ? (
              <span className="grad-text">Campaign ready!</span>
            ) : (
              <>
                Agents at work<span className="grad-text">…</span>
              </>
            )}
          </h2>
          <p>
            {failed
              ? "One of the agents hit an error — details below."
              : status === "completed"
                ? "Opening your campaign dashboard."
                : `Ten agents are building the campaign · ${Math.floor(elapsed)}s elapsed`}
          </p>
          <span className="run-id-tag">{runId}</span>
        </div>

        <div className="stages">
          {STAGES.map((stage, index) => {
            const done = index < activeIndex;
            const active = index === activeIndex && !failed && status === "running";
            return (
              <div
                key={stage.name}
                className={`stage${done ? " done" : ""}${active ? " active" : ""}`}
              >
                <div className="stage-node">{done ? "✓" : stage.icon}</div>
                <div className="stage-info">
                  <div className="stage-name">{stage.name}</div>
                  <div className="stage-desc">{stage.desc}</div>
                </div>
              </div>
            );
          })}
        </div>

        {!failed && status === "running" && (
          <div className="progress-note">
            Stage indicator is estimated from elapsed time — the pipeline
            reports overall status only.
          </div>
        )}

        {failed && (
          <div className="progress-error">
            <h3>Error</h3>
            <p>{error ?? "Unknown failure — check the backend logs."}</p>
          </div>
        )}

        <button className="back-link" onClick={onBack}>
          ← Back to home
        </button>
      </div>
    </main>
  );
}
