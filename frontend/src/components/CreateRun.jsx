import { useState } from "react";
import { createRun } from "../api.js";

export default function CreateRun({ recentRuns, onRunStarted, onOpenRun }) {
  const [websiteUrl, setWebsiteUrl] = useState("");
  const [businessName, setBusinessName] = useState("");
  const [instagramUsername, setInstagramUsername] = useState("");
  const [budgetUsd, setBudgetUsd] = useState("10");
  const [openId, setOpenId] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  async function handleSubmit(event) {
    event.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const { run_id } = await createRun({
        websiteUrl: websiteUrl.trim(),
        businessName: businessName.trim(),
        instagramUsername: instagramUsername.trim().replace(/^@/, ""),
        budgetUsd: budgetUsd.trim() || "10",
      });
      onRunStarted(run_id);
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main>
      <section className="hero reveal">
        <div className="hero-badge">
          <span className="pulse-dot" /> 10 AI agents · one campaign
        </div>
        <h1>
          A week of marketing,
          <br />
          <span className="grad-text">generated in minutes.</span>
        </h1>
        <p className="hero-sub">
          Point MarketingOS at a business. Ten specialised agents research it,
          craft the strategy, plan the week, write every caption, and design
          every creative — packaged and ready to publish.
        </p>
      </section>

      <form className="run-form glass reveal reveal-2" onSubmit={handleSubmit}>
        <div className="form-grid">
          <div className="field full">
            <label>
              Website URL <span className="req">*</span>
            </label>
            <input
              type="url"
              required
              placeholder="https://your-business.com"
              value={websiteUrl}
              onChange={(e) => setWebsiteUrl(e.target.value)}
            />
          </div>
          <div className="field">
            <label>Business name</label>
            <input
              type="text"
              placeholder="Acme Coffee Roasters"
              value={businessName}
              onChange={(e) => setBusinessName(e.target.value)}
            />
          </div>
          <div className="field">
            <label>Instagram username</label>
            <input
              type="text"
              placeholder="@acmecoffee"
              value={instagramUsername}
              onChange={(e) => setInstagramUsername(e.target.value)}
            />
          </div>
          <div className="field">
            <label>Budget (USD)</label>
            <input
              type="number"
              min="0.01"
              step="0.01"
              value={budgetUsd}
              onChange={(e) => setBudgetUsd(e.target.value)}
            />
          </div>
        </div>

        <div className="form-actions">
          <button className="btn btn-primary" type="submit" disabled={submitting}>
            {submitting ? "Launching agents…" : "✨ Generate campaign"}
          </button>
        </div>

        {error && <div className="form-error">{error}</div>}
      </form>

      <section className="recent glass reveal reveal-3">
        <div className="recent-title">Open a campaign</div>

        {recentRuns.length > 0 && (
          <div className="recent-list">
            {recentRuns.map((run) => (
              <div
                key={run.runId}
                className="recent-chip"
                onClick={() =>
                  run.status === "running"
                    ? onRunStarted(run.runId)
                    : onOpenRun(run.runId)
                }
              >
                <span className="rid">
                  {run.subject ? `${run.subject} · ` : ""}
                  {run.runId}
                </span>
                <span
                  className={`badge ${
                    run.status === "completed"
                      ? "badge-green"
                      : run.status === "failed"
                        ? "badge-red"
                        : "badge-dim"
                  }`}
                >
                  {run.status ?? "open"}
                </span>
              </div>
            ))}
          </div>
        )}

        <div className="open-by-id">
          <input
            placeholder="Paste a run id…"
            value={openId}
            onChange={(e) => setOpenId(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && openId.trim()) {
                e.preventDefault();
                onOpenRun(openId.trim());
              }
            }}
          />
          <button
            type="button"
            className="btn btn-ghost"
            disabled={!openId.trim()}
            onClick={() => onOpenRun(openId.trim())}
          >
            Open
          </button>
        </div>
      </section>
    </main>
  );
}
