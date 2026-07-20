import { useEffect, useMemo, useState } from "react";
import { archiveUrl, assetUrl, getRunDocument, getRunPackage } from "../api.js";

// Pull a human-readable line out of a value that may be a plain string or a
// structured object (strategy/QA docs vary per field).
function textOf(value) {
  if (value == null) return "";
  if (typeof value === "string") return value;
  if (typeof value !== "object") return String(value);
  for (const key of ["statement", "message", "description", "text", "name", "metric", "value"]) {
    if (typeof value[key] === "string" && value[key]) return value[key];
  }
  return JSON.stringify(value);
}

function listOf(value) {
  if (Array.isArray(value)) return value;
  if (value == null) return [];
  return [value];
}

function qaBadgeClass(status) {
  if (!status) return "badge-dim";
  if (status.startsWith("passed_with")) return "badge-amber";
  if (status.startsWith("passed")) return "badge-green";
  return "badge-red";
}

function platformBadgeClass(platform) {
  const p = (platform ?? "").toLowerCase();
  if (p.includes("insta")) return "badge-violet";
  if (p.includes("linked")) return "badge-cyan";
  return "badge-dim";
}

function findDocPath(assetIndex, suffix) {
  return assetIndex.find(
    (e) => e.kind === "document" && e.packaged_path.endsWith(suffix)
  )?.packaged_path;
}

function ContentCard({ item, caption, media, runId, index }) {
  const [expanded, setExpanded] = useState(false);
  const hashtags = listOf(caption?.hashtags).map(textOf).filter(Boolean);

  return (
    <article className={`content-card glass card-hover reveal reveal-${Math.min(index + 1, 7)}`}>
      <div className="content-media">
        {media?.kind === "video" ? (
          <video controls preload="metadata" src={assetUrl(runId, media.packaged_path)} />
        ) : media ? (
          <img src={assetUrl(runId, media.packaged_path)} alt={caption?.headline ?? item.topic} />
        ) : null}
        <span className="day-chip">Day {item.day}</span>
        <span className={`fmt-chip badge ${String(item.format).includes("video") ? "badge-violet" : "badge-cyan"}`}>
          {String(item.format).includes("video") ? "🎬 video" : "🖼️ post"}
        </span>
      </div>

      <div className="content-body">
        <div className="badges-row">
          <span className={`badge ${platformBadgeClass(item.platform)}`}>{item.platform}</span>
          {item.publish_time && <span className="badge badge-dim">🕘 {String(item.publish_time).slice(0, 5)}</span>}
        </div>

        {caption?.headline && <h3 className="content-headline">{caption.headline}</h3>}

        {caption?.caption && (
          <>
            <p className={`content-caption${expanded ? " expanded" : ""}`}>{caption.caption}</p>
            <button className="caption-toggle" onClick={() => setExpanded((v) => !v)}>
              {expanded ? "Show less ▲" : "Read full caption ▼"}
            </button>
          </>
        )}

        {hashtags.length > 0 && (
          <div className="hashtags">
            {hashtags.map((tag) => (
              <span key={tag} className="hashtag">
                {tag.startsWith("#") ? tag : `#${tag}`}
              </span>
            ))}
          </div>
        )}

        <div className="content-detail">
          <span><b>Topic</b> — {item.topic}</span>
          {item.call_to_action && <span><b>CTA</b> — {item.call_to_action}</span>}
        </div>
      </div>
    </article>
  );
}

export default function Results({ runId, onBack, onLabelKnown }) {
  const [pkg, setPkg] = useState(null);
  const [docs, setDocs] = useState({});
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const packageJson = await getRunPackage(runId);
        if (cancelled) return;
        setPkg(packageJson);
        onLabelKnown?.(runId, { subject: packageJson.subject, status: "completed" });

        const index = packageJson.asset_index ?? [];
        const wanted = {
          weekPlan: findDocPath(index, "content/week_plan.json"),
          captions: findDocPath(index, "content/captions.json"),
          strategy: findDocPath(index, "content/strategy.json"),
          qa: findDocPath(index, "qa_report.json"),
        };
        const loaded = {};
        await Promise.all(
          Object.entries(wanted).map(async ([key, path]) => {
            if (!path) return;
            try {
              loaded[key] = await getRunDocument(runId, path);
            } catch {
              /* a missing side-document shouldn't sink the dashboard */
            }
          })
        );
        if (!cancelled) setDocs(loaded);
      } catch (err) {
        if (!cancelled) setError(err);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [runId, onLabelKnown]);

  const mediaByItem = useMemo(() => {
    const map = {};
    for (const entry of pkg?.asset_index ?? []) {
      if (entry.item_id && (entry.kind === "image" || entry.kind === "video")) {
        map[entry.item_id] = entry;
      }
    }
    return map;
  }, [pkg]);

  const captionByItem = useMemo(() => {
    const map = {};
    for (const cap of docs.captions?.captions ?? []) map[cap.item_id] = cap;
    return map;
  }, [docs.captions]);

  const planItems = useMemo(
    () => [...(docs.weekPlan?.items ?? [])].sort((a, b) => a.day - b.day),
    [docs.weekPlan]
  );

  if (error) {
    return (
      <main>
        <div className="progress-wrap glass reveal">
          <div className="progress-head">
            <h2>Couldn't open this run</h2>
            <p>{error.message}</p>
            <span className="run-id-tag">{runId}</span>
          </div>
          {error.status === 409 && (
            <p style={{ textAlign: "center", color: "var(--text-dim)", fontSize: 14 }}>
              The run exists but hasn't completed — it may still be running, or it failed
              before packaging.
            </p>
          )}
          <button className="back-link" onClick={onBack}>← Back to home</button>
        </div>
      </main>
    );
  }

  if (!pkg) {
    return (
      <main>
        <div className="loading-view">
          <div className="spinner" />
          Opening campaign…
        </div>
      </main>
    );
  }

  const strategy = docs.strategy;
  const qa = docs.qa;
  const stats = [
    { num: pkg.metadata?.post_count ?? "–", lbl: "posts" },
    { num: pkg.metadata?.video_count ?? "–", lbl: "videos" },
    {
      num: (pkg.asset_index ?? []).filter((e) => e.kind !== "document").length,
      lbl: "assets",
    },
  ];

  return (
    <main>
      {/* Header band */}
      <section className="results-head glass reveal">
        <div className="results-head-left">
          <h1 className="grad-text">{pkg.subject}</h1>
          <div className="results-meta">
            <span className={`badge ${qaBadgeClass(pkg.metadata?.qa_status)}`}>
              QA {String(pkg.metadata?.qa_status ?? "unknown").replaceAll("_", " ")}
            </span>
            <span>
              {pkg.created_at ? new Date(pkg.created_at).toLocaleString() : ""}
            </span>
          </div>
        </div>
        <div className="stat-tiles">
          {stats.map((s) => (
            <div key={s.lbl} className="stat-tile">
              <div className="num">{s.num}</div>
              <div className="lbl">{s.lbl}</div>
            </div>
          ))}
        </div>
        <a className="btn btn-primary" href={archiveUrl(runId)} download>
          ⬇️ Download campaign.zip
        </a>
      </section>

      {/* Content calendar */}
      {planItems.length > 0 && (
        <section className="section">
          <h2 className="section-title">
            <span className="dot" /> The week's content
          </h2>
          <div className="calendar-grid">
            {planItems.map((item, i) => (
              <ContentCard
                key={item.id}
                index={i}
                item={item}
                caption={captionByItem[item.id]}
                media={mediaByItem[item.id]}
                runId={runId}
              />
            ))}
          </div>
        </section>
      )}

      {/* Strategy */}
      {strategy && (
        <section className="section">
          <h2 className="section-title">
            <span className="dot" /> Strategy
          </h2>
          <div className="strategy-grid">
            {strategy.positioning && (
              <div className="strategy-card glass card-hover reveal">
                <h3><span className="ico">🧭</span> Positioning</h3>
                <p>{textOf(strategy.positioning)}</p>
              </div>
            )}
            {listOf(strategy.goals).length > 0 && (
              <div className="strategy-card glass card-hover reveal reveal-1">
                <h3><span className="ico">🎯</span> Goals</h3>
                <ul>
                  {listOf(strategy.goals).map((g, i) => (
                    <li key={i}>{textOf(g)}</li>
                  ))}
                </ul>
              </div>
            )}
            {strategy.target_audience && (
              <div className="strategy-card glass card-hover reveal reveal-2">
                <h3><span className="ico">👥</span> Target audience</h3>
                <p>{textOf(strategy.target_audience)}</p>
              </div>
            )}
            {listOf(strategy.key_messages).length > 0 && (
              <div className="strategy-card glass card-hover reveal reveal-3">
                <h3><span className="ico">💬</span> Key messages</h3>
                <ul>
                  {listOf(strategy.key_messages).map((m, i) => (
                    <li key={i}>{textOf(m)}</li>
                  ))}
                </ul>
              </div>
            )}
            {listOf(strategy.content_pillars).length > 0 && (
              <div className="strategy-card glass card-hover reveal reveal-4">
                <h3><span className="ico">🏛️</span> Content pillars</h3>
                <div className="pillars">
                  {listOf(strategy.content_pillars).map((p, i) => (
                    <span key={i} className="pillar">{textOf(p)}</span>
                  ))}
                </div>
              </div>
            )}
            {listOf(strategy.success_metrics).length > 0 && (
              <div className="strategy-card glass card-hover reveal reveal-5">
                <h3><span className="ico">📈</span> Success metrics</h3>
                <ul>
                  {listOf(strategy.success_metrics).map((m, i) => (
                    <li key={i}>{textOf(m)}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </section>
      )}

      {/* QA */}
      {qa && (
        <section className="section">
          <h2 className="section-title">
            <span className="dot" /> Quality audit
          </h2>
          <div className="qa-panel glass reveal">
            <div className="qa-row">
              <span className={`badge ${qaBadgeClass(qa.status)}`}>
                {String(qa.status ?? "").replaceAll("_", " ")}
              </span>
              {qa.model_reviewed != null && (
                <span className="badge badge-dim">
                  {qa.model_reviewed ? "🤖 model reviewed" : "rule checks only"}
                </span>
              )}
            </div>
            {listOf(qa.findings).length > 0 && (
              <div className="qa-findings">
                {listOf(qa.findings).map((f, i) => {
                  const severity = (f.severity ?? "").toLowerCase();
                  const icon =
                    severity === "error" ? "⛔" : severity === "warning" ? "⚠️" : "ℹ️";
                  return (
                    <div key={i} className="qa-finding">
                      <span className="sev">{icon}</span>
                      <span>
                        {f.code ? <b>{f.code}</b> : null}
                        {f.item_id ? ` [${f.item_id}] ` : " "}
                        {textOf(f.message ?? f)}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </section>
      )}

      <button className="back-link" onClick={onBack}>← Back to home</button>
    </main>
  );
}
