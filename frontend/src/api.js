// API client for the MarketingOS FastAPI backend.
//
// Locally the Vite dev server proxies /api/* to http://localhost:8000
// (see vite.config.js), so no CORS setup is needed. In production set
// VITE_API_URL to the deployed backend's origin and requests go direct.
const BASE = import.meta.env.VITE_API_URL || "/api";

async function request(path, options = {}) {
  const response = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body — keep the status text */
    }
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

export function createRun({ websiteUrl, businessName, instagramUsername, budgetUsd }) {
  const body = { website_url: websiteUrl, budget_usd: budgetUsd };
  if (businessName) body.business_name = businessName;
  if (instagramUsername) body.instagram_username = instagramUsername;
  return request("/runs", { method: "POST", body: JSON.stringify(body) });
}

export function getRunStatus(runId) {
  return request(`/runs/${runId}`);
}

export function getRunPackage(runId) {
  return request(`/runs/${runId}/package`);
}

export function getRunDocument(runId, packagedPath) {
  return request(`/runs/${runId}/assets/${packagedPath}`);
}

export function assetUrl(runId, packagedPath) {
  return `${BASE}/runs/${runId}/assets/${packagedPath}`;
}

export function archiveUrl(runId) {
  return `${BASE}/runs/${runId}/archive`;
}
