# How to Reproduce a MarketingOS Campaign

A short guide to running the pipeline yourself and finding every output it
produces. Written for someone running the project for the first time.

---

## What you need

- **Python 3.11+** with the project's virtual environment (`.venv` at the repo
  root, already set up).
- **Node.js 18+** (only if you want the web UI; the API works without it).
- **A Google Gemini API key** (free tier is fine). Get one at
  <https://aistudio.google.com/apikey>.

---

## Step 1 — Put your API key in `.env`

Create/edit **`marketingos/.env`** (copy from `marketingos/.env.example`):

```
GEMINI_API_KEY=your-key-here
```

The backend loads this file automatically on startup — you do **not** need to
export anything in your shell.

---

## Step 2 — Confirm the model in the config

Open `marketingos/config/models.yaml` and `marketingos/config/agents.yaml`.
The model should be one your account can actually call. This project uses:

```yaml
default_llm: gemini-2.0-flash
```

> If you ever get a **403 "project denied access"**, it almost always means the
> configured model name is not available to your account — not that your key is
> bad. List the models your key can use and pick one that appears there:
> `GET https://generativelanguage.googleapis.com/v1beta/models?key=YOUR_KEY`

Image and video are set to **placeholder** providers (`image_provider:
placeholder`, `video_provider: placeholder`) so generation costs **₹0**. Swap
these to real providers (e.g. `flux_schnell`) in the same YAML if you want real
media — no code changes needed.

---

## Step 3 — Start the backend

From the `marketingos/` folder:

```powershell
cd C:\Users\Admin\MarketingOS\marketingos
..\.venv\Scripts\python.exe -m uvicorn marketingos.api.main:app --reload --port 8000
```

You should see `Application startup complete`. The API is now at
<http://localhost:8000> (health check: <http://localhost:8000/health>).

> **Note:** `--reload` watches `.py` files only. If you edit a `.yaml` config or
> the `.env`, **restart** uvicorn (Ctrl+C, then relaunch) to pick up the change.

---

## Step 4 — Start a campaign

**Option A — Web UI (recommended):**

```powershell
cd C:\Users\Admin\MarketingOS\frontend
npm install   # first time only
npm run dev
```

Open <http://localhost:5173>, enter a website URL (e.g. `https://www.nike.in/`),
and submit. The progress screen animates the 10 stages; when it finishes it
shows the results dashboard.

**Option B — API directly:**

```powershell
curl -X POST http://localhost:8000/runs `
  -H "Content-Type: application/json" `
  -d '{"website_url": "https://www.nike.in/"}'
```

The response includes a `run_id`. Poll status with:

```powershell
curl http://localhost:8000/runs/<run_id>
```

---

## Step 5 — Where each output is stored

Everything for a run lives under **`marketingos/data/runs/<run-id>/`**:

| Output | Location |
|--------|----------|
| Run status, budget, timeline | `run.json` |
| **Per-stage results + quality scores** (the main content) | `eval/<AgentName>_eval.json` |
| Synthetic source pack | `eval/SyntheticSourceAgent_eval.json` |
| Business-context analysis | `eval/BusinessAnalysisAgent_eval.json` |
| Content strategy | `eval/StrategistAgent_eval.json` |
| Seven-day plan (5 posts + 2 videos) | `eval/PlannerAgent_eval.json` |
| Captions | `eval/CopywriterAgent_eval.json` |
| Image prompts / video shot plans | `eval/DesignerAgent_eval.json`, `eval/VideoDirectorAgent_eval.json` |
| Quality report | `05_qa/qa_report.json` |
| **Spend log** (proves cost ≤ ₹100) | `06_cost/cost_ledger.json` |
| Final rendered images & videos | `04_creatives/posts/`, `04_creatives/videos/` |
| Final packaged zip | `package/` |

Each `eval/*.json` file has three useful parts:
- **`score`** — a 0–100 quality score for that stage.
- **`validation`** — whether the output passed structural checks.
- **`artifacts`** — the actual generated content (as JSON).

---

## Troubleshooting

| Symptom | Cause & fix |
|---------|-------------|
| **403 "project denied access"** | Configured model isn't available to your account. List your models (Step 2) and set an available one. |
| **429 "quota exceeded"** | Gemini free-tier per-minute limit. Wait ~60 seconds and retry; the system also retries automatically with backoff. |
| **Run fails at packaging** | Older builds hard-blocked on failed quality checks. This is now advisory — pull the latest code. |
| **`.env` changes ignored** | Restart uvicorn — `--reload` doesn't watch `.env`. |
| **Empty `04_creatives` / `package`** | The run didn't reach packaging, or image/video are in placeholder mode. See the run's `run.json` `status`/`error`. |

---

## A note on cost

With **Gemini free tier** for text and **placeholder providers** for image/video,
a full run costs **₹0**. If you switch to paid image/video providers, the
`cost_ledger.json` spend log tracks every charge in INR and the built-in cost
guard stops the run before it exceeds the budget set in
`config/budget.yaml` (default ₹100).
