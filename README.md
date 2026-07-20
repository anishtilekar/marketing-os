# MarketingOS

**One website URL in — a packaged first-week marketing campaign out.**

MarketingOS is a multi-agent AI system that researches a business from its public
website and produces a complete first-week social campaign: business analysis,
content strategy, a seven-day plan (5 posts + 2 short videos), finished captions,
creative briefs, a QA audit, and a spend ledger — all delivered as one bundle
through a FastAPI backend and a React dashboard.

Built with **Python 3.12 · FastAPI · LangGraph · Pydantic v2 · Gemini** on the
backend and **React 18 + Vite** on the frontend. MIT licensed.

---

## The pipeline

Ten specialist agents run in sequence, each with validated input/output schemas
and its own 0–100 evaluation score:

| # | Agent | What it does |
|---|-------|--------------|
| 1 | **Research** | Scrapes the public website (headings, JSON-LD, metadata) into sourced facts |
| 2 | **Synthetic Source** | Distills research into a reusable, no-private-data source pack |
| 3 | **Business Analysis** | Separates observed facts from labeled assumptions; finds opportunities and gaps |
| 4 | **Strategist** | Goals, audience, positioning, content pillars, key messages, success metrics |
| 5 | **Planner** | Seven-day plan: exactly 5 posts + 2 short-form videos across platforms |
| 6 | **Copywriter** | Headline + caption + hashtags for every plan item |
| 7 | **Designer** | Image prompts and creative direction per post |
| 8 | **Video Director** | Shot-by-shot scripts for the two videos |
| 9 | **QA** | Brand-safety and platform-fit audit (advisory — findings ship with the package) |
| 10 | **Packaging** | Bundles everything into a downloadable archive with a README |

## Design highlights

- **Config-driven providers** — LLM, image, and video backends are chosen in
  [`marketingos/config/models.yaml`](marketingos/config/models.yaml); swapping
  `placeholder` → `flux_schnell` (Together AI) or `gemini` is a one-line config
  edit, never a code change.
- **Evaluation framework** — every agent run persists validation results, a
  scored report, and its full artifacts under `data/runs/<run-id>/eval/`.
- **Cost guard** — every tool call is priced against a hard budget ceiling
  (default ₹100, [`config/budget.yaml`](marketingos/config/budget.yaml)); the
  spend ledger proves what a run cost.
- **Resilient LLM client** — automatic retry with exponential backoff on
  429/5xx, honoring the provider's suggested retry delay.
- **Zero-cost demo profile** — placeholder image/video providers plus Gemini's
  free tier mean a full run costs ₹0 while developing.

## Repository layout

```
├── marketingos/          # Python backend (installable package)
│   ├── src/marketingos/  #   agents, orchestration, tools, evaluation, API
│   ├── config/           #   models.yaml, agents.yaml, budget.yaml
│   ├── tests/            #   unit tests (pytest)
│   └── docs/             #   campaign output report + reproduction guide
└── frontend/             # React 18 + Vite dashboard
```

## Quickstart

### 1. Backend

Requires Python **3.12+**.

```bash
cd marketingos
python -m venv .venv
.venv\Scripts\activate            # Windows — use `source .venv/bin/activate` on macOS/Linux
pip install -e ".[dev]"

cp .env.example .env              # then put your GEMINI_API_KEY in .env
uvicorn marketingos.api.main:app --reload --port 8000
```

The API loads `marketingos/.env` automatically at startup — no shell exports
needed. Get a free Gemini key at <https://aistudio.google.com/apikey>.

### 2. Frontend

Requires Node.js **18+**.

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>, paste a website URL, and launch a campaign. The
dev server proxies `/api` to the backend on port 8000.

### 3. Or use the API directly

```bash
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"website_url": "https://example.com/"}'
```

## API

| Method | Route | Purpose |
|--------|-------|---------|
| `POST` | `/runs` | Start a campaign run |
| `GET` | `/runs/{id}` | Poll run status |
| `GET` | `/runs/{id}/package` | Package manifest + asset index |
| `GET` | `/runs/{id}/assets/{path}` | Serve a generated image/video/document |
| `GET` | `/runs/{id}/archive` | Download the full campaign zip |
| `GET` | `/health` | Liveness probe |

## Where outputs land

Everything for a run is written under `marketingos/data/runs/<run-id>/`:
per-agent results and scores in `eval/`, the QA report in `05_qa/`, the spend
ledger in `06_cost/`, rendered media in `04_creatives/`, and the final bundle in
`package/`.

## Testing

```bash
cd marketingos
pytest tests            # unit suite — fully mocked, no API calls, no cost
ruff check src tests    # lint
mypy src                # types
```

## Documentation

- [Campaign output report](marketingos/docs/CAMPAIGN_OUTPUT.md) — a real run
  against nike.in, with every deliverable and where it's stored
- [Reproduction guide](marketingos/docs/REPRODUCE_README.md) — step-by-step
  setup, output map, troubleshooting

## License

MIT
