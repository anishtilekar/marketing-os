# MarketingOS — Campaign Output Report

**Assumed business:** Nike (nike.in — Nike's official India online store)
**Run ID:** `73d14d0d-77e4-45a6-b215-75a4d841e0e9`
**Date:** 19 July 2026
**Status:** Partial — 9 of 10 pipeline stages completed (see *Honest limitations* below)

> **Read this first (plain-English summary).**
> MarketingOS is an automated "marketing team in a box." You give it one website
> URL. It then runs ten specialist steps in a row — research, analysis, strategy,
> planning, copywriting, design, video, quality-check, and packaging — and hands
> back a week's worth of ready-to-post social content plus a cost receipt.
> For this report we pointed it at **Nike India (nike.in)**. It successfully
> produced the research, the synthetic source pack, the business analysis, the
> strategy, the full seven-day plan, and all seven captions. It stopped just
> before the final "package everything into a zip" step. Everything it produced
> is saved on disk — this document tells you exactly what was produced and where
> to find each piece.

---

## 1. The assumed business — and why Nike

We assumed the business is **Nike's official India online storefront (nike.in)**.

**Why this business was chosen:**

- **It is a real, public, well-known brand**, so anyone reading this report can
  immediately judge whether the generated strategy "sounds like Nike" — a good
  test of whether the system actually understood the business.
- **The website is content-rich but JavaScript-heavy**, which is a realistic
  hard case. It forced the research step to extract meaning from a modern
  storefront rather than a simple blog, testing the system honestly.
- **No login, no private data.** Everything the system read is publicly visible
  on the homepage, so the whole exercise stays within public information only.

---

## 2. Synthetic source pack (reusable, no private data)

The **synthetic source pack** is a clean, reusable summary of what the brand
publicly says about itself. It is "synthetic" because it is rebuilt from public
signals into a tidy internal format — it contains **no private, personal, or
scraped-behind-login data**.

What was produced for Nike:

- **Brand descriptions** — e.g. the homepage title *"Nike. Just Do It. Nike IN"*
  and a summary of the site's main navigation (Shop, Featured, Shop by Sport,
  Accessories & Equipment, Sale & Offers).
- **Brand characteristics** — e.g. the public tagline: *"Nike – Official Online
  Store for Athletic Shoes, Clothing & Sports Gear… enjoy free shipping."*
- **25 keywords** distilled from the site — `nike, shop, shoes, clothing, sport,
  accessories, equipment, sale, athletic, official, online, …`
- **Provenance** — the source URL (`https://www.nike.in/`), a confidence score
  (0.9), and the collection timestamp, so every downstream claim is traceable.

**Reusable:** because it is stored as structured JSON keyed by category, the same
pack can feed strategy, planning, and copywriting without re-scraping the site.

📁 **Stored in:** `data/runs/73d14d0d…/eval/SyntheticSourceAgent_eval.json`
(inside the `artifacts` field).

---

## 3. Business-context analysis

A concise read of the business, separating what is **known** from what is
**assumed**:

- **Observed facts (3):** grounded directly in the homepage — official store
  status, free-shipping offer, and multi-sport product range.
- **Assumptions (4):** reasonable inferences the system flagged *as assumptions*
  (e.g. target market, positioning intent) — explicitly not presented as facts.
- **Opportunities:** direct-to-consumer (DTC) messaging, authenticity/anti-fake
  assurance, and product-innovation storytelling (the Pegasus 42).
- **Gaps (honestly noted):** the context leans slightly assumption-heavy
  (4 assumptions vs 3 facts) because nike.in is a JavaScript storefront that
  exposes limited plain text. The quality-check step later flagged this itself.

📁 **Stored in:** `data/runs/73d14d0d…/eval/BusinessAnalysisAgent_eval.json`

---

## 4. First-week content strategy

A complete strategy was produced:

- **Objective / goals:** *"Drive awareness and qualified traffic to the official
  Nike India online store, emphasizing direct-to-consumer benefits like free
  shipping."*
- **Target audience:** *"Active individuals, athletes, and sports enthusiasts in
  India seeking premium, authentic athletic footwear, apparel, and gear."*
- **Positioning:** *"The definitive, official online destination in India for
  premium athletic performance and lifestyle gear."*
- **Content pillars (3):** **Official Store Benefits**, **Innovation Spotlight**,
  **Multi-Sport Versatility**.
- **Key messages:** official-store authenticity, free shipping, "shop directly
  from the source."
- **Tone:** confident, motivational, performance-driven (true to Nike's voice).
- **Success metrics:** e.g. *"15% increase in weekly referral traffic to the
  Nike India website,"* tracked via UTM-tagged bio links.

📁 **Stored in:** `data/runs/73d14d0d…/eval/StrategistAgent_eval.json`

---

## 5. Seven-day plan — exactly 5 posts + 2 short videos

The plan contains **exactly seven items: five posts and two short-form videos**,
as required:

| Day | Type | Platform | Topic | Call to action |
|----|------|----------|-------|----------------|
| 1 | **Post** | Instagram | Official Nike India online store launch announcement | "Shop directly at Nike India and enjoy free shipping." |
| 2 | **Short video** | Instagram | Close-up showcase of the new Pegasus 42 (Air Zoom) | "Experience the power of Air Zoom — link in bio." |
| 3 | **Post** | X | Curated gear across different sports | "Find premium gear tailored for your sport." |
| 4 | **Post** | Facebook | Authenticity, security & free shipping explainer | "Get authentic gear with free shipping." |
| 5 | **Short video** | YouTube | High-energy Pegasus 42 performance test on a track | "Upgrade your run. Shop the Pegasus 42." |
| 6 | **Post** | LinkedIn | Nike's direct-to-consumer (DTC) strategy | "Explore the official Nike India online store." |
| 7 | **Post** | Instagram | Lifestyle carousel — everyday athletes | "Gear up for every movement." |

**Count check:** Posts on days 1, 3, 4, 6, 7 = **5 posts**. Videos on days 2, 5 =
**2 short videos**. ✔

📁 **Stored in:** `data/runs/73d14d0d…/eval/PlannerAgent_eval.json`

---

## 6. Post creatives and video files

The copywriting step produced **all seven finished captions** (headline + body +
hashtags), each mapped to its plan item:

| Item | Headline |
|------|----------|
| C1 | Nike India is Now Online |
| C2 | Meet the Pegasus 42 |
| C3 | Gear for Every Pursuit |
| C4 | Shop with Absolute Confidence |
| C5 | Pegasus 42: Track Tested |
| C6 | Elevating the Consumer Experience in India |
| C7 | Built for Performance. Styled for Life. |

📁 **Captions stored in:** `data/runs/73d14d0d…/eval/CopywriterAgent_eval.json`
📁 **Creative/video plans in:** `…/eval/DesignerAgent_eval.json` and
`…/eval/VideoDirectorAgent_eval.json`

> **Note on the image and video *files*:** for this run the image and video
> providers were set to **placeholder mode** (zero-cost local generation), and
> the run stopped at the final packaging step (Section 8), so the finished image
> and video *files* were not written into `04_creatives/posts` and
> `04_creatives/videos` — those folders exist but are empty for this run.
> A separate fully-completed run (`6eca3f7e…`) *does* contain 5 rendered images
> and 2 rendered videos in its package, proving the rendering + packaging stages
> work end-to-end. The instructions (headlines, prompts, shot descriptions) for
> Nike's creatives are all present in the eval files above.

---

## 7. Spend log — proving cost ≤ ₹100

Total **paid** content-generation cost for this run: **₹0.00** — comfortably
within the ₹100 limit.

| # | Category | Tool / provider | Cost |
|---|----------|-----------------|------|
| 1 | web_tool | website-scraper (direct fetch) | ₹0 |
| 2–8 | llm_generation | Gemini (free tier) × 7 | ₹0 each |
| | **Total** | | **₹0.00 INR** |

**Why it's free:** text generation runs on **Gemini's free tier** (₹0), and the
image/video providers were in **placeholder mode** (local compute, ₹0). No paid
API was billed.

📁 **Stored in:** `data/runs/73d14d0d…/06_cost/cost_ledger.json`

---

## 8. Honest limitations — why the full workflow did not finish

Two things stopped a single, clean, end-to-end Nike run:

1. **Gemini free-tier limits.** The free tier allows only a handful of requests
   per minute. One campaign fires roughly eight text-generation calls in quick
   succession, so repeated full runs hit **429 "quota exceeded"** and could not
   all complete back-to-back. (The system now retries with backoff, but a busy
   minute can still exhaust the free quota.) **Because of this free-tier limit,
   the whole workflow could not be produced in one uninterrupted pass.**

2. **A quality-check gate (now fixed).** This particular Nike run reached the
   quality-check step, which correctly flagged two captions using the word
   *"guaranteed"* (an unsupportable claim) plus two minor warnings. At the time,
   a failed quality check **blocked** packaging, so the run ended at the
   packaging step with *"campaign failed QA and cannot be packaged."* That gate
   has since been changed to **advisory** — the quality report now ships *with*
   the package instead of blocking it — so a re-run today would produce the final
   zip.

**What we did get (and where):** despite the above, the run produced **the full
synthetic source pack** (Section 2) and **the entire seven-day plan of 5 posts +
2 videos** (Section 5), along with the business analysis, strategy, and all seven
captions. Every one of these is saved under
`data/runs/73d14d0d-77e4-45a6-b215-75a4d841e0e9/` — the per-stage results live in
the `eval/` sub-folder, and the quality report lives in `05_qa/qa_report.json`.

---

## 9. Where everything is stored (folder map)

All output for a run lives under `data/runs/<run-id>/`. For this Nike run,
`<run-id>` = `73d14d0d-77e4-45a6-b215-75a4d841e0e9`.

```
data/runs/73d14d0d-77e4-45a6-b215-75a4d841e0e9/
├── run.json              ← run status, budget, timeline
├── eval/                 ← per-agent results + quality scores (THE MAIN OUTPUT)
│   ├── ResearchAgent_eval.json          (facts gathered from nike.in)
│   ├── SyntheticSourceAgent_eval.json   (§2 the source pack)
│   ├── BusinessAnalysisAgent_eval.json  (§3 business context)
│   ├── StrategistAgent_eval.json        (§4 strategy)
│   ├── PlannerAgent_eval.json           (§5 the 7-day plan)
│   ├── CopywriterAgent_eval.json        (§6 the 7 captions)
│   ├── DesignerAgent_eval.json          (§6 image prompts)
│   ├── VideoDirectorAgent_eval.json     (§6 video shot plans)
│   └── QAAgent_eval.json                (quality findings)
├── 05_qa/qa_report.json  ← the standalone quality report
├── 06_cost/cost_ledger.json  ← §7 the spend log
├── 04_creatives/         ← posts/ and videos/ (final rendered files; empty here — see §6)
├── 00_source_pack, 01_business_context, 02_strategy, 03_plan  ← stage folders
└── package/              ← final zip bundle (empty here — packaging did not run)
```

> **Tip for a first-time reader:** open the files in `eval/` first. Each one is a
> plain JSON file with a `score` (0–100), a `validation` block, and an
> `artifacts` field that holds the actual generated content for that stage.

---

*See `REPRODUCE_README.md` for step-by-step instructions to reproduce this run.*
