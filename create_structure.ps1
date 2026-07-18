# MarketingOS — Project Structure Scaffold (Windows PowerShell)
# Usage: Open PowerShell in the parent folder where you want the project, then run:
#   .\create_structure.ps1
# If you get a script-execution error, first run (as that user):
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

$root = "marketingos"

$dirs = @(
    # top-level
    "$root/configs/environments",

    # api
    "$root/src/marketingos/api/routes",
    "$root/src/marketingos/api/schemas",

    # orchestration
    "$root/src/marketingos/orchestration/nodes",

    # agents
    "$root/src/marketingos/agents/research",
    "$root/src/marketingos/agents/synthetic_source",
    "$root/src/marketingos/agents/business_analysis",
    "$root/src/marketingos/agents/strategist",
    "$root/src/marketingos/agents/planner",
    "$root/src/marketingos/agents/copywriter",
    "$root/src/marketingos/agents/designer",
    "$root/src/marketingos/agents/video_director",
    "$root/src/marketingos/agents/qa",
    "$root/src/marketingos/agents/packaging",

    # memory
    "$root/src/marketingos/memory/backends",

    # prompts
    "$root/src/marketingos/prompts/templates/research/v1",
    "$root/src/marketingos/prompts/templates/synthetic_source/v1",
    "$root/src/marketingos/prompts/templates/business_analysis/v1",
    "$root/src/marketingos/prompts/templates/strategist/v1",
    "$root/src/marketingos/prompts/templates/planner/v1",
    "$root/src/marketingos/prompts/templates/copywriter/v1",
    "$root/src/marketingos/prompts/templates/designer/v1",
    "$root/src/marketingos/prompts/templates/video_director/v1",
    "$root/src/marketingos/prompts/templates/qa/v1",
    "$root/src/marketingos/prompts/templates/packaging/v1",
    "$root/src/marketingos/prompts/policies",

    # tools
    "$root/src/marketingos/tools/web",
    "$root/src/marketingos/tools/llm",
    "$root/src/marketingos/tools/image",
    "$root/src/marketingos/tools/video",

    # models
    "$root/src/marketingos/models",

    # services
    "$root/src/marketingos/services",

    # observability
    "$root/src/marketingos/observability",

    # config
    "$root/src/marketingos/config",

    # ui
    "$root/src/marketingos/ui/dashboard",

    # data
    "$root/data/runs",
    "$root/data/cache",

    # tests
    "$root/tests/unit",
    "$root/tests/contract",
    "$root/tests/integration",
    "$root/tests/golden",

    # scripts
    "$root/scripts",

    # docs
    "$root/docs/architecture/diagrams",
    "$root/docs/architecture/adr"
)

$files = @(
    "$root/pyproject.toml",
    "$root/README.md",
    "$root/.env.example",

    "$root/configs/base.yaml",
    "$root/configs/agents.yaml",
    "$root/configs/models.yaml",
    "$root/configs/budget.yaml",
    "$root/configs/workflow.yaml",
    "$root/configs/environments/dev.yaml",
    "$root/configs/environments/prod.yaml",

    "$root/src/marketingos/__init__.py",

    "$root/src/marketingos/api/__init__.py",
    "$root/src/marketingos/api/main.py",
    "$root/src/marketingos/api/routes/__init__.py",
    "$root/src/marketingos/api/routes/runs.py",
    "$root/src/marketingos/api/routes/outputs.py",
    "$root/src/marketingos/api/routes/health.py",
    "$root/src/marketingos/api/schemas/__init__.py",
    "$root/src/marketingos/api/dependencies.py",

    "$root/src/marketingos/orchestration/__init__.py",
    "$root/src/marketingos/orchestration/graph.py",
    "$root/src/marketingos/orchestration/state.py",
    "$root/src/marketingos/orchestration/nodes/__init__.py",
    "$root/src/marketingos/orchestration/edges.py",
    "$root/src/marketingos/orchestration/checkpointer.py",
    "$root/src/marketingos/orchestration/approval_gates.py",

    "$root/src/marketingos/agents/__init__.py",
    "$root/src/marketingos/agents/base.py",

    "$root/src/marketingos/memory/__init__.py",
    "$root/src/marketingos/memory/store.py",
    "$root/src/marketingos/memory/backends/__init__.py",
    "$root/src/marketingos/memory/backends/sqlite_backend.py",
    "$root/src/marketingos/memory/backends/vector_backend.py",
    "$root/src/marketingos/memory/schemas.py",

    "$root/src/marketingos/prompts/__init__.py",
    "$root/src/marketingos/prompts/registry.py",

    "$root/src/marketingos/tools/__init__.py",
    "$root/src/marketingos/tools/base.py",
    "$root/src/marketingos/tools/web/__init__.py",
    "$root/src/marketingos/tools/web/website_scraper.py",
    "$root/src/marketingos/tools/web/instagram_public_reader.py",
    "$root/src/marketingos/tools/llm/__init__.py",
    "$root/src/marketingos/tools/llm/openai_client.py",
    "$root/src/marketingos/tools/llm/ollama_client.py",
    "$root/src/marketingos/tools/image/__init__.py",
    "$root/src/marketingos/tools/image/image_gen_client.py",
    "$root/src/marketingos/tools/image/compositor.py",
    "$root/src/marketingos/tools/video/__init__.py",
    "$root/src/marketingos/tools/video/video_assembler.py",
    "$root/src/marketingos/tools/registry.py",

    "$root/src/marketingos/models/__init__.py",
    "$root/src/marketingos/models/business_context.py",
    "$root/src/marketingos/models/plan.py",
    "$root/src/marketingos/models/creative.py",
    "$root/src/marketingos/models/run.py",
    "$root/src/marketingos/models/cost.py",

    "$root/src/marketingos/services/__init__.py",
    "$root/src/marketingos/services/cost_ledger.py",
    "$root/src/marketingos/services/cost_guard.py",
    "$root/src/marketingos/services/packaging_service.py",
    "$root/src/marketingos/services/run_manager.py",
    "$root/src/marketingos/services/approval_service.py",

    "$root/src/marketingos/observability/__init__.py",
    "$root/src/marketingos/observability/logging_config.py",
    "$root/src/marketingos/observability/tracing.py",
    "$root/src/marketingos/observability/metrics.py",

    "$root/src/marketingos/config/__init__.py",
    "$root/src/marketingos/config/settings.py",
    "$root/src/marketingos/config/loader.py",

    "$root/src/marketingos/ui/__init__.py",

    "$root/docs/architecture/vision.md",
    "$root/docs/architecture/nfr.md",
    "$root/docs/roadmap.md"
)

Write-Host "Creating directories..." -ForegroundColor Cyan
foreach ($d in $dirs) {
    New-Item -ItemType Directory -Path $d -Force | Out-Null
}

Write-Host "Creating placeholder files..." -ForegroundColor Cyan
foreach ($f in $files) {
    if (-not (Test-Path $f)) {
        New-Item -ItemType File -Path $f -Force | Out-Null
    }
}

# .gitkeep for empty runtime dirs so git tracks them
$gitkeeps = @(
    "$root/data/runs/.gitkeep",
    "$root/data/cache/.gitkeep"
)
foreach ($g in $gitkeeps) {
    New-Item -ItemType File -Path $g -Force | Out-Null
}

Write-Host "`nDone. Project scaffold created at .\$root" -ForegroundColor Green
Write-Host "Next: cd $root, then set up pyproject.toml + a virtual env (uv or Poetry)." -ForegroundColor Yellow