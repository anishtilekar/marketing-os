"""FastAPI entrypoint: create a run, poll its status, fetch its package."""

from __future__ import annotations

import asyncio
import os
from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langgraph.graph import END

from marketingos.api.dependencies import (
    RunAdapters,
    build_run_dependencies,
    run_manager,
)
from marketingos.api.routes.outputs import router as outputs_router
from marketingos.api.schemas import CreateRunRequest, RunStatusResponse
from marketingos.exceptions.workflow import WorkflowExecutionError
from marketingos.orchestration.graph import GraphBuilder
from marketingos.orchestration.nodes import (
    make_business_analysis_node,
    make_copywriter_node,
    make_creative_node,
    make_packaging_node,
    make_planner_node,
    make_qa_node,
    make_research_node,
    make_strategist_node,
    make_synthetic_resource_node,
    make_video_director_node,
)
from marketingos.orchestration.state import BudgetState, MarketingState
from marketingos.services.run_manager import RunHandle

#: Comma-separated frontend origins allowed to call this API (e.g. your
#: Vercel deployment's URL). Empty by default so a fresh deploy fails closed
#: on CORS rather than silently allowing every origin.
_ALLOWED_ORIGINS_ENV = "ALLOWED_ORIGINS"

app = FastAPI(title="MarketingOS API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.environ.get(_ALLOWED_ORIGINS_ENV, "").split(",")
        if origin.strip()
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(outputs_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness/readiness probe for the hosting platform."""
    return {"status": "ok"}

_NODE_ORDER = (
    "research",
    "synthetic",
    "business_analysis",
    "strategist",
    "planner",
    "copywriter",
    "creative",
    "video_director",
    "qa",
    "packaging",
)

#: Held so background run tasks are never garbage-collected mid-execution.
_background_tasks: set[asyncio.Task] = set()


def _build_graph(adapters: RunAdapters):
    nodes = {
        "research": make_research_node(
            website_scraper=adapters.website_scraper,
            instagram_reader=adapters.instagram_reader,
            search_tool=adapters.search_tool,
            run_manager=run_manager,
        ),
        "synthetic": make_synthetic_resource_node(run_manager=run_manager),
        "business_analysis": make_business_analysis_node(
            llm=adapters.llm, run_manager=run_manager
        ),
        "strategist": make_strategist_node(llm=adapters.llm, run_manager=run_manager),
        "planner": make_planner_node(llm=adapters.llm, run_manager=run_manager),
        "copywriter": make_copywriter_node(llm=adapters.llm, run_manager=run_manager),
        "creative": make_creative_node(
            image_generator=adapters.image_generator, run_manager=run_manager
        ),
        "video_director": make_video_director_node(
            llm=adapters.llm,
            video_generator=adapters.video_generator,
            run_manager=run_manager,
        ),
        "qa": make_qa_node(
            budget_ledger=adapters.budget_ledger, llm=adapters.llm, run_manager=run_manager
        ),
        "packaging": make_packaging_node(
            packaging_service=adapters.packaging_service, run_manager=run_manager
        ),
    }
    builder = GraphBuilder(entry_point="research").add_nodes(nodes)
    for source, target in zip(_NODE_ORDER, _NODE_ORDER[1:]):
        builder.add_edge(source, target)
    builder.add_edge(_NODE_ORDER[-1], END)
    return builder.compile()


async def _close_adapters(adapters: RunAdapters) -> None:
    for adapter in (
        adapters.llm,
        adapters.image_generator,
        adapters.website_scraper,
        adapters.instagram_reader,
    ):
        aclose = getattr(adapter, "aclose", None)
        if aclose is not None:
            await aclose()


def _package_path(handle: RunHandle):
    return run_manager.run_dir(handle.run_id) / "package" / "campaign_package.json"


async def _execute_run(
    handle: RunHandle, adapters: RunAdapters, initial_state: MarketingState
) -> None:
    graph = _build_graph(adapters)
    try:
        final = await graph.ainvoke(initial_state)
        campaign_package = (
            final["campaign_package"]
            if isinstance(final, dict)
            else final.campaign_package
        )
        if campaign_package is None:
            raise WorkflowExecutionError(
                "Graph completed without producing a campaign package."
            )
        path = _package_path(handle)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(campaign_package.model_dump_json(indent=2), encoding="utf-8")
        run_manager.complete_run(handle)
    except Exception as exc:  # noqa: BLE001 - recorded on the run, never swallowed
        run_manager.fail_run(handle, error=str(exc))
    finally:
        await _close_adapters(adapters)


def _load_handle(run_id: str) -> RunHandle:
    try:
        return run_manager.load_run(UUID(run_id))
    except (ValueError, WorkflowExecutionError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/runs")
async def create_run(request: CreateRunRequest) -> dict[str, str]:
    handle, adapters = build_run_dependencies(max_budget=request.budget_usd)

    source_pack: dict[str, str] = {"website_url": str(request.website_url)}
    if request.business_name is not None:
        source_pack["business_name"] = request.business_name
    if request.instagram_username is not None:
        source_pack["instagram_username"] = request.instagram_username

    initial_state = MarketingState(
        run_id=handle.run_id,
        workflow_id="marketingos_first_week_campaign",
        budget=BudgetState(
            cost_ledger=handle.guard.ledger, total_budget=request.budget_usd
        ),
        source_pack=source_pack,
    )

    task = asyncio.create_task(_execute_run(handle, adapters, initial_state))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return {"run_id": str(handle.run_id), "status": "running"}


@app.get("/runs/{run_id}", response_model=RunStatusResponse)
async def get_run_status(run_id: str) -> RunStatusResponse:
    handle = _load_handle(run_id)
    return RunStatusResponse(
        run_id=str(handle.record.run_id),
        status=handle.record.status.value,
        error=handle.record.error,
    )


def main() -> None:
    """Entrypoint for the ``marketingos`` console script (see pyproject.toml).

    Binds ``0.0.0.0`` and reads ``$PORT`` — the convention most PaaS hosts
    (Railway, Render, Fly, Heroku) use to tell a process which port to
    listen on, injected at deploy time rather than hardcoded here.
    """
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
