"""Routes serving a completed run's package: manifest and staged assets.

``GET /runs/{run_id}/package`` returns the full ``CampaignPackage`` JSON
(manifest, metadata, and — critically — ``asset_index``, so a frontend can
enumerate every generated image/video and its ``packaged_path``).
``GET /runs/{run_id}/assets/{path}`` then serves one staged file's bytes by
that path, and ``GET /runs/{run_id}/archive`` streams the whole campaign as
a zip. Whichever provider generated an asset (a local placeholder today, a
real Together AI/Gemini image once billing is live), these routes serve
whatever bytes exist on disk — a frontend built against this API needs no
change when the underlying provider switches.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast
from uuid import UUID

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from marketingos.api.dependencies import run_manager
from marketingos.exceptions.workflow import WorkflowExecutionError
from marketingos.models.run import RunStatus
from marketingos.services.run_manager import RunHandle

__all__ = ["router"]

router = APIRouter(tags=["outputs"])


def _load_handle(run_id: str) -> RunHandle:
    try:
        return run_manager.load_run(UUID(run_id))
    except (ValueError, WorkflowExecutionError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _require_completed(run_id: str) -> RunHandle:
    """Load ``run_id``'s handle, rejecting any run that isn't finished.

    A package only exists once packaging has run, so any other status
    (``running``, ``failed``, ...) is a 409, not a 404 — the run is real,
    just not ready yet.
    """
    handle = _load_handle(run_id)
    if handle.record.status is not RunStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail=f"Run {run_id} is {handle.record.status.value}, not completed.",
        )
    return handle


def _package_dir(handle: RunHandle) -> Path:
    return run_manager.run_dir(handle.run_id) / "package"


def _load_package(handle: RunHandle) -> dict[str, object]:
    path = _package_dir(handle) / "campaign_package.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Package artifact not found.")
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))


@router.get("/runs/{run_id}/package")
async def get_run_package(run_id: str) -> dict[str, object]:
    """Return the full campaign package: manifest, metadata, and asset index.

    ``asset_index`` entries carry the ``packaged_path`` each asset was
    written under, relative to this run's package directory — pass that
    straight through to ``GET /runs/{run_id}/assets/{packaged_path}`` to
    fetch the bytes.
    """
    handle = _require_completed(run_id)
    return _load_package(handle)


@router.get("/runs/{run_id}/assets/{asset_path:path}")
async def get_run_asset(run_id: str, asset_path: str) -> FileResponse:
    """Serve one staged asset's bytes by its ``packaged_path``.

    Resolves ``asset_path`` against this run's package directory and
    rejects anything that resolves outside it (a ``..``-laden path, or one
    escaping via a symlink) before ever opening the file.
    """
    handle = _require_completed(run_id)
    base = _package_dir(handle).resolve()
    target = (base / asset_path).resolve()
    if not target.is_relative_to(base):
        raise HTTPException(status_code=400, detail="Invalid asset path.")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Asset not found.")
    return FileResponse(target)


@router.get("/runs/{run_id}/archive")
async def get_run_archive(run_id: str) -> FileResponse:
    """Stream the full campaign package as a single zip download."""
    handle = _require_completed(run_id)
    package = _load_package(handle)
    archive = package["archive"]
    assert isinstance(archive, dict)
    archive_path = Path(str(archive["uri"]))
    if not archive_path.is_file():
        raise HTTPException(status_code=404, detail="Archive not found.")
    return FileResponse(
        archive_path,
        media_type=str(archive.get("media_type", "application/zip")),
        filename=f"{run_id}-campaign.zip",
    )
