"""Tests for the run-output routes: package manifest and asset serving.

Exercises the API layer end-to-end via FastAPI's ``TestClient``, against a
run manufactured directly through ``RunManager`` (no pipeline execution
needed) — the routes only care that a completed run's directory and
``campaign_package.json`` exist.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import marketingos.api.routes.outputs as outputs_module
from marketingos.api.main import app
from marketingos.services.run_manager import RunHandle, RunManager

client = TestClient(app)


def _minimal_package(*, run_id: str, asset_relpath: str) -> dict[str, object]:
    """A structurally minimal package JSON with one asset and an archive ref."""
    return {
        "run_id": run_id,
        "subject": "Test Subject",
        "root_path": f"runs/{run_id}",
        "manifest": {
            "schema_version": "1.0",
            "package_run_id": run_id,
            "subject": "Test Subject",
            "source_context_run_id": "ctx",
            "source_strategy_run_id": "strat",
            "source_plan_run_id": "plan",
            "source_caption_run_id": "cap",
            "source_creative_run_id": "creative",
            "source_video_run_id": "video",
            "qa_run_id": "qa",
            "media_asset_count": 1,
            "document_count": 0,
            "created_at": "2026-01-01T00:00:00Z",
        },
        "metadata": {
            "generator": "MarketingOS",
            "subject": "Test Subject",
            "qa_status": "passed",
            "qa_run_id": "qa",
            "post_count": 1,
            "video_count": 0,
            "created_at": "2026-01-01T00:00:00Z",
        },
        "asset_index": [
            {
                "asset_id": "img_1",
                "kind": "image",
                "item_id": "C1",
                "source_uri": None,
                "packaged_path": asset_relpath,
                "media_type": "image/png",
                "size_bytes": 4,
                "checksum": {"algorithm": "sha256", "value": "0" * 64},
            }
        ],
        "readme": "A test package.",
        "archive": {
            "uri": "",  # filled in by the fixture once the zip path is known
            "size_bytes": 4,
            "checksum": {"algorithm": "sha256", "value": "0" * 64},
            "media_type": "application/zip",
        },
        "created_at": "2026-01-01T00:00:00Z",
    }


@pytest.fixture
def completed_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RunHandle:
    """A completed run, with a real staged image and a package manifest on disk."""
    manager = RunManager(runs_root=tmp_path)
    monkeypatch.setattr(outputs_module, "run_manager", manager)

    handle = manager.start_run(max_budget=Decimal("100"))
    manager.complete_run(handle)

    package_dir = manager.run_dir(handle.run_id) / "package"
    package_dir.mkdir(parents=True, exist_ok=True)

    asset_relpath = "assets/images/C1.png"
    asset_path = package_dir / asset_relpath
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    asset_path.write_bytes(b"\x89PNG")

    archive_path = package_dir / "campaign.zip"
    archive_path.write_bytes(b"PK\x03\x04")

    package = _minimal_package(run_id=str(handle.run_id), asset_relpath=asset_relpath)
    package["archive"]["uri"] = str(archive_path)  # type: ignore[index]
    (package_dir / "campaign_package.json").write_text(
        json.dumps(package), encoding="utf-8"
    )

    return handle


# ---------------------------------------------------------------------------
# GET /runs/{run_id}/package
# ---------------------------------------------------------------------------


def test_package_route_returns_full_manifest_including_asset_index(
    completed_run: RunHandle,
) -> None:
    """The route must expose asset_index, not just the archive reference."""
    response = client.get(f"/runs/{completed_run.run_id}/package")

    assert response.status_code == 200
    body = response.json()
    assert "asset_index" in body
    assert body["asset_index"][0]["packaged_path"] == "assets/images/C1.png"
    assert "archive" in body


def test_package_route_404s_for_unknown_run() -> None:
    response = client.get("/runs/00000000-0000-0000-0000-000000000000/package")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /runs/{run_id}/assets/{path}
# ---------------------------------------------------------------------------


def test_asset_route_serves_the_real_file(completed_run: RunHandle) -> None:
    response = client.get(f"/runs/{completed_run.run_id}/assets/assets/images/C1.png")

    assert response.status_code == 200
    assert response.content == b"\x89PNG"


def test_asset_route_rejects_path_traversal(completed_run: RunHandle) -> None:
    response = client.get(
        f"/runs/{completed_run.run_id}/assets/../../../../etc/passwd"
    )

    assert response.status_code in (400, 404)
    assert response.content != b"\x89PNG"


def test_asset_route_404s_for_missing_file(completed_run: RunHandle) -> None:
    response = client.get(f"/runs/{completed_run.run_id}/assets/does/not/exist.png")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /runs/{run_id}/archive
# ---------------------------------------------------------------------------


def test_archive_route_streams_the_zip(completed_run: RunHandle) -> None:
    response = client.get(f"/runs/{completed_run.run_id}/archive")

    assert response.status_code == 200
    assert response.content == b"PK\x03\x04"
    assert response.headers["content-type"] == "application/zip"
