"""Tests for the local, zero-cost placeholder video generator."""

from __future__ import annotations

from pathlib import Path

from marketingos.agents.video_director import Scene, Shot, SubtitleLine, VideoDirection
from marketingos.tools.video.placeholder_video_client import PlaceholderVideoClient


def _direction() -> VideoDirection:
    return VideoDirection(
        item_id="C6",
        script="A short script.",
        scenes=(
            Scene(
                number=1,
                description="Opening shot.",
                voice_over="Welcome.",
                shots=(
                    Shot(number=1, description="Wide shot.", framing="wide", duration_seconds=3.0),
                ),
            ),
            Scene(
                number=2,
                description="Closing shot.",
                voice_over="Thanks for watching.",
                shots=(
                    Shot(
                        number=1,
                        description="Close-up.",
                        framing="close-up",
                        duration_seconds=4.5,
                    ),
                ),
            ),
        ),
        subtitles=(SubtitleLine(start_seconds=0.0, end_seconds=3.0, text="Welcome."),),
    )


async def test_render_reports_planned_duration_not_stub_runtime(tmp_path: Path) -> None:
    """The returned duration must match the direction, not the stub file's own length."""
    client = PlaceholderVideoClient(output_dir=tmp_path)
    direction = _direction()

    ref = await client.render(direction=direction)

    assert ref.duration_seconds == direction.total_duration_seconds
    assert ref.duration_seconds == 7.5


async def test_render_writes_a_real_file(tmp_path: Path) -> None:
    """The returned uri must point to an actual file on disk."""
    client = PlaceholderVideoClient(output_dir=tmp_path)

    ref = await client.render(direction=_direction())

    assert Path(ref.uri).is_file()
    assert Path(ref.uri).parent == tmp_path
    assert ref.media_type == "video/mp4"


async def test_render_is_idempotent_across_calls(tmp_path: Path) -> None:
    """Each call gets its own asset id and file, even for the same direction."""
    client = PlaceholderVideoClient(output_dir=tmp_path)
    direction = _direction()

    first = await client.render(direction=direction)
    second = await client.render(direction=direction)

    assert first.asset_id != second.asset_id
    assert Path(first.uri) != Path(second.uri)
    assert Path(first.uri).is_file()
    assert Path(second.uri).is_file()
