"""TranscriptLoader — reads transcript JSON into a ``TranscriptBlob``.

Accepted shapes:

  Shape A — Whisper-style segments:
    {"segments": [{"start": 0.0, "end": 4.2, "text": "Hello.", "speaker": "spk_0"}, ...]}

  Shape B — already-canonical TranscriptStamp list:
    {"transcripts": [{"start_timestamp_in_seconds": 0.0,
                      "speaker": "spk_0", "text": "Hello."}, ...]}

  Shape C — plain list of stamps (either of the above field names per row):
    [{"start": 0.0, "speaker": "spk_0", "text": "Hello."}, ...]

VTT / SRT parsing isn't here — those are separate enough to deserve
their own loader if a user surfaces. For now, JSON only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from .blob import TranscriptBlob, TranscriptStamp
from .interface import LoaderInterface


def _row_to_stamp(row: dict[str, Any]) -> TranscriptStamp | None:
    text = (row.get("text") or "").strip()
    if not text:
        return None
    start = row.get("start_timestamp_in_seconds", row.get("start", 0.0))
    speaker = row.get("speaker", "")
    return TranscriptStamp(
        text=text,
        start_timestamp_in_seconds=float(start or 0.0),
        speaker=str(speaker or ""),
    )


class TranscriptLoader(LoaderInterface):
    loader_name: ClassVar[str] = "transcript"
    supported_extensions: ClassVar[tuple[str, ...]] = (
        "transcript.json", "whisper.json",
    )
    supported_mime_types: ClassVar[tuple[str, ...]] = (
        "application/x-transcript+json",
    )

    @classmethod
    def can_handle(cls, extension: str = "", mime_type: str = "") -> bool:
        if super().can_handle(extension=extension, mime_type=mime_type):
            return True
        ext = extension.lower().lstrip(".") if extension else ""
        return ext.endswith("transcript.json") or ext.endswith("whisper.json")

    async def load(self, file_path: str | Path, **kwargs) -> TranscriptBlob:
        path = Path(file_path)
        with open(path) as f:
            data = json.load(f)

        rows: list[dict] = []
        if isinstance(data, dict):
            if "transcripts" in data and isinstance(data["transcripts"], list):
                rows = data["transcripts"]
            elif "segments" in data and isinstance(data["segments"], list):
                rows = data["segments"]
            else:
                raise ValueError(
                    f"TranscriptLoader: {path} doesn't match expected shape "
                    "(no 'transcripts' or 'segments' key)"
                )
        elif isinstance(data, list):
            rows = data
        else:
            raise ValueError(
                f"TranscriptLoader: {path} root is not a list or object"
            )

        stamps: list[TranscriptStamp] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            stamp = _row_to_stamp(row)
            if stamp is not None:
                stamps.append(stamp)

        raw_text = "\n".join(
            f"[{s.speaker or '?'} @ {s.start_timestamp_in_seconds:.1f}s] {s.text}"
            for s in stamps
        )
        return TranscriptBlob(
            source_path=str(path),
            raw_text=raw_text,
            transcripts=stamps,
        )
