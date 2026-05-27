"""Explicit voice-capture producer helpers for the gateway.

This module intentionally contains no platform SDK imports.  It turns a
post-STT Discord event into a minimal stdin JSON payload for the existing
homelab Obsidian capture queue script.  Capture is opt-in at two levels:
configuration must enable it, and the transcript/caption must contain a narrow
explicit capture phrase.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


DEFAULT_TRIGGER_PHRASES: tuple[str, ...] = (
    "capture this",
    "save this to obsidian",
    "save the raw capture",
    "save raw capture",
    "put this in today's note",
    "put this in todays note",
    "page up today's daily note",
    "page up todays daily note",
    "add this to my daily note",
)
DEFAULT_SOURCE = "discord-voice-note"
DEFAULT_MAX_TRANSCRIPT_BYTES = 64 * 1024


@dataclass(frozen=True)
class CaptureDecision:
    """Outcome of an explicit voice-capture enqueue attempt."""

    attempted: bool
    enqueued: bool
    reason: str
    event_id: str | None = None
    trigger_phrase: str | None = None
    returncode: int | None = None
    queue_path: str | None = None


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off", ""}:
            return False
    return default


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(part).strip() for part in value if str(part).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip().lower()


_NEGATED_CAPTURE_PREFIX_RE = re.compile(
    r"(?:^|\b)(?:do\s+not|don['’]?t|dont|never|no)\s+$",
)


def _is_negated_trigger_match(haystack: str, match: re.Match[str]) -> bool:
    """Return True when a trigger match is immediately negated.

    The capture producer is intentionally opt-in.  A short, local look-behind
    catches natural phrases such as "do not capture this" without adding broad
    NLP or trying to infer intent from distant context.
    """

    prefix = haystack[max(0, match.start() - 32) : match.start()]
    prefix = re.sub(r"\s+", " ", prefix).lower()
    return bool(_NEGATED_CAPTURE_PREFIX_RE.search(prefix))


def detect_capture_trigger(
    *,
    transcript: str,
    content: str | None = None,
    trigger_phrases: Sequence[str] = DEFAULT_TRIGGER_PHRASES,
) -> str | None:
    """Return the matching explicit capture phrase, if present.

    Matching is intentionally simple and narrow.  It searches normalized
    transcript plus optional caption/content text, so Discord voice notes with
    either spoken or typed intent can trigger a capture.  Immediately negated
    trigger phrases are ignored.
    """

    haystack = _normalize_text("\n".join(part for part in (transcript, content or "") if part))
    if not haystack:
        return None
    for phrase in trigger_phrases:
        normalized = _normalize_text(str(phrase))
        if not normalized:
            continue
        pattern = r"(?<!\w)" + r"\s+".join(re.escape(part) for part in normalized.split()) + r"(?!\w)"
        for match in re.finditer(pattern, haystack):
            if not _is_negated_trigger_match(haystack, match):
                return phrase
    return None


def capture_config_from_mapping(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize the ``voice_capture_queue`` config block.

    Environment variables are supported as a conservative prototype/deployment
    path and override config values only when set.
    """

    cfg = dict(config or {})
    env_enabled = os.getenv("HERMES_DISCORD_VOICE_CAPTURE_QUEUE_ENABLED")
    env_channels = os.getenv("HERMES_DISCORD_VOICE_CAPTURE_QUEUE_CHANNELS")
    env_script = os.getenv("HERMES_DISCORD_VOICE_CAPTURE_QUEUE_SCRIPT")
    env_python = os.getenv("HERMES_DISCORD_VOICE_CAPTURE_QUEUE_PYTHON")
    env_max_bytes = os.getenv("HERMES_DISCORD_VOICE_CAPTURE_MAX_TRANSCRIPT_BYTES")

    enabled = _coerce_bool(env_enabled, _coerce_bool(cfg.get("enabled"), False))
    channels = _coerce_list(env_channels if env_channels is not None else cfg.get("channels"))
    queue_script = str(env_script if env_script is not None else cfg.get("queue_script") or "").strip()
    python_executable = str(env_python if env_python is not None else cfg.get("python_executable") or sys.executable).strip()

    trigger_phrases = _coerce_list(cfg.get("trigger_phrases")) or list(DEFAULT_TRIGGER_PHRASES)
    try:
        max_bytes = int(env_max_bytes if env_max_bytes is not None else cfg.get("max_transcript_bytes") or DEFAULT_MAX_TRANSCRIPT_BYTES)
    except (TypeError, ValueError):
        max_bytes = DEFAULT_MAX_TRANSCRIPT_BYTES

    return {
        "enabled": enabled,
        "channels": channels,
        "queue_script": queue_script,
        "python_executable": python_executable or sys.executable,
        "trigger_phrases": trigger_phrases,
        "max_transcript_bytes": max(1, max_bytes),
    }


def is_channel_enabled(channel_id: str | int | None, channels: Sequence[str]) -> bool:
    """Return whether *channel_id* is in the enabled capture channel set."""

    if channel_id is None:
        return False
    normalized = str(channel_id).strip()
    allowed = {str(ch).strip() for ch in channels if str(ch).strip()}
    return "*" in allowed or normalized in allowed


def _iso_timestamp(value: Any) -> str:
    if value is None or value == "":
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    text = str(value).strip()
    return text or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def maybe_enqueue_discord_voice_capture(
    *,
    message_id: str,
    created_at: Any,
    transcript: str,
    content: str | None = None,
    queue_script: str,
    python_executable: str | None = None,
    trigger_phrases: Sequence[str] = DEFAULT_TRIGGER_PHRASES,
    max_transcript_bytes: int = DEFAULT_MAX_TRANSCRIPT_BYTES,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    timeout: float = 15.0,
    env: Mapping[str, str] | None = None,
) -> CaptureDecision:
    """Enqueue a Discord voice capture only when explicit capture intent exists.

    The raw transcript is sent only to the queue script via stdin.  Return values
    and logs should use metadata only; callers must not log the raw transcript on
    failures.
    """

    cleaned_transcript = (transcript or "").strip()
    if not cleaned_transcript:
        return CaptureDecision(False, False, "empty_transcript")

    transcript_bytes = len(cleaned_transcript.encode("utf-8"))
    if transcript_bytes > max_transcript_bytes:
        return CaptureDecision(False, False, "transcript_too_large")

    trigger = detect_capture_trigger(
        transcript=cleaned_transcript,
        content=content,
        trigger_phrases=trigger_phrases,
    )
    if not trigger:
        return CaptureDecision(False, False, "no_trigger")

    script = Path(queue_script).expanduser() if queue_script else None
    if script is None or not script.is_file():
        return CaptureDecision(True, False, "queue_script_missing")

    event_id = f"discord-{str(message_id).strip()}"
    payload = {
        "event_id": event_id,
        "source": DEFAULT_SOURCE,
        "captured_at": _iso_timestamp(created_at),
        "transcript": cleaned_transcript,
    }
    command = [python_executable or sys.executable, str(script), "enqueue"]

    try:
        completed = run(
            command,
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=timeout,
            env=dict(env) if env is not None else None,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CaptureDecision(True, False, "enqueue_timeout", event_id=event_id, trigger_phrase=trigger)
    except Exception:
        return CaptureDecision(True, False, "enqueue_exception", event_id=event_id, trigger_phrase=trigger)

    if completed.returncode != 0:
        return CaptureDecision(
            True,
            False,
            "enqueue_failed",
            event_id=event_id,
            trigger_phrase=trigger,
            returncode=completed.returncode,
        )

    queue_path = None
    try:
        parsed = json.loads(completed.stdout or "{}")
        queue_path = parsed.get("queue_path") if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    return CaptureDecision(
        True,
        True,
        "enqueued",
        event_id=event_id,
        trigger_phrase=trigger,
        returncode=completed.returncode,
        queue_path=queue_path,
    )
