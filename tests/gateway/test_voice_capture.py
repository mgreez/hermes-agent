from __future__ import annotations

import subprocess
from datetime import datetime, timezone

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.voice_capture import (
    CaptureDecision,
    capture_config_from_mapping,
    detect_capture_trigger,
    is_channel_enabled,
    maybe_enqueue_discord_voice_capture,
)


def test_no_trigger_phrase_does_not_enqueue(tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    decision = maybe_enqueue_discord_voice_capture(
        message_id="123",
        created_at="2026-05-18T12:00:00Z",
        transcript="this is just a reflection with no durable intent",
        content=None,
        queue_script=str(tmp_path / "missing.py"),
        run=fake_run,
    )

    assert decision == CaptureDecision(False, False, "no_trigger")
    assert calls == []


def test_trigger_phrase_builds_payload_and_uses_queue_script(tmp_path):
    script = tmp_path / "obsidian-capture-queue.py"
    script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='{"ok": true, "event_id": "discord-123", "queue_path": "/tmp/item.json"}',
            stderr="",
        )

    decision = maybe_enqueue_discord_voice_capture(
        message_id="123",
        created_at=datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc),
        transcript="Capture this: v2 always-on Discord voice test",
        content="",
        queue_script=str(script),
        python_executable="python-test",
        run=fake_run,
    )

    assert decision.enqueued is True
    assert decision.event_id == "discord-123"
    assert decision.queue_path == "/tmp/item.json"
    assert calls[0][0] == ["python-test", str(script), "enqueue"]
    assert '"event_id": "discord-123"' in calls[0][1]["input"]
    assert '"source": "discord-voice-note"' in calls[0][1]["input"]
    assert "v2 always-on Discord voice test" in calls[0][1]["input"]


def test_caption_can_supply_trigger_phrase(tmp_path):
    script = tmp_path / "queue.py"
    script.write_text("", encoding="utf-8")

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    decision = maybe_enqueue_discord_voice_capture(
        message_id="456",
        created_at="2026-05-18T12:00:00Z",
        transcript="raw words I want preserved",
        content="save this to Obsidian",
        queue_script=str(script),
        run=fake_run,
    )

    assert decision.enqueued is True
    assert decision.trigger_phrase == "save this to obsidian"


def test_large_or_empty_transcript_skips_enqueue(tmp_path):
    calls = []
    common = {
        "message_id": "123",
        "created_at": "2026-05-18T12:00:00Z",
        "content": "capture this",
        "queue_script": str(tmp_path / "queue.py"),
        "run": lambda *a, **k: calls.append((a, k)),
    }

    empty = maybe_enqueue_discord_voice_capture(transcript="  ", **common)
    large = maybe_enqueue_discord_voice_capture(
        transcript="capture this " + ("x" * 20),
        max_transcript_bytes=10,
        **common,
    )

    assert empty.reason == "empty_transcript"
    assert large.reason == "transcript_too_large"
    assert calls == []


def test_enqueue_failure_returns_safe_metadata_without_raw_transcript(tmp_path):
    script = tmp_path / "queue.py"
    script.write_text("", encoding="utf-8")
    secretish_transcript = "capture this: API key abc123 should not appear in decision"

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 2, stdout="", stderr=secretish_transcript)

    decision = maybe_enqueue_discord_voice_capture(
        message_id="789",
        created_at="2026-05-18T12:00:00Z",
        transcript=secretish_transcript,
        content=None,
        queue_script=str(script),
        run=fake_run,
    )

    assert decision.attempted is True
    assert decision.enqueued is False
    assert decision.reason == "enqueue_failed"
    assert decision.returncode == 2
    assert secretish_transcript not in repr(decision)


def test_config_defaults_off_and_channel_matching(monkeypatch):
    monkeypatch.delenv("HERMES_DISCORD_VOICE_CAPTURE_QUEUE_ENABLED", raising=False)
    cfg = capture_config_from_mapping({})
    assert cfg["enabled"] is False
    assert is_channel_enabled("1505333822944972842", []) is False

    cfg = capture_config_from_mapping(
        {
            "enabled": True,
            "channels": ["1505333822944972842"],
            "queue_script": "/tmp/queue.py",
        }
    )
    assert cfg["enabled"] is True
    assert cfg["queue_script"] == "/tmp/queue.py"
    assert is_channel_enabled("1505333822944972842", cfg["channels"]) is True
    assert is_channel_enabled("other", cfg["channels"]) is False


def test_detect_capture_trigger_normalizes_whitespace_and_case():
    assert detect_capture_trigger(transcript="  CAPTURE   this please ") == "capture this"
    assert detect_capture_trigger(transcript="Please save the raw capture after the follow-up") == "save the raw capture"
    assert detect_capture_trigger(transcript="recapture this moment") is None


def test_channel_wildcard_matches_any_channel():
    assert is_channel_enabled("anything", ["*"]) is True


def test_negated_capture_phrases_do_not_trigger():
    assert detect_capture_trigger(transcript="do not capture this private aside") is None
    assert detect_capture_trigger(transcript="please don't save the raw capture") is None
    assert detect_capture_trigger(transcript="never save this to Obsidian") is None
    assert (
        detect_capture_trigger(transcript="do not capture this part. Actually, capture this follow-up")
        == "capture this"
    )


@pytest.mark.asyncio
async def test_gateway_voice_capture_disabled_config_still_returns_transcription(monkeypatch, tmp_path):
    calls = []
    runner = _make_runner(enabled=False, channels=["voice-channel"], queue_script=str(tmp_path / "queue.py"))
    event = _make_event(platform=Platform.DISCORD, chat_id="voice-channel")

    monkeypatch.setattr(
        "tools.transcription_tools.transcribe_audio",
        lambda path: {"success": True, "transcript": "capture this disabled config"},
    )
    monkeypatch.setattr(
        "gateway.voice_capture.maybe_enqueue_discord_voice_capture",
        lambda **kwargs: calls.append(kwargs) or CaptureDecision(True, True, "enqueued"),
    )

    enriched = await runner._enrich_message_with_transcription("", ["voice.ogg"], event=event)

    assert "capture this disabled config" in enriched
    assert calls == []


@pytest.mark.asyncio
async def test_gateway_voice_capture_wrong_channel_and_non_discord_skip_enqueue(monkeypatch, tmp_path):
    calls = []
    runner = _make_runner(enabled=True, channels=["voice-channel"], queue_script=str(tmp_path / "queue.py"))
    wrong_channel = _make_event(platform=Platform.DISCORD, chat_id="other-channel")
    non_discord = _make_event(platform=Platform.TELEGRAM, chat_id="voice-channel")

    monkeypatch.setattr(
        "tools.transcription_tools.transcribe_audio",
        lambda path: {"success": True, "transcript": "capture this but not from an allowed source"},
    )
    monkeypatch.setattr(
        "gateway.voice_capture.maybe_enqueue_discord_voice_capture",
        lambda **kwargs: calls.append(kwargs) or CaptureDecision(True, True, "enqueued"),
    )

    wrong_channel_text = await runner._enrich_message_with_transcription("", ["wrong.ogg"], event=wrong_channel)
    non_discord_text = await runner._enrich_message_with_transcription("", ["telegram.ogg"], event=non_discord)

    assert "capture this but not from an allowed source" in wrong_channel_text
    assert "capture this but not from an allowed source" in non_discord_text
    assert calls == []


@pytest.mark.asyncio
async def test_gateway_enqueue_failure_does_not_break_conversational_transcription(monkeypatch, tmp_path):
    runner = _make_runner(enabled=True, channels=["voice-channel"], queue_script=str(tmp_path / "queue.py"))
    event = _make_event(platform=Platform.DISCORD, chat_id="voice-channel", message_id="msg-fail")

    monkeypatch.setattr(
        "tools.transcription_tools.transcribe_audio",
        lambda path: {"success": True, "transcript": "capture this despite queue failure"},
    )
    monkeypatch.setattr(
        "gateway.voice_capture.maybe_enqueue_discord_voice_capture",
        lambda **kwargs: CaptureDecision(True, False, "enqueue_failed", event_id="discord-msg-fail", returncode=2),
    )

    enriched = await runner._enrich_message_with_transcription("caption", ["voice.ogg"], event=event)

    assert "matched explicit capture intent" in enriched
    assert "failed safely (enqueue_failed)" in enriched
    assert "raw transcript is intentionally withheld" in enriched
    assert "capture this despite queue failure" not in enriched
    assert "\n\ncaption" not in enriched
    assert "caption/content is intentionally withheld" in enriched


@pytest.mark.asyncio
async def test_gateway_multiple_audio_attachments_attempt_capture_once(monkeypatch, tmp_path):
    calls = []
    runner = _make_runner(enabled=True, channels=["voice-channel"], queue_script=str(tmp_path / "queue.py"))
    event = _make_event(platform=Platform.DISCORD, chat_id="voice-channel", message_id="msg-dupe")
    transcripts = iter(["capture this first attachment", "capture this second attachment"])

    monkeypatch.setattr(
        "tools.transcription_tools.transcribe_audio",
        lambda path: {"success": True, "transcript": next(transcripts)},
    )
    monkeypatch.setattr(
        "gateway.voice_capture.maybe_enqueue_discord_voice_capture",
        lambda **kwargs: calls.append(kwargs) or CaptureDecision(
            True,
            True,
            "enqueued",
            event_id="discord-msg-dupe",
            trigger_phrase="capture this",
        ),
    )

    enriched = await runner._enrich_message_with_transcription("", ["one.ogg", "two.ogg"], event=event)

    assert "matched explicit capture intent" in enriched
    assert "additional Discord voice attachment" in enriched
    assert "first attachment" not in enriched
    assert "second attachment" not in enriched
    assert len(calls) == 1
    assert calls[0]["message_id"] == "msg-dupe"
    assert calls[0]["transcript"] == "capture this first attachment"


@pytest.mark.asyncio
async def test_gateway_voice_capture_uses_stable_synthetic_message_id(monkeypatch, tmp_path):
    calls = []
    runner = _make_runner(enabled=True, channels=["voice-channel"], queue_script=str(tmp_path / "queue.py"))
    event = _make_event(platform=Platform.DISCORD, chat_id="voice-channel", message_id=None)

    monkeypatch.setattr(
        "tools.transcription_tools.transcribe_audio",
        lambda path: {"success": True, "transcript": "capture this stable retry"},
    )
    monkeypatch.setattr(
        "gateway.voice_capture.maybe_enqueue_discord_voice_capture",
        lambda **kwargs: calls.append(kwargs) or CaptureDecision(
            True,
            True,
            "enqueued",
            event_id=f"discord-{kwargs['message_id']}",
            trigger_phrase="capture this",
        ),
    )

    await runner._enrich_message_with_transcription("", ["voice.ogg"], event=event)
    await runner._enrich_message_with_transcription("", ["voice.ogg"], event=event)

    assert len(calls) == 2
    assert calls[0]["message_id"].startswith("synthetic-")
    assert calls[0]["message_id"] == calls[1]["message_id"]


def _make_runner(*, enabled: bool, channels: list[str], queue_script: str) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        stt_enabled=True,
        platforms={
            Platform.DISCORD: PlatformConfig(
                extra={
                    "voice_capture_queue": {
                        "enabled": enabled,
                        "channels": channels,
                        "queue_script": queue_script,
                        "python_executable": "python-test",
                    }
                }
            )
        },
    )
    runner._has_setup_skill = lambda: False
    return runner


def _make_event(*, platform: Platform, chat_id: str, message_id: str | None = "msg-123") -> MessageEvent:
    return MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=SessionSource(platform=platform, chat_id=chat_id),
        message_id=message_id,
        timestamp=datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc),
    )
