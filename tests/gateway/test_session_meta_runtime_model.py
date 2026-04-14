from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


@pytest.mark.asyncio
async def test_new_session_meta_uses_runtime_model_not_config_default(monkeypatch):
    runner = gateway_run.GatewayRunner.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {}
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    runner.delivery_router = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()
    runner._session_db = None
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._service_tier = None
    runner._show_reasoning = False
    runner._smart_model_routing = {}
    runner._clear_session_env = lambda _tokens: None
    runner._set_session_env = lambda _context: []
    runner._should_send_voice_reply = lambda *args, **kwargs: False

    session_entry = SimpleNamespace(
        session_id="session-1",
        session_key="agent:main:telegram:dm:user-1",
        created_at=datetime(2026, 4, 14, 10, 9, 0),
        updated_at=datetime(2026, 4, 14, 10, 9, 0),
        was_auto_reset=False,
        last_prompt_tokens=0,
    )

    appended = []
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.update_session = MagicMock()
    runner.session_store.append_to_transcript.side_effect = (
        lambda session_id, message, skip_db=False: appended.append((session_id, message, skip_db))
    )

    monkeypatch.setattr(gateway_run, "build_session_context", lambda source, config, entry: {})
    monkeypatch.setattr(gateway_run, "build_session_context_prompt", lambda context, redact_pii=False: "")
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")

    runner._prepare_inbound_message_text = AsyncMock(return_value="hello")
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "hi",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
            "history_offset": 0,
            "tools": [{"type": "function", "function": {"name": "terminal"}}],
            "last_prompt_tokens": 123,
            "model": "gpt-5.4-2026-03-05",
        }
    )

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_name="Chat",
        chat_type="dm",
        user_id="user-1",
        user_name="User",
    )
    event = MessageEvent(text="hello", message_type=MessageType.TEXT, source=source)

    runtime_model = "gpt-5.4-2026-03-05"
    monkeypatch.setattr(
        gateway_run.GatewayRunner,
        "_resolve_session_agent_runtime",
        lambda self, **kwargs: (runtime_model, {
            "provider": "custom",
            "api_key": "***",
            "base_url": "https://tokenx24.com/v1",
            "api_mode": "openai",
        }),
    )

    result = await runner._handle_message_with_agent(event, source, session_entry.session_key)

    assert result == "hi"
    session_meta = next(msg for _, msg, _ in appended if msg.get("role") == "session_meta")
    assert session_meta["model"] == runtime_model


@pytest.mark.asyncio
async def test_new_session_meta_falls_back_to_config_model_when_agent_result_has_no_model(monkeypatch):
    runner = gateway_run.GatewayRunner.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {}
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    runner.delivery_router = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()
    runner._session_db = None
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._service_tier = None
    runner._show_reasoning = False
    runner._smart_model_routing = {}
    runner._clear_session_env = lambda _tokens: None
    runner._set_session_env = lambda _context: []
    runner._should_send_voice_reply = lambda *args, **kwargs: False

    session_entry = SimpleNamespace(
        session_id="session-2",
        session_key="agent:main:telegram:dm:user-1",
        created_at=datetime(2026, 4, 14, 10, 9, 0),
        updated_at=datetime(2026, 4, 14, 10, 9, 0),
        was_auto_reset=False,
        last_prompt_tokens=0,
    )

    appended = []
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.update_session = MagicMock()
    runner.session_store.append_to_transcript.side_effect = (
        lambda session_id, message, skip_db=False: appended.append((session_id, message, skip_db))
    )

    monkeypatch.setattr(gateway_run, "build_session_context", lambda source, config, entry: {})
    monkeypatch.setattr(gateway_run, "build_session_context_prompt", lambda context, redact_pii=False: "")
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")

    runner._prepare_inbound_message_text = AsyncMock(return_value="hello")
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "hi",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
            "history_offset": 0,
            "tools": [{"type": "function", "function": {"name": "terminal"}}],
            "last_prompt_tokens": 123,
            "model": "",
        }
    )

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_name="Chat",
        chat_type="dm",
        user_id="user-1",
        user_name="User",
    )
    event = MessageEvent(text="hello", message_type=MessageType.TEXT, source=source)

    monkeypatch.setattr(
        gateway_run.GatewayRunner,
        "_resolve_session_agent_runtime",
        lambda self, **kwargs: ("gpt-5.4-2026-03-05", {
            "provider": "custom",
            "api_key": "***",
            "base_url": "https://tokenx24.com/v1",
            "api_mode": "openai",
        }),
    )

    result = await runner._handle_message_with_agent(event, source, session_entry.session_key)

    assert result == "hi"
    session_meta = next(msg for _, msg, _ in appended if msg.get("role") == "session_meta")
    assert session_meta["model"] == "gpt-5.4"